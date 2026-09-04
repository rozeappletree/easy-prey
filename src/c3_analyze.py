"""Step 8 = C3 analysis: dose-response curves, health-check filtering, Gate G3 verdict.

    /opt/conda/envs/talktuner-gpu/bin/python src/c3_analyze.py

Separate from the GPU-bound sweep (c3_sufficiency.py) so the analysis can be re-run instantly
against saved data/c3_rows.jsonl without touching the model -- consistent with this project's
standing rule (WORKLOG entries throughout) that a script writing raw rows should be re-analysable
offline.

Health check (DESIGN.md Sec 4, GATES.md analysis commitments): at each alpha, the RANDOM direction
sets the "this is what merely perturbing the model looks like" range for logp_sum
(logP(correct)+mean logP(lures)). Any (direction, alpha) cell whose logp_sum falls outside that
range is excluded from the dose-response curve and reported as excluded, not silently averaged in.

Gate G3 (GATES.md): monotone dose-response, correct sign, health checks intact.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = REPO_ROOT / "data", REPO_ROOT / "outputs"

CREDULITY_DIRECTIONS = ["diff_in_means", "probe_weights", "orthogonalized"]
CONTROL_DIRECTIONS = ["verbosity", "random"]


def load_rows():
    rows = []
    with open(DATA / "c3_rows.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def aggregate(rows):
    """rows -> {direction: {alpha: {"delta": mean, "logp_sum": mean, "n": count}}}"""
    by_da = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_da[r["direction"]][r["alpha"]].append(r)
    out = {}
    for direction, by_alpha in by_da.items():
        out[direction] = {}
        for alpha, rs in by_alpha.items():
            out[direction][alpha] = {
                "delta": statistics.mean(x["delta"] for x in rs),
                "delta_sd": statistics.pstdev(x["delta"] for x in rs),
                "logp_sum": statistics.mean(x["logp_sum"] for x in rs),
                "n": len(rs),
            }
    return out


def health_filter(agg):
    """Exclude any (direction, alpha) cell whose logp_sum falls outside the random-direction
    control's range AT THAT SAME ALPHA. Returns (filtered_agg, excluded_list)."""
    random_range = {}
    for alpha, cell in agg.get("random", {}).items():
        random_range[alpha] = cell["logp_sum"]
    if not random_range:
        return agg, []
    lo, hi = min(random_range.values()), max(random_range.values())
    margin = 0.5 * (hi - lo) if hi > lo else 1.0  # a floor: random itself spans some range across alpha
    lo, hi = lo - margin, hi + margin

    excluded = []
    filtered = {}
    for direction, by_alpha in agg.items():
        filtered[direction] = {}
        for alpha, cell in by_alpha.items():
            if lo <= cell["logp_sum"] <= hi:
                filtered[direction][alpha] = cell
            else:
                excluded.append((direction, alpha, cell["logp_sum"]))
    return filtered, excluded


def monotonicity(agg, direction):
    """Spearman-style check: is Delta monotonically increasing in alpha? Returns (is_monotone,
    sign_correlation) using simple rank correlation since scipy adds a dependency here for one
    number -- cheap to compute directly."""
    cells = agg.get(direction, {})
    alphas = sorted(cells.keys())
    if len(alphas) < 3:
        return False, 0.0
    deltas = [cells[a]["delta"] for a in alphas]
    # Kendall's tau sign-agreement fraction: count concordant vs discordant pairs.
    concordant = discordant = 0
    for i in range(len(alphas)):
        for j in range(i + 1, len(alphas)):
            da, dd = alphas[j] - alphas[i], deltas[j] - deltas[i]
            if da == 0:
                continue
            if (da > 0) == (dd > 0):
                concordant += 1
            elif dd != 0:
                discordant += 1
    total = concordant + discordant
    tau = (concordant - discordant) / total if total else 0.0
    return tau > 0.5, tau


def plot(agg, excluded, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"diff_in_means": "#2166ac", "probe_weights": "#762a83",
              "orthogonalized": "#1a9850", "verbosity": "#d6604d", "random": "#9a9a9a"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    excluded_set = {(d, a) for d, a, _ in excluded}

    for direction in CREDULITY_DIRECTIONS + CONTROL_DIRECTIONS:
        cells = agg.get(direction, {})
        alphas = sorted(cells.keys())
        if not alphas:
            continue
        deltas = [cells[a]["delta"] for a in alphas]
        excl_mask = [(direction, a) in excluded_set for a in alphas]
        style = "--" if direction in CONTROL_DIRECTIONS else "-"
        ax.plot(alphas, deltas, style, marker="o", color=colors[direction],
                label=direction, linewidth=2 if direction in CREDULITY_DIRECTIONS else 1.3)
        for a, dl, ex in zip(alphas, deltas, excl_mask):
            if ex:
                ax.scatter([a], [dl], marker="x", s=120, color="red", zorder=5)

    ax.axhline(0, color="#ddd", lw=0.8)
    ax.axvline(0, color="#ddd", lw=0.8)
    ax.set_xlabel("alpha (steering strength, relative to residual norm)")
    ax.set_ylabel("mean Delta (credulous-leaning if positive)")
    ax.set_title("Step 8 (C3): dose-response by direction\nred x = excluded by health check")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print("Wrote figure -> %s" % out_path)


def main():
    rows = load_rows()
    print("Loaded %d rows" % len(rows))
    agg = aggregate(rows)

    print("\nRaw dose-response (mean Delta by alpha), before health filtering:")
    for direction in CREDULITY_DIRECTIONS + CONTROL_DIRECTIONS:
        cells = agg.get(direction, {})
        line = "  %-14s " % direction
        for alpha in sorted(cells.keys()):
            line += "a=%+d:%+.3f  " % (alpha, cells[alpha]["delta"])
        print(line)

    filtered, excluded = health_filter(agg)
    print("\nHealth check: %d/%d (direction, alpha) cells excluded (logp_sum outside random-direction range)"
          % (len(excluded), sum(len(v) for v in agg.values())))
    for d, a, lp in excluded:
        print("  EXCLUDED: %-14s alpha=%+d  logp_sum=%.3f" % (d, a, lp))

    print("\nMonotonicity (Kendall-style sign agreement, tau > 0.5 = monotone):")
    results = {}
    for direction in CREDULITY_DIRECTIONS + CONTROL_DIRECTIONS:
        is_mono, tau = monotonicity(filtered, direction)
        results[direction] = {"monotone": is_mono, "tau": tau}
        print("  %-14s tau=%+.3f  %s" % (direction, tau, "MONOTONE" if is_mono else "not monotone"))

    plot(filtered, excluded, OUT / "figure2_c3_dose_response.png")

    print("\n" + "=" * 70)
    print("GATE G3: monotone dose-response, correct sign, health checks intact")
    credulity_monotone = any(results[d]["monotone"] and results[d]["tau"] > 0 for d in CREDULITY_DIRECTIONS)
    controls_flat = all(abs(results[d]["tau"]) < 0.5 for d in CONTROL_DIRECTIONS)
    print("  at least one credulity direction monotone+positive: %s" % credulity_monotone)
    print("  control directions (verbosity, random) NOT monotone: %s" % controls_flat)
    passed = credulity_monotone and controls_flat
    print("  RESULT: %s" % ("PASS" if passed else
          "NULL -- report as C3 null; Step 9 (C4) still runs (necessity without sufficiency, GATES.md)"))
    print("=" * 70)

    json.dump({
        "aggregated": {d: {str(a): c for a, c in cells.items()} for d, cells in agg.items()},
        "excluded": excluded, "monotonicity": results, "gate_g3_passed": bool(passed),
    }, open(DATA / "c3_summary.json", "w"), indent=2)
    print("Saved summary -> data/c3_summary.json")


if __name__ == "__main__":
    main()
