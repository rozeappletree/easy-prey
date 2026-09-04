"""Merge the Step 2 top-up batch into the main prefix set, re-run QC, and report.

See docs/WORKLOG.md entry 27. Triggered because the first 462-conversation batch matched to only
73/class -- below both the 80/class fallback floor and the 100/class top-up threshold in GATES.md.

    /opt/conda/envs/talktuner-gpu/bin/python src/merge_topup.py

Renumbers prefix_id sequentially per label across the merge (the two source files both start
counting from 0, so IDs collide) and asserts no duplicate raw_text survived the merge -- the same
uniqueness check the toy run's analysis notebook applied to its 80 conversations.
"""

import json
from pathlib import Path

from prefix_qc import gate_a, gate_b, is_usable, length_match, marker_table

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

MAIN = DATA_DIR / "prefixes_v2.json"
TOPUP = DATA_DIR / "prefixes_v2_topup.json"
OUT = DATA_DIR / "prefixes_v2.json"          # merge back into the canonical name


def renumber(prefixes):
    counters = {}
    for p in prefixes:
        n = counters.get(p["label"], 0)
        p["prefix_id"] = "%s_%03d" % (p["label"], n)
        counters[p["label"]] = n + 1
    return prefixes


def main():
    main_batch = json.load(open(MAIN))
    topup_batch = json.load(open(TOPUP))
    print("main batch:  %d (from the original 480-conversation run)" % len(main_batch))
    print("top-up:      %d (seed 1, +100/class)" % len(topup_batch))

    merged = renumber(main_batch + topup_batch)
    raw_texts = [p["raw_text"] for p in merged]
    dupes = len(raw_texts) - len(set(raw_texts))
    print("merged:      %d total, %d duplicate raw_text (should be 0)" % (len(merged), dupes))
    assert dupes == 0, "duplicate conversations survived the merge -- investigate before using this data"

    usable = [p for p in merged if is_usable(p["turns"], p["raw_text"])]
    print("usable:      %d/%d (%.1f%%)\n" % (len(usable), len(merged), 100 * len(usable) / len(merged)))

    print("Markers by class (post-merge)")
    for label, stats in marker_table(usable).items():
        print("  %-10s %s" % (label, json.dumps(stats)))

    ga = gate_a(usable)
    print("\nGate A: %s" % json.dumps(ga))
    assert ga["passed"], "Gate A regressed after the merge -- should not happen, investigate"

    gb_pre = gate_b(usable)
    print("Gate B (pre-match): %s" % json.dumps(gb_pre))

    matched = length_match(usable)
    gb_post = gate_b(matched)
    print("Gate B (post-match): %s" % json.dumps(gb_post))

    n_per_class = min(gb_post["per_class_n"].values()) if gb_post.get("per_class_n") else 0
    print("\nMatched per class: %d  (fallback floor: 80, top-up target: 100)" % n_per_class)
    if n_per_class >= 100:
        print("PASSES the top-up threshold. Proceed with the matched %d/class set." % n_per_class)
    elif n_per_class >= 80:
        print("Clears the fallback floor but not the top-up target. Per GATES.md, this is "
              "acceptable: use the full set with prefix length as a covariate, and say so.")
    else:
        print("STILL below the fallback floor. Do not proceed without revisiting the design "
              "(see GATES.md fallback clause) -- this should not happen after a top-up.")

    json.dump(merged, open(OUT, "w"), indent=2)
    print("\nWrote merged set (%d conversations) -> %s" % (len(merged), OUT.relative_to(REPO_ROOT)))
    matched_path = DATA_DIR / "prefixes_v2_matched.json"
    json.dump(matched, open(matched_path, "w"), indent=2)
    print("Wrote length-matched usable set (%d conversations) -> %s"
          % (len(matched), matched_path.relative_to(REPO_ROOT)))


if __name__ == "__main__":
    main()
