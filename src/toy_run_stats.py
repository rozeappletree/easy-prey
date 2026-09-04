"""Recover the measured facts from the toy run that reshaped the design.

    /opt/conda/envs/talktuner-gpu/bin/python src/toy_run_stats.py

Reads the executed cell outputs of nb/generate_toy_data.ipynb plus data/prefixes.json, and
re-derives: generation throughput, the malformed-output diagnosis, and the item-bank category
skew. These numbers are cited in docs/WORKLOG.md entries 2, 6, 9 and 10 -- this script exists so
they can be re-checked rather than taken on trust.
"""

import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NB = REPO_ROOT / "nb" / "generate_toy_data.ipynb"


def notebook_outputs(path):
    nb = json.load(open(path))
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            text = out.get("text") or out.get("data", {}).get("text/plain")
            if text:
                yield "".join(text)


def throughput():
    print("=" * 78)
    print("1. Generation throughput (the finding that reshaped the plan)")
    print("=" * 78)
    blob = "\n".join(notebook_outputs(NB))

    minutes = re.search(r"Done in ([\d.]+) min", blob)
    yield_m = re.search(r"(\d+)/(\d+) well-formed", blob)
    vram = re.search(r"GPU memory allocated: ([\d.]+) GB", blob)
    model = re.search(r"Loaded (\S+) in", blob)

    if not (minutes and yield_m):
        print("  could not find timing output in the notebook -- was it executed?")
        return
    mins, good, total = float(minutes.group(1)), int(yield_m.group(1)), int(yield_m.group(2))
    prefixes = json.load(open(REPO_ROOT / "data" / "prefixes.json"))
    gen_words = sum(len(p["raw_text"].split()) for p in prefixes)
    approx_tokens = gen_words * 1.3                       # ~1.3 tokens/word for English

    print("  generator        %s, 4-bit NF4, batch 4" % (model.group(1) if model else "?"))
    print("  VRAM allocated   %s GB" % (vram.group(1) if vram else "?"))
    print("  wall clock       %.1f min for %d conversations" % (mins, total))
    print("  rate             %.2f conversations/min  (%.1f min each)" % (total / mins, mins / total))
    print("  throughput       ~%.1f aggregate tokens/s" % (approx_tokens / (mins * 60)))
    print("  usable           %d/%d (%.1f%%) under the toy rule of exactly 12 turns" %
          (good, total, 100 * good / total))
    print("  effective cost   %.1f min per USABLE conversation" % (mins / good))
    print()
    print("  Extrapolation: 480 conversations at this rate = %.1f hours of a 20-hour budget."
          % (480 * mins / total / 60))
    print("  Conclusion: switch the generator to bf16 and raise the batch size. NF4 dequantization")
    print("  dominates decode, and batch 4 left ~18 GB of VRAM idle. See WORKLOG.md entry 6.")


def malformed_diagnosis():
    print("\n" + "=" * 78)
    print("2. Why the yield was low: early stopping, not truncation")
    print("=" * 78)
    prefixes = json.load(open(REPO_ROOT / "data" / "prefixes.json"))
    good = [p for p in prefixes if p["well_formed"]]
    bad = [p for p in prefixes if not p["well_formed"]]

    ends_clean = lambda p: p["raw_text"].strip().endswith((".", "!", "?", '"'))
    wc = lambda ps: sum(len(p["raw_text"].split()) for p in ps) / len(ps) if ps else 0

    dist = Counter(len(p["turns"]) for p in prefixes)
    print("  turn-count distribution (target 12 = 6 exchanges):")
    for k in sorted(dist):
        print("    %2d turns  %3d convos%s" % (k, dist[k], "   <- mode" if dist[k] == max(dist.values()) else ""))
    print()
    print("  malformed ending on clean punctuation  %d/%d (%.0f%%)"
          % (sum(map(ends_clean, bad)), len(bad), 100 * sum(map(ends_clean, bad)) / len(bad)))
    print("  mean words   malformed %.0f  |  well-formed %.0f  (%.0f%% shorter)"
          % (wc(bad), wc(good), 100 * (1 - wc(bad) / wc(good))))
    print()
    print("  A truncated generation stops mid-sentence. These stop cleanly, only slightly short,")
    print("  with a mode of 5 exchanges instead of 6 -- the model decided it was finished.")
    print("  So max_new_tokens was never the lever. See WORKLOG.md entry 9.")


def item_skew():
    print("\n" + "=" * 78)
    print("3. Item-bank skew under random sampling (why v2 uses quotas)")
    print("=" * 78)
    items = json.load(open(REPO_ROOT / "data" / "items.json"))
    counts = Counter(i["category"] for i in items)
    print("  toy item bank, %d items sampled at random:" % len(items))
    for c, n in counts.most_common():
        flag = "   <- unusable as a category" if n <= 2 else ""
        print("    %-24s %3d%s" % (c, n, flag))
    new = REPO_ROOT / "data" / "items_v2.json"
    if new.exists():
        v2 = Counter(i["category"] for i in json.load(open(new)))
        print("\n  v2 item bank, %d items under fixed quotas:" % sum(v2.values()))
        for c, n in v2.most_common():
            print("    %-24s %3d" % (c, n))
    print("\n  See WORKLOG.md entries 10 and 15.")


if __name__ == "__main__":
    throughput()
    malformed_diagnosis()
    item_skew()
    print()
