"""Enforce Channel A's two standards on any persona-template file, the same way prefix_qc.py
enforces them on Channel B's conversations.

    /opt/conda/envs/talktuner-gpu/bin/python src/check_persona_templates.py \
        data/persona_templates_for_T3_human_augmented.json

Why this exists as a script and not a one-off check: the length-balance rule and the
construct-purity rule were applied to Channel A three times in conversation (see WORKLOG.md
entries 25, 28-29) but never saved as reusable code, unlike Channel B's identical rules
(prefix_qc.gate_b, prefix_qc.FORBIDDEN_RE). If this file changes again, this script re-derives the
same verdict instead of someone re-deriving the arithmetic by hand.

Two checks, matching DESIGN.md 5.2's Channel B gates:

  LENGTH BALANCE (same failure mode as prefix_qc.gate_b / the toy conversation data: credulous
  12.2 vs skeptical 18.1 words/turn). Reported as relative spread + Mann-Whitney p, no hard
  threshold -- 24 hand-written sentences can't be decile-matched like 750 conversations can, so
  this is a warning, not a gate.

  CONSTRUCT PURITY (same failure mode as prefix_qc's verification/acceptance regexes, generalized):
  scans for phrases that smuggle in a trait other than credulity -- trust/distrust of others'
  intentions, cognitive-capacity language, stubbornness-as-skepticism. This list is NOT
  exhaustive -- it is exactly the phrases found by manual review so far. Extend it when a new one
  is found; do not treat a clean scan as proof there are no others.
"""

import argparse
import json
import statistics
import sys

from scipy.stats import mannwhitneyu

# Each entry: (pattern, why it's a problem). Checked case-insensitively as a substring.
DRIFT_PHRASES = [
    ("trying to help", "trust-in-intentions, not credulity (the original help/cheat pair, entry 25)"),
    ("trying to cheat", "distrust-of-intentions, not credulity (the original help/cheat pair, entry 25)"),
    ("their own agenda", "distrust-of-intentions, not credulity (entry 28)"),
    ("mentally challenged", "cognitive-capacity confound, not credulity (entry 25)"),
    ("stubborn", "resistance-to-updating is arguably anti-skeptical, not pro-skeptical (entry 28)"),
]

TRAIT_WORDS = ["credulous", "gullible", "trusting", "naive",
               "skeptical", "sceptical", "doubtful", "suspicious", "cynical"]


def check_length(data):
    print("=" * 70)
    print("LENGTH BALANCE  (DESIGN.md Sec 5.2 Gate B; same rule as prefix_qc.gate_b)")
    print("=" * 70)
    counts = {label: [len(s.split()) for s in sentences] for label, sentences in data.items()
              if label != "_meta"}
    means = {label: statistics.mean(c) for label, c in counts.items()}
    spread = (max(means.values()) - min(means.values())) / statistics.mean(sum(counts.values(), []))
    labels = list(counts)
    for label in labels:
        print("  %-12s n=%2d  mean %.1fw  range %d-%d"
              % (label, len(counts[label]), means[label], min(counts[label]), max(counts[label])))
    if len(labels) == 2:
        p = mannwhitneyu(counts[labels[0]], counts[labels[1]]).pvalue
        print("  relative spread: %.1f%%   Mann-Whitney p = %.3f" % (100 * spread, p))
        flag = spread > 0.10
    else:
        print("  relative spread: %.1f%%  (2+ classes, pairwise test skipped)" % (100 * spread))
        flag = spread > 0.10
    print("  %s (no hard gate -- 24 sentences can't be decile-matched; use judgement)"
          % ("FLAGGED: spread > 10%%, consider trimming outliers" if flag else "OK"))
    return not flag


def check_construct_purity(data):
    print("\n" + "=" * 70)
    print("CONSTRUCT PURITY  (DESIGN.md Sec 5.2 Gate A, generalized; list is NOT exhaustive)")
    print("=" * 70)
    hits = []
    for label, sentences in data.items():
        if label == "_meta":
            continue
        for i, s in enumerate(sentences):
            low = s.lower()
            for phrase, why in DRIFT_PHRASES:
                if phrase in low:
                    hits.append((label, i, phrase, why))
            for w in TRAIT_WORDS:
                if w in low:
                    hits.append((label, i, w, "the trait word itself must never appear (leaks the label)"))
    if hits:
        for label, i, phrase, why in hits:
            print("  FLAGGED  %-10s [%d]  %r -- %s" % (label, i, phrase, why))
    else:
        print("  OK -- no known drift phrases or trait words found")
        print("  (this scan only catches phrases seen before; read new sentences too)")
    return not hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    data = json.load(open(args.path))

    print("Checking %s\n" % args.path)
    ok_length = check_length(data)
    ok_purity = check_construct_purity(data)

    print("\n" + "=" * 70)
    print("RESULT: %s" % ("PASS" if ok_length and ok_purity else "NEEDS REVIEW"))
    print("=" * 70)
    return 0 if ok_length and ok_purity else 1


if __name__ == "__main__":
    sys.exit(main())
