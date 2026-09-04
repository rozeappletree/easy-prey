"""Step 7 (C2) prerequisite: build a clean, ITEM-INDEPENDENT set of "combined" stimuli for probing.

    /opt/conda/envs/talktuner-gpu/bin/python src/build_combined_stimuli.py

Why this can't reuse data/c1_g5_llama13b_combined.json: that file's (prefix, template) pairing was
drawn by a per-item random-number stream inside c1_total_effect.py's run_combined(), so the same
(prefix_id, template_idx) pair is not reliably reused across items and there is no clean, closed set
of "the N combined stimuli" independent of which item happened to be scored alongside them. A probe
needs activations extracted ONCE from a fixed stimulus set, decoupled from items (see the
causal-invariance argument in WORKLOG entry 22, re-verified for this combined system+conversation
format before this script was written).

Design (WORKLOG entry 39):
  - All 12 hand-written stated-persona templates per class are used (too few to hold any out).
  - 20 conversation prefixes per class, split 14 train / 6 held-out BY PREFIX IDENTITY -- so
    generalisation is tested on unseen conversations, not just unseen (template, prefix) pairs.
  - FULL CROSS within each split: every template paired with every prefix in that split.
    => 12 x 14 = 168 train, 12 x 6 = 72 held-out, per class. 720 stimuli total.
  - A stimulus's read position is the last token of [system turn] + [conversation turns], BEFORE
    any item's question is appended -- verified causally invariant to what follows.

Writes data/c2_stimuli.json: one row per stimulus with everything needed to build the prompt.
"""

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"

SEED = 0
N_PREFIXES_PER_CLASS = 20
N_HELDOUT_PREFIXES = 6           # -> 14 train / 6 held-out
CLASSES = ("credulous", "neutral", "skeptical")


def main():
    rng = random.Random(SEED)
    matched = json.load(open(DATA / "prefixes_v2_matched.json"))
    templates = json.load(open(DATA / "persona_templates_for_T3_human_augmented_with_neutral.json"))

    by_label = {}
    for p in matched:
        by_label.setdefault(p["label"], []).append(p)

    stimuli = []
    for label in CLASSES:
        pool = by_label[label]
        assert len(pool) >= N_PREFIXES_PER_CLASS, "%s pool too small: %d" % (label, len(pool))
        chosen = rng.sample(pool, N_PREFIXES_PER_CLASS)
        rng.shuffle(chosen)
        held_out_prefixes = chosen[:N_HELDOUT_PREFIXES]
        train_prefixes = chosen[N_HELDOUT_PREFIXES:]

        for split, prefixes in [("train", train_prefixes), ("held_out", held_out_prefixes)]:
            for prefix in prefixes:
                for t_idx, sentence in enumerate(templates[label]):
                    stimuli.append({
                        "stimulus_id": "%s_%s_t%02d" % (label, prefix["prefix_id"], t_idx),
                        "label": label,
                        "split": split,
                        "prefix_id": prefix["prefix_id"],
                        "template_idx": t_idx,
                        "template_text": sentence,
                        "turns": prefix["turns"],
                    })

    out = DATA / "c2_stimuli.json"
    json.dump(stimuli, open(out, "w"), indent=2)

    from collections import Counter
    counts = Counter((s["label"], s["split"]) for s in stimuli)
    print("Wrote %d combined stimuli -> %s" % (len(stimuli), out.relative_to(REPO_ROOT)))
    for label in CLASSES:
        print("  %-10s train=%3d  held_out=%3d" % (label, counts[(label, "train")], counts[(label, "held_out")]))


if __name__ == "__main__":
    main()
