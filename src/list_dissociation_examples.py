"""List, item by item, where Llama-2-13b's stated-persona effect and behavioural-conversation
effect agree vs disagree in direction -- with the actual conversations and sentences shown, so the
result in docs/WORKLOG.md entry 36 / nb/03 Sec3 can be read as concrete examples, not just d_z.

    /opt/conda/envs/talktuner-gpu/bin/python src/list_dissociation_examples.py

For every one of the 100 high-stakes items:
  effect_behav   = mean_Delta(credulous, behavioural) - mean_Delta(skeptical, behavioural)
  effect_ceiling = mean_Delta(credulous, ceiling)      - mean_Delta(skeptical, ceiling)
Positive = credulous condition pulled the model MORE toward the lure than skeptical did.

Items split into two files by whether effect_behav and effect_ceiling AGREE in sign (both channels
push the same direction for this item) or DIVERGE (opposite signs -- the two ways of conveying
credulity disagree about which persona makes this specific item worse).

Writes:
  outputs/examples_dissociation.txt  -- items where the two channels disagree
  outputs/examples_agreement.txt     -- items where the two channels agree

Each item shows: the question, correct answer, lures, both effect sizes, one example credulous and
one example skeptical BEHAVIOURAL conversation (of the 6 sampled per item -- the rest are the same
persona, different topic, and are not shown), and the actual stated-persona sentences used in the
CEILING condition with their individual Delta values.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
OUT = REPO_ROOT / "outputs"


def load():
    items = {i["item_id"]: i for i in json.load(open(DATA / "items_v2.json"))}
    behav = json.load(open(DATA / "c1_behavioural.json"))
    ceil_ = json.load(open(DATA / "c1_ceiling.json"))
    matched = {p["prefix_id"]: p for p in json.load(open(DATA / "prefixes_v2_matched.json"))}
    templates = json.load(open(DATA / "persona_templates_for_T3_human_augmented.json"))
    return items, behav, ceil_, matched, templates


def per_item_effect(rows, item_id, credulous_key="credulous", skeptical_key="skeptical"):
    """mean Delta(credulous) - mean Delta(skeptical) for one item, plus the raw rows for display."""
    by_cond = defaultdict(list)
    for r in rows:
        if r["item_id"] == item_id:
            by_cond[r["condition"]].append(r)
    cred = [r["delta"] for r in by_cond[credulous_key]]
    skep = [r["delta"] for r in by_cond[skeptical_key]]
    if not cred or not skep:
        return None, by_cond
    return statistics.mean(cred) - statistics.mean(skep), by_cond


def format_conversation(prefix):
    lines = []
    for t in prefix["turns"]:
        role = t["role"].upper() + ":"
        lines.append("    %-11s %s" % (role, t["content"]))
    return "\n".join(lines)


def format_item_block(item, effect_behav, effect_ceiling, behav_rows, ceil_rows, matched, templates):
    lines = []
    lines.append("=" * 88)
    lines.append("ITEM %s  [%s]" % (item["item_id"], item["category"]))
    lines.append("Q:       %s" % item["question"])
    lines.append("CORRECT: %s" % item["correct"])
    for i, lure in enumerate(item["lures"], 1):
        lines.append("LURE %d:  %s" % (i, lure))
    lines.append("")
    tag_b = "credulous MORE lure-leaning" if effect_behav > 0 else "credulous LESS lure-leaning"
    tag_c = "credulous MORE lure-leaning" if effect_ceiling > 0 else "credulous LESS lure-leaning"
    lines.append("BEHAVIOURAL effect (credulous - skeptical): %+.3f   (%s)" % (effect_behav, tag_b))
    lines.append("CEILING     effect (credulous - skeptical): %+.3f   (%s)" % (effect_ceiling, tag_c))
    verdict = "AGREE -- both channels push the same direction" \
        if (effect_behav > 0) == (effect_ceiling > 0) \
        else "DIVERGE -- the two channels push in OPPOSITE directions"
    lines.append("=> %s" % verdict)
    lines.append("")

    cred_rows = behav_rows.get("credulous", [])
    skep_rows = behav_rows.get("skeptical", [])
    if cred_rows:
        r = cred_rows[0]
        p = matched[r["prefix_id"]]
        lines.append("--- one CREDULOUS behavioural conversation (%s, 1 of %d sampled), "
                      "Delta = %+.3f ---" % (r["prefix_id"], len(cred_rows), r["delta"]))
        lines.append(format_conversation(p))
        lines.append("")
    if skep_rows:
        r = skep_rows[0]
        p = matched[r["prefix_id"]]
        lines.append("--- one SKEPTICAL behavioural conversation (%s, 1 of %d sampled), "
                      "Delta = %+.3f ---" % (r["prefix_id"], len(skep_rows), r["delta"]))
        lines.append(format_conversation(p))
        lines.append("")

    for label in ("credulous", "skeptical"):
        rows = ceil_rows.get(label, [])
        if not rows:
            continue
        best = max(rows, key=lambda r: r["delta"] if label == "credulous" else -r["delta"])
        sentence = templates[label][best["template_idx"]]
        lines.append("--- most extreme STATED %s persona (template #%d), Delta = %+.3f ---"
                      % (label.upper(), best["template_idx"], best["delta"]))
        lines.append('    "%s"' % sentence)
        lines.append("")

    return "\n".join(lines)


def main():
    items, behav, ceil_, matched, templates = load()

    diverge, agree = [], []
    for item_id, item in items.items():
        eb, behav_rows = per_item_effect(behav, item_id)
        ec, ceil_rows = per_item_effect(ceil_, item_id)
        if eb is None or ec is None:
            continue
        block = format_item_block(item, eb, ec, behav_rows, ceil_rows, matched, templates)
        record = (abs(eb - ec), eb, ec, block)
        if (eb > 0) == (ec > 0):
            agree.append(record)
        else:
            diverge.append(record)

    diverge.sort(key=lambda r: -r[0])                                    # biggest split first
    agree.sort(key=lambda r: -min(abs(r[1]), abs(r[2])))                 # strongest concordant signal first

    header = (
        "Llama-2-13b-chat: per-item comparison of the BEHAVIOURAL (natural conversation) effect\n"
        "against the CEILING (explicitly stated persona) effect. See docs/WORKLOG.md entry 36 and\n"
        "nb/03_results_what_we_found.ipynb Sec3 for the aggregate statistics this breaks down.\n\n"
        "effect = mean_Delta(credulous) - mean_Delta(skeptical), for that one item.\n"
        "Positive = the credulous condition pulled the model MORE toward the lure than skeptical did.\n\n"
        "Generated by src/list_dissociation_examples.py from data/c1_behavioural.json, "
        "c1_ceiling.json,\ndata/items_v2.json, data/prefixes_v2_matched.json, "
        "data/persona_templates_for_T3_human_augmented.json.\n"
    )

    with open(OUT / "examples_dissociation.txt", "w") as f:
        f.write(header)
        f.write("\n%d / %d items DIVERGE (behavioural and ceiling disagree on direction),\n"
                "sorted by how large the split is (biggest disagreement first).\n\n"
                % (len(diverge), len(diverge) + len(agree)))
        for _, eb, ec, block in diverge:
            f.write(block + "\n")

    with open(OUT / "examples_agreement.txt", "w") as f:
        f.write(header)
        f.write("\n%d / %d items AGREE (behavioural and ceiling push the same direction),\n"
                "sorted by strength of the weaker of the two signals (strongest agreement first).\n\n"
                % (len(agree), len(diverge) + len(agree)))
        for _, eb, ec, block in agree:
            f.write(block + "\n")

    print("DIVERGE: %d items -> outputs/examples_dissociation.txt" % len(diverge))
    print("AGREE:   %d items -> outputs/examples_agreement.txt" % len(agree))
    print("(%d items skipped: missing rows in one condition)"
          % (len(items) - len(diverge) - len(agree)))


if __name__ == "__main__":
    main()
