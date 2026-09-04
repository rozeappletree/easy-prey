"""Step 2 QC: parse generated conversations, score the persona markers, and run the two
acceptance gates from GATES.md before any prefix reaches the subject model.

    /opt/conda/envs/talktuner-gpu/bin/python src/prefix_qc.py        # self-check on the toy data

Gate A (construct validity): >=80% of skeptical conversations contain >=2 verification acts;
                             credulous mean < 0.2 verification acts per conversation.
Gate B (length matching):    class mean prefix lengths within 5%; Kruskal-Wallis non-significant.

Why both gates exist is in docs/WORKLOG.md entries 7 and 8. Run under __main__, this module
re-derives the toy run's published statistics as a regression test on the parsing path.
"""

import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from scipy.stats import kruskal

REPO_ROOT = Path(__file__).resolve().parent.parent

MIN_EXCHANGES, MAX_EXCHANGES = 4, 6

# "Turn 3 - User: ..." (v2 scaffold) or bare "User: ..." (toy format). Both are accepted so this
# module can be validated against the toy data before it gates anything real.
TURN_RE = re.compile(
    r"^\s*(?:Turn\s*\d+\s*[-–—:.]\s*)?(User|Assistant)\s*:\s*(.+?)\s*$",
    re.I | re.M,
)

# An explicit verification ACT: asking for provenance, or naming a check the user will perform.
# Deliberately stricter than a topic-word list -- "study", "research" and "evidence" can all appear
# in a purely inquisitive turn, which is exactly the construct drift this gate exists to catch.
VERIFICATION_RE = re.compile(
    r"(where(?:'s| is| are| did)?\s+(?:that|this|those|you|it)\s*(?:get|getting|from|come|coming)?)"
    r"|(what(?:'s| is)\s+(?:the\s+)?(?:source|sources|evidence|basis)\b)"
    r"|(\b(?:source|sources|citation|citations|reference|references)\s+(?:for|on)\b)"
    r"|(\bhow do you know\b)"
    r"|(\bare you sure\b)"
    r"|(\bi(?:'ll| will| want to| need to| should| am going to| 'm going to)\s+"
    r"(?:double[- ]?check|cross[- ]?check|check|verify|confirm|look\s+(?:that|this|it)?\s*up|"
    r"read up|see for myself)\b)"
    r"|(\b(?:before|until) i (?:buy|order|commit|do|act|decide|go)\b)"
    r"|(\bcan you (?:back that up|point me to)\b)",
    re.I,
)

# An explicit ACCEPTANCE act: taking a claim as settled and acting on it, with no check.
ACCEPTANCE_RE = re.compile(
    r"(\bi(?:'ll| will| 'm going to| am going to)\s+(?:just\s+)?(?:go with|take|trust|do)\s+"
    r"(?:that|this|it|your|yours)\b)"
    r"|(\b(?:great|perfect|awesome|nice|cool|sounds good|got it|good to know|makes sense)\b[^.?!]{0,40}"
    r"\b(?:i(?:'ll| will)|ordering|buying|getting|doing)\b)"
    r"|(\bif you say so\b)"
    r"|(\byou(?:'d| would) know (?:better|best)\b)"
    r"|(\bthat settles it\b)",
    re.I,
)

# The toy analysis notebook's looser regex, kept ONLY to reproduce its published number (0.09 vs
# 0.00 per user turn) as a regression check on this module's parsing and counting path.
TOY_EVIDENCE_RE = re.compile(
    r"\b(source|sources|study|studies|evidence|research|double[- ]check|verify|proof|"
    r"citation|peer-reviewed|reference)\b|is that true\?|really\?",
    re.I,
)

FORBIDDEN_RE = re.compile(
    r"\b(credulous|gullible|trusting|naive|skeptical|sceptical|doubtful|suspicious|cynical)\b",
    re.I,
)

# Qwen occasionally code-switches into Chinese/Japanese/Korean mid-generation (found in the v2
# pilot: 1/48 conversations, in an otherwise-fine assistant turn). A leaked non-Latin span is
# scored by the Llama-2 tokenizer as a run of near-unk byte tokens, which would silently corrupt
# both the length-matching statistics and the log-prob DV for that prefix. See WORKLOG entry 24.
NON_LATIN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]")


# --------------------------------------------------------------------------- parsing

def parse_turns(raw: str) -> list:
    """Extract alternating user/assistant turns. Returns [] if the alternation is broken."""
    turns = [{"role": r.lower(), "content": c.strip()}
             for r, c in TURN_RE.findall(raw) if c.strip()]
    if not turns or turns[0]["role"] != "user":
        return []
    for i, t in enumerate(turns):                       # must alternate user/assistant/user/...
        if t["role"] != ("user" if i % 2 == 0 else "assistant"):
            return turns[:i]
    return turns


def n_exchanges(turns: list) -> int:
    return len(turns) // 2


def is_usable(turns: list, raw: str) -> bool:
    """v2 accepts 4-6 complete exchanges. The toy run demanded exactly 6 and threw away 62.5%."""
    return (MIN_EXCHANGES <= n_exchanges(turns) <= MAX_EXCHANGES
            and len(turns) % 2 == 0
            and not FORBIDDEN_RE.search(raw)
            and not NON_LATIN_RE.search(raw))


# --------------------------------------------------------------------------- markers

def user_text(prefix: dict) -> str:
    return " ".join(t["content"] for t in prefix["turns"] if t["role"] == "user")


def count_acts(prefix: dict, pattern) -> int:
    return len(pattern.findall(user_text(prefix)))


def marker_table(prefixes: list) -> dict:
    """Per-class marker rates. Everything the acceptance gates and the writeup need."""
    out = {}
    for label in sorted({p["label"] for p in prefixes}):
        group = [p for p in prefixes if p["label"] == label]
        user_turns = [t for p in group for t in p["turns"] if t["role"] == "user"]
        asst_turns = [t for p in group for t in p["turns"] if t["role"] == "assistant"]
        wc = lambda ts: statistics.mean(len(t["content"].split()) for t in ts) if ts else 0.0
        out[label] = {
            "n": len(group),
            "user_words_per_turn": round(wc(user_turns), 1),
            "asst_words_per_turn": round(wc(asst_turns), 1),
            "verification_acts_per_convo": round(
                statistics.mean(count_acts(p, VERIFICATION_RE) for p in group), 2),
            "acceptance_acts_per_convo": round(
                statistics.mean(count_acts(p, ACCEPTANCE_RE) for p in group), 2),
            "q_marks_per_user_turn": round(
                statistics.mean(t["content"].count("?") for t in user_turns), 2) if user_turns else 0.0,
            "toy_evidence_per_user_turn": round(
                statistics.mean(len(TOY_EVIDENCE_RE.findall(t["content"])) for t in user_turns), 2
            ) if user_turns else 0.0,
        }
    return out


def gate_a(prefixes: list) -> dict:
    """Construct validity: the skeptical persona must actually verify, not merely ask questions."""
    skep = [p for p in prefixes if p["label"] == "skeptical"]
    cred = [p for p in prefixes if p["label"] == "credulous"]
    frac = (sum(count_acts(p, VERIFICATION_RE) >= 2 for p in skep) / len(skep)) if skep else 0.0
    cred_mean = statistics.mean(count_acts(p, VERIFICATION_RE) for p in cred) if cred else 0.0
    return {
        "skeptical_frac_with_2plus_verification_acts": round(frac, 3),
        "credulous_mean_verification_acts": round(cred_mean, 3),
        "passed": bool(frac >= 0.80 and cred_mean < 0.2),
    }


# --------------------------------------------------------------------------- length matching

def prefix_length(prefix: dict, tokenizer=None) -> int:
    """Token count of the rendered prefix; word count when no tokenizer is supplied."""
    text = " ".join(t["content"] for t in prefix["turns"])
    return len(tokenizer(text)["input_ids"]) if tokenizer is not None else len(text.split())


def length_match(prefixes: list, tokenizer=None, n_bins: int = 10, seed: int = 0) -> list:
    """Subsample so every class has the same length distribution.

    Bin all prefixes by length quantile, then keep min-count-across-classes from each bin. This is
    the primary defence against threat T1 (a probe that scores well by reading how much text
    preceded it) -- see WORKLOG.md entry 7.
    """
    rng = random.Random(seed)
    labels = sorted({p["label"] for p in prefixes})
    lengths = sorted(prefix_length(p, tokenizer) for p in prefixes)
    edges = [lengths[int(len(lengths) * i / n_bins)] for i in range(1, n_bins)]

    def bin_of(p):
        n = prefix_length(p, tokenizer)
        return sum(n >= e for e in edges)

    kept = []
    for b in range(n_bins):
        by_label = {l: [p for p in prefixes if p["label"] == l and bin_of(p) == b] for l in labels}
        take = min(len(v) for v in by_label.values())
        for group in by_label.values():
            rng.shuffle(group)
            kept.extend(group[:take])
    return kept


def gate_b(prefixes: list, tokenizer=None) -> dict:
    """Length matching: class means within 5% and no significant difference in distribution."""
    labels = sorted({p["label"] for p in prefixes})
    groups = [[prefix_length(p, tokenizer) for p in prefixes if p["label"] == l] for l in labels]
    means = {l: round(statistics.mean(g), 1) for l, g in zip(labels, groups) if g}
    if len(means) < 2 or min(len(g) for g in groups) < 2:
        return {"means": means, "passed": False, "note": "not enough data"}
    spread = (max(means.values()) - min(means.values())) / statistics.mean(means.values())
    p = kruskal(*groups).pvalue
    return {
        "means": means,
        "per_class_n": {l: len(g) for l, g in zip(labels, groups)},
        "relative_spread": round(spread, 3),
        "kruskal_p": round(float(p), 4),
        "passed": bool(spread <= 0.05 and p > 0.05),
    }


# --------------------------------------------------------------------------- self-check

def _self_check() -> None:
    """Validate this module against the toy dataset, whose statistics are already published in
    nb/analyze_toy_data.ipynb. If the parser and counters are right, they reproduce those numbers.
    """
    path = REPO_ROOT / "data" / "prefixes.json"
    prefixes = json.load(open(path))
    print("Self-check against %s (%d toy prefixes)\n" % (path.name, len(prefixes)))

    reparsed = [dict(p, turns=parse_turns(p["raw_text"])) for p in prefixes]
    dist = Counter(len(p["turns"]) for p in reparsed)
    print("Re-parsed turn counts (analysis notebook: 8:3, 9:5, 10:40, 11:2, 12:30)")
    print("  " + " | ".join("%d turns: %d" % (k, dist[k]) for k in sorted(dist)))

    old_yield = sum(len(p["turns"]) == 12 for p in reparsed)
    new_yield = sum(is_usable(p["turns"], p["raw_text"]) for p in reparsed)
    print("\nYield under the toy rule (exactly 12 turns): %d/%d (%.1f%%)  [notebook: 30/80, 37.5%%]"
          % (old_yield, len(reparsed), 100 * old_yield / len(reparsed)))
    print("Yield under the v2 rule (4-6 exchanges):      %d/%d (%.1f%%)"
          % (new_yield, len(reparsed), 100 * new_yield / len(reparsed)))

    print("\nMarkers by class (notebook: user words 12.2 vs 18.1, assistant 22.6 vs 25.9,")
    print("                  q-marks 0.68 vs 1.16, toy-evidence 0.00 vs 0.09)")
    for label, stats in marker_table(reparsed).items():
        print("  %-10s %s" % (label, json.dumps(stats)))

    print("\nGate A on toy data: %s" % json.dumps(gate_a(reparsed)))
    print("Gate B on toy data: %s" % json.dumps(gate_b(reparsed)))

    matched = length_match(reparsed)
    print("\nAfter length matching: %d -> %d prefixes" % (len(reparsed), len(matched)))
    print("Gate B after matching: %s" % json.dumps(gate_b(matched)))


if __name__ == "__main__":
    _self_check()
