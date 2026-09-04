"""Step 1 of the Family-7 mediation plan: build the stratified item bank.

See docs/PLAN_TECHNICAL.md Step 1. Pure CPU, no network needed (TruthfulQA is cached).

    /opt/conda/envs/talktuner-gpu/bin/python src/build_items.py

Writes data/items_v2.json (100 high-stakes) and data/items_lowstakes.json (30 low-stakes).

Why a quota instead of random sampling: the toy run sampled 30 items at random from a 239-item
pool and landed on Finance n=1, Nutrition n=2 -- too few to say anything per-category.
"""

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

SEED = 0

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Category quotas. Pool sizes (after filtering) are in the comments; see PLAN_TECHNICAL.md Step 1.
HIGH_STAKES_QUOTA = {
    "Misconceptions": 25,          # pool 89
    "Law": 25,                     # pool 58
    "Health": 25,                  # pool 54
    "Economics|Finance": 15,       # pool 31 + 9
    "Nutrition": 10,               # pool 16
}
LOW_STAKES_QUOTA = {
    "Fiction": 12,                 # pool 26
    "Myths and Fairytales": 10,    # pool 19
    "Proverbs": 8,                 # pool 15
}

N_LURES = 2                        # first two entries of Incorrect Answers
REFUSAL = "i have no comment"
MAX_LURE_LENGTH_GAP = 3.0          # words; toy run measured 10.1 vs 10.7


def split_answers(cell) -> list:
    """Semicolon-separated answer fields -> list of strings. Same logic as generate_toy_data.ipynb."""
    if pd.isna(cell):
        return []
    return [a.strip() for a in str(cell).split(";") if a.strip()]


def word_count(s: str) -> int:
    return len(str(s).split())


def load_candidates() -> pd.DataFrame:
    """TruthfulQA, minus refusals, minus items without enough lures."""
    path = hf_hub_download("domenicrosati/TruthfulQA", "TruthfulQA.csv", repo_type="dataset")
    tqa = pd.read_csv(path)

    tqa["correct_list"] = tqa["Correct Answers"].apply(split_answers)
    tqa["incorrect_list"] = tqa["Incorrect Answers"].apply(split_answers)

    # Refusals let a model score "truthful" by dodging, which confounds the whole experiment.
    answerable = tqa["Best Answer"].str.strip().str.lower() != REFUSAL
    has_lures = tqa["incorrect_list"].apply(len) >= N_LURES

    candidates = tqa[answerable & has_lures].reset_index(drop=True)
    print(f"TruthfulQA: {len(tqa)} rows -> {len(candidates)} usable "
          f"(dropped {(~answerable).sum()} refusals, "
          f"{(answerable & ~has_lures).sum()} with <{N_LURES} lures)")
    return candidates


def select(candidates: pd.DataFrame, quota: dict, stakes: str, rng: random.Random) -> list:
    """Greedily fill each category's quota, skipping items that would duplicate a question or lure.

    Quota keys may be 'A|B' to merge categories that are individually too small (Economics+Finance).
    """
    items, used_questions, used_lures = [], set(), set()

    for quota_key, n_wanted in quota.items():
        categories = quota_key.split("|")
        pool = candidates[candidates["Category"].isin(categories)]
        order = list(pool.index)
        rng.shuffle(order)

        n_taken = 0
        for idx in order:
            if n_taken >= n_wanted:
                break
            row = pool.loc[idx]
            lures = row["incorrect_list"][:N_LURES]

            if row["Question"] in used_questions or any(l in used_lures for l in lures):
                continue  # a shared lure across items would double-count the same falsehood

            used_questions.add(row["Question"])
            used_lures.update(lures)
            items.append({
                "item_id": f"{'hs' if stakes == 'high' else 'ls'}_{len(items):03d}",
                "stakes": stakes,
                "category": row["Category"],
                "type": row["Type"],
                "question": row["Question"],
                "correct": row["Best Answer"],
                "correct_answers": row["correct_list"],
                "incorrect_answers": row["incorrect_list"],
                "lures": lures,
            })
            n_taken += 1

        if n_taken < n_wanted:
            raise SystemExit(
                f"Quota unmet for {quota_key}: wanted {n_wanted}, got {n_taken}. "
                f"Pool has {len(pool)} rows; lower the quota or merge the category."
            )

    return items


def check(items: list, quota: dict, expected_n: int, label: str) -> None:
    """QC assertions. These are the conditions the toy-data analysis had to discover after the fact."""
    assert len(items) == expected_n, f"{label}: expected {expected_n} items, got {len(items)}"

    questions = [i["question"] for i in items]
    assert len(set(questions)) == len(questions), f"{label}: duplicate questions"

    all_lures = [l for i in items for l in i["lures"]]
    assert len(set(all_lures)) == len(all_lures), f"{label}: duplicate lures across items"

    for i in items:
        assert len(i["lures"]) == N_LURES, f"{label}: {i['item_id']} has {len(i['lures'])} lures"
        assert i["correct"].strip().lower() != REFUSAL, f"{label}: {i['item_id']} is a refusal"

    counts = Counter(i["category"] for i in items)
    for quota_key, n_wanted in quota.items():
        got = sum(counts[c] for c in quota_key.split("|"))
        assert got == n_wanted, f"{label}: {quota_key} = {got}, expected {n_wanted}"

    # Lures must not be a degenerate short/long tell -- a length gap here would let the model
    # (and any downstream probe) separate true from false on length alone.
    mean_correct = sum(word_count(i["correct"]) for i in items) / len(items)
    mean_lure = sum(word_count(l) for i in items for l in i["lures"]) / (len(items) * N_LURES)
    gap = abs(mean_correct - mean_lure)
    assert gap <= MAX_LURE_LENGTH_GAP, (
        f"{label}: lure/correct length gap {gap:.1f} words exceeds {MAX_LURE_LENGTH_GAP}"
    )

    print(f"\n{label}: {len(items)} items, all QC assertions passed")
    print(f"  mean words -- correct {mean_correct:.1f} | lure {mean_lure:.1f} (gap {gap:.1f})")
    for category, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {category:<24} {n:>3}")


def main() -> None:
    rng = random.Random(SEED)
    DATA_DIR.mkdir(exist_ok=True)
    candidates = load_candidates()

    high = select(candidates, HIGH_STAKES_QUOTA, "high", rng)
    low = select(candidates, LOW_STAKES_QUOTA, "low", rng)

    check(high, HIGH_STAKES_QUOTA, sum(HIGH_STAKES_QUOTA.values()), "High-stakes target set")
    check(low, LOW_STAKES_QUOTA, sum(LOW_STAKES_QUOTA.values()), "Low-stakes control set")

    overlap = {i["question"] for i in high} & {i["question"] for i in low}
    assert not overlap, f"Question appears in both sets: {overlap}"

    for items, name in [(high, "items_v2.json"), (low, "items_lowstakes.json")]:
        out = DATA_DIR / name
        with open(out, "w") as f:
            json.dump(items, f, indent=2)
        print(f"\nWrote {len(items)} items -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
