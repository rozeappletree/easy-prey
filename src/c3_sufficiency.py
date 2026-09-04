"""Step 8 = C3: sufficiency M -> Y -- GATE G3

    /opt/conda/envs/talktuner-gpu/bin/python src/c3_sufficiency.py

Does adding the credulity direction, under a NEUTRAL prompt, shift Delta toward the lure -- with a
monotone dose-response as the steering strength (alpha) increases? This is sufficiency: the
direction CAN drive the behaviour. It is not necessity (Step 9/C4 tests that) -- see the spare-key
argument in nb/01: a key that can open a door is not proof it opened it this morning.

Design (WORKLOG entry 42):
  - X is the COMBINED condition (Gate G5): a neutral stated-persona sentence + a neutral
    behavioural conversation, then the item's question. 5 fixed neutral stimuli (from
    data/c2_stimuli.json) are used per item, matching C1's pattern of averaging over several
    prefixes to reduce prefix-specific noise, and averaged in the same way here.
  - Layer band: 5 decoder-block outputs centred on L* (best_layer=9 from Step 7). Because
    hidden_states[i] is the INPUT to block i (subject_model.py's documented convention), the
    direction was found in hidden_states[9] = the OUTPUT of block 8 -- so the band of block
    indices to hook is centred on 8, not 9. Off-by-one here would silently steer the wrong point
    in the network, so this is stated explicitly rather than left implicit.
  - Five directions from c2_directions.npz: diff-in-means, probe weights, verbosity (T4 control),
    random (T5 control), orthogonalized (diff-in-means with verbosity component removed).
  - Health check: logP(correct) + logP(lure) must not collapse relative to the random-direction
    control at the same alpha. Any (item, alpha, direction) cell failing this is excluded and
    reported as excluded, never silently averaged in (DESIGN.md Sec 4, GATES.md analysis
    commitments).
  - Writes incremental JSONL as it goes (checkpointing owed since entry 34/38 -- C3 is ~9x more
    passes than the largest single run so far, the first place losing progress would actually hurt).

Gate G3 (GATES.md): monotone dose-response, correct sign, health checks intact.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from subject_model import (encode_answer, encode_prompt, hooked, load_subject,
                           random_direction, score_answers, steering_hook)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = REPO_ROOT / "data", REPO_ROOT / "outputs"
SUBJECT_MODEL = "NousResearch/Llama-2-13b-chat-hf"
SEED = 0

ALPHAS = [-4, -2, -1, 0, 1, 2, 4]
DIRECTIONS = ["diff_in_means", "probe_weights", "verbosity", "random", "orthogonalized"]
N_NEUTRAL_STIMULI = 5
BAND_WIDTH = 5


def neutral_messages(stimulus):
    return [{"role": "system", "content": stimulus["template_text"]}] + stimulus["turns"]


def health_ok(logp_sum, random_range):
    lo, hi = random_range
    return lo <= logp_sum <= hi


def main():
    items = json.load(open(DATA / "items_v2.json"))
    stimuli = json.load(open(DATA / "c2_stimuli.json"))
    neutral_pool = [s for s in stimuli if s["label"] == "neutral" and s["split"] == "train"]
    rng = np.random.default_rng(SEED)
    neutral_stimuli = [neutral_pool[i] for i in
                       rng.choice(len(neutral_pool), N_NEUTRAL_STIMULI, replace=False)]
    print("Using %d fixed neutral stimuli: %s"
          % (len(neutral_stimuli), [s["stimulus_id"] for s in neutral_stimuli]))

    d = np.load(DATA / "c2_directions.npz")
    best_layer_hs_idx = int(d["best_layer"])         # hidden_states index the direction lives in
    block_idx = best_layer_hs_idx - 1                # hidden_states[i] = INPUT to block i = OUTPUT of block i-1
    half = BAND_WIDTH // 2
    n_blocks = 40
    layer_band = [i for i in range(block_idx - half, block_idx + half + 1) if 0 <= i < n_blocks]
    print("Direction anchored at hidden_states[%d] -> steering block outputs %s"
          % (best_layer_hs_idx, layer_band))

    directions = {}
    for name in DIRECTIONS:
        vec = d["random"] if name == "random" else d[name]
        directions[name] = torch.tensor(vec, dtype=torch.bfloat16)

    model, tok = load_subject(SUBJECT_MODEL, dtype=torch.bfloat16, device="cuda")
    print("Loaded %s | %.1f GB allocated\n" % (SUBJECT_MODEL, torch.cuda.memory_allocated() / 1e9))

    rows_path = DATA / "c3_rows.jsonl"
    rows_f = open(rows_path, "w")

    t0 = time.time()
    n_done = 0
    n_configs = len(neutral_stimuli) * len(DIRECTIONS) * len(ALPHAS)
    n_rows_total = len(items) * n_configs
    print("Plan: %d stimuli x %d directions x %d alphas = %d configs, "
          "each scoring all %d items batched together (%d rows total)\n"
          % (len(neutral_stimuli), len(DIRECTIONS), len(ALPHAS), n_configs, len(items), n_rows_total))

    config_i = 0
    for stim in neutral_stimuli:
        # Fixed for this stimulus, independent of direction/alpha -- built once, reused ~35x.
        prompt_ids, item_ids = [], []
        for item in items:
            prompt_ids.append(encode_prompt(
                tok, neutral_messages(stim) + [{"role": "user", "content": item["question"]}]))
            item_ids.append(item["item_id"])
        cand_ids_by_item = {item["item_id"]: [encode_answer(tok, c)
                                              for c in [item["correct"]] + item["lures"]]
                            for item in items}

        # Flatten to one (prompt, answer) pair per (item, candidate) for a single batched call.
        flat_prompts, flat_answers, flat_index = [], [], []
        for pid, iid in zip(prompt_ids, item_ids):
            for cand in cand_ids_by_item[iid]:
                flat_prompts.append(pid)
                flat_answers.append(cand)
                flat_index.append(iid)

        for direction_name in DIRECTIONS:
            direction = directions[direction_name]
            for alpha in ALPHAS:
                config_i += 1
                if alpha == 0:
                    scores = score_answers(model, tok, flat_prompts, flat_answers, batch_size=24)
                else:
                    with hooked(model, layer_band, steering_hook(direction, alpha)):
                        scores = score_answers(model, tok, flat_prompts, flat_answers, batch_size=24)

                # Regroup the flat score list back into 3 candidates (correct + 2 lures) per item.
                by_item = {}
                for iid, sc in zip(flat_index, scores):
                    by_item.setdefault(iid, []).append(sc)
                for iid, sc in by_item.items():
                    correct_lp, lure_lps = sc[0], sc[1:]
                    delta = sum(lure_lps) / len(lure_lps) - correct_lp
                    logp_sum = correct_lp + sum(lure_lps) / len(lure_lps)
                    rows_f.write(json.dumps({
                        "item_id": iid, "stimulus_id": stim["stimulus_id"],
                        "direction": direction_name, "alpha": alpha,
                        "delta": delta, "logp_sum": logp_sum,
                    }) + "\n")
                    n_done += 1
                rows_f.flush()

                elapsed = time.time() - t0
                rate = config_i / elapsed
                eta_min = (n_configs - config_i) / rate / 60 if rate > 0 else float("nan")
                print("  config %d/%d (%s, alpha=%+d) | %d rows so far | ETA %.1f min"
                      % (config_i, n_configs, direction_name, alpha, n_done, eta_min))

    rows_f.close()
    print("\nDone: %d rows in %.1f min -> %s" % (n_done, (time.time() - t0) / 60, rows_path.name))


if __name__ == "__main__":
    main()
