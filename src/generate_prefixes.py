"""Step 2: generate the three-class behavioural prefixes.

    # pilot first -- 16 per class, then read a sample by eye before committing
    /opt/conda/envs/talktuner-gpu/bin/python src/generate_prefixes.py --n-per-class 16 \
        --out data/prefixes_pilot.json

    # full run
    /opt/conda/envs/talktuner-gpu/bin/python src/generate_prefixes.py --n-per-class 160 \
        --out data/prefixes_v2.json

Prompts come from src/prefix_prompts.py, QC from src/prefix_qc.py. This file is only the driver:
batching, parsing, timing, and the acceptance report.

Generator is Qwen2.5-14B-Instruct in bf16 at batch 16 -- NOT the toy run's 32B in 4-bit NF4, which
measured 3.0 minutes per usable conversation (see outputs/toy_run_stats.txt). Note the tokenizer
uses LEFT padding here: this is generation, unlike scoring in subject_model.py, which uses right.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prefix_prompts import BATCH_SIZE, CLASSES, GEN_KWARGS, GEN_MODEL_ID, TOPICS, build_prompt
from prefix_qc import gate_a, gate_b, is_usable, marker_table, n_exchanges, parse_turns

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_jobs(n_per_class, seed):
    """One job per conversation, topics balanced within each class, classes interleaved so that
    every batch is mixed (a batch of one class would confound batch effects with class)."""
    rng = random.Random(seed)
    jobs = []
    for label in CLASSES:
        topics = [TOPICS[i % len(TOPICS)] for i in range(n_per_class)]
        rng.shuffle(topics)
        jobs += [{"label": label, "topic": t} for t in topics]
    rng.shuffle(jobs)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=160)
    ap.add_argument("--out", default="data/prefixes_v2.json")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--model", default=GEN_MODEL_ID)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    jobs = build_jobs(args.n_per_class, args.seed)
    print("Generating %d conversations (%d per class) with %s"
          % (len(jobs), args.n_per_class, args.model))

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                      # generation, unlike scoring
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.eval()
    print("  loaded in %.0fs | %.1f GB allocated | batch %d\n"
          % (time.time() - t0, torch.cuda.memory_allocated() / 1e9, args.batch_size))

    prefixes = []
    t_start = time.time()
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start:start + args.batch_size]
        chats = [tok.apply_chat_template(
                    [{"role": "user", "content": build_prompt(j["topic"], j["label"])}],
                    tokenize=False, add_generation_prompt=True) for j in batch]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")

        with torch.no_grad():
            out = model.generate(**enc, pad_token_id=tok.pad_token_id, **GEN_KWARGS)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

        for job, raw in zip(batch, texts):
            turns = parse_turns(raw)
            prefixes.append({
                "prefix_id": "%s_%03d" % (job["label"], len(prefixes)),
                "label": job["label"],
                "topic": job["topic"],
                "turns": turns,
                "n_turns": len(turns),
                "n_exchanges": n_exchanges(turns),
                "usable": is_usable(turns, raw),
                "raw_text": raw.strip(),
            })

        done, elapsed = len(prefixes), time.time() - t_start
        print("  %3d/%d | %.1f convo/min | %.0f%% usable so far"
              % (done, len(jobs), 60 * done / elapsed,
                 100 * sum(p["usable"] for p in prefixes) / done))

    elapsed = time.time() - t_start
    usable = [p for p in prefixes if p["usable"]]
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(exist_ok=True)
    json.dump(prefixes, open(out_path, "w"), indent=2)

    print("\n" + "=" * 78)
    print("Generation: %d conversations in %.1f min (%.2f min each)"
          % (len(prefixes), elapsed / 60, elapsed / 60 / len(prefixes)))
    print("Usable (4-6 complete exchanges, no trait-word leak): %d/%d (%.1f%%)"
          % (len(usable), len(prefixes), 100 * len(usable) / len(prefixes)))
    print("Toy run for comparison: 1.10 min each, 37.5%% usable under its own rule")
    print("Wrote %s" % out_path)

    print("\n" + "=" * 78)
    print("Markers by class")
    for label, stats in marker_table(usable).items():
        print("  %-10s %s" % (label, json.dumps(stats)))

    print("\nGate A (construct validity): %s" % json.dumps(gate_a(usable)))
    print("Gate B (length matching, pre-match): %s" % json.dumps(gate_b(usable)))
    print("\nNext: read a sample by eye, then see docs/PLAN_TECHNICAL.md Step 2 for the top-up rule.")


if __name__ == "__main__":
    main()
