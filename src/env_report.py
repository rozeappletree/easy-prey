"""Environment and materials audit for the Family-7 mediation project.

    /opt/conda/envs/talktuner-gpu/bin/python src/env_report.py

Records the facts the design depends on, so they can be re-checked rather than trusted:
GPU, interpreter, packages, cached model completeness, the Llama-2 chat-template gap, and the
TruthfulQA category pool. See docs/WORKLOG.md entries 3, 4, 5 and 11.
"""

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

CACHE = Path.home() / ".cache" / "huggingface" / "hub"
SUBJECT = "models--NousResearch--Llama-2-13b-chat-hf"
PACKAGES = ["torch", "transformers", "accelerate", "bitsandbytes", "nnsight",
            "scikit-learn", "scipy", "numpy", "pandas", "matplotlib", "statsmodels",
            "transformer-lens"]

HIGH_STAKES = ["Misconceptions", "Law", "Health", "Economics", "Nutrition", "Finance"]
LOW_STAKES = ["Fiction", "Myths and Fairytales", "Proverbs", "Misquotations"]


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def gpu():
    rule("GPU")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        print("  " + out.stdout.strip())
    except Exception as exc:                                    # noqa: BLE001
        print("  nvidia-smi unavailable: %s" % exc)
    try:
        import torch
        print("  torch.cuda.is_available(): %s" % torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print("  %s | %.1f GB total | capability %d.%d" % (p.name, p.total_memory / 1e9,
                                                               p.major, p.minor))
    except ImportError:
        print("  torch NOT INSTALLED in this interpreter")


def interpreter():
    rule("Interpreter and packages")
    print("  executable: %s" % sys.executable)
    print("  version:    %s" % sys.version.split()[0])
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:                                          # py<3.8
        return
    for name in PACKAGES:
        try:
            print("  %-16s %s" % (name, version(name)))
        except PackageNotFoundError:
            print("  %-16s NOT INSTALLED" % name)


def models():
    rule("Hugging Face cache")
    if not CACHE.exists():
        print("  no cache at %s" % CACHE)
        return
    for d in sorted(CACHE.glob("models--*")) + sorted(CACHE.glob("datasets--*")):
        # snapshots/ holds symlinks into blobs/; following them would double-count
        size = sum(f.stat().st_size for f in d.rglob("*")
                   if f.is_file() and not f.is_symlink())
        incomplete = len(list(d.rglob("*.incomplete")))
        flag = "  <-- INCOMPLETE DOWNLOAD" if incomplete else ""
        print("  %-52s %7.1f GB%s" % (d.name, size / 1e9, flag))

    rule("Subject model config and the chat-template gap")
    cfgs = glob.glob(str(CACHE / SUBJECT / "snapshots" / "*" / "config.json"))
    if not cfgs:
        print("  %s not cached" % SUBJECT)
        return
    cfg = json.load(open(cfgs[0]))
    keys = ["num_hidden_layers", "hidden_size", "num_attention_heads",
            "max_position_embeddings", "torch_dtype", "vocab_size"]
    for k in keys:
        print("  %-26s %s" % (k, cfg.get(k)))
    tok = json.load(open(cfgs[0].replace("config.json", "tokenizer_config.json")))
    has_template = "chat_template" in tok and tok["chat_template"]
    print("  %-26s %s" % ("chat_template present", bool(has_template)))
    if not has_template:
        print("\n  ACTION REQUIRED: transformers 4.45 removed the built-in Llama-2 default, so")
        print("  apply_chat_template() will raise. Set tokenizer.chat_template explicitly at load")
        print("  and assert the rendered prompt contains '[INST]'. See WORKLOG.md entry 5.")


def truthfulqa():
    rule("TruthfulQA pool (after filtering)")
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError as exc:                                   # noqa: BLE001
        print("  unavailable: %s" % exc)
        return
    path = hf_hub_download("domenicrosati/TruthfulQA", "TruthfulQA.csv", repo_type="dataset")
    tqa = pd.read_csv(path)
    tqa["inc"] = tqa["Incorrect Answers"].fillna("").apply(
        lambda s: [a.strip() for a in str(s).split(";") if a.strip()])
    answerable = tqa["Best Answer"].str.strip().str.lower() != "i have no comment"
    has_lures = tqa["inc"].apply(len) >= 2
    ok = tqa[answerable & has_lures]
    print("  %d rows -> %d usable (%d refusals, %d with <2 lures dropped)"
          % (len(tqa), len(ok), (~answerable).sum(), (answerable & ~has_lures).sum()))
    counts = ok["Category"].value_counts()
    print("\n  high-stakes pool:")
    for c in HIGH_STAKES:
        print("    %-24s %3d" % (c, counts.get(c, 0)))
    print("  low-stakes pool:")
    for c in LOW_STAKES:
        print("    %-24s %3d" % (c, counts.get(c, 0)))
    print("\n  Note: Science (%d), Statistics (%d) and Misinformation (%d) are too small to use."
          % (counts.get("Science", 0), counts.get("Statistics", 0), counts.get("Misinformation", 0)))


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    gpu()
    interpreter()
    models()
    truthfulqa()
    print()
