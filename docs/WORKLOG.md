# Work log — Family-7 mediation project

A running record of **what was done, why, how, what it showed, what followed from it, and what
happens next.** Chronological. Each entry is self-contained so you can cite it in the writeup's
methods section without reconstructing the reasoning six days later.

**Companion documents:** [`../DESIGN.md`](../DESIGN.md) (the reasoned design),
[`PLAN_ELI5.md`](PLAN_ELI5.md) and [`PLAN_TECHNICAL.md`](PLAN_TECHNICAL.md) (the execution steps),
[`../GATES.md`](../GATES.md) (the preregistration).

## Evidence on disk

Every number quoted below is reproducible. `outputs/` holds the captured runs; regenerate the whole
directory with one CPU-only command that costs none of the GPU budget:

```bash
bash src/run_all_checks.sh
```

| Output | Produced by | Entries it backs |
|---|---|---|
| [`../outputs/env_report.txt`](../outputs/env_report.txt) | `src/env_report.py` | 3, 4, 5, 11 |
| [`../outputs/toy_run_stats.txt`](../outputs/toy_run_stats.txt) | `src/toy_run_stats.py` | 2, 6, 9, 10 |
| [`../outputs/build_items.txt`](../outputs/build_items.txt) | `src/build_items.py` | 15 |
| [`../outputs/prefix_qc_selfcheck.txt`](../outputs/prefix_qc_selfcheck.txt) | `src/prefix_qc.py` | 18, 19 |

Each file carries a header recording the command and the UTC timestamp, so a stale artifact is
obvious at a glance.

---

## Session 1 — 2026-09-04

### 1. Read the four reference documents

- **What:** Read the definition ladder (D1–D9), the dataset-generation design, the original
  family-7 plan, and the GPU-constrained redesign.
- **Why:** The claim to be tested is D8 (mediation). Everything else in those documents is either a
  weaker rung on the same ladder or an earlier draft of the same experiment — worth knowing so the
  new design doesn't silently repeat a decision that was already reconsidered.
- **Found:** The four agree on the causal chain (X → M → Y) and on the ablation machinery, but they
  disagree about hardware and they under-specify the *controls*. The GPU redesign in particular
  asserts "compute is not your constraint" — an assertion that turned out to be wrong once measured
  (entry 6).
- **Conclusion:** Keep the chain and the ablation design. Rebuild the controls and the throughput
  assumptions from measurement rather than inheritance.
- **Next step:** → entry 2, check the claims against what the toy run actually did.

### 2. Mapped the repository

- **What:** Listed the tree; extracted the source and outputs of `nb/generate_toy_data.ipynb` and
  `nb/analyze_toy_data.ipynb`; inspected `data/items.json`, `data/prefixes.json`,
  `data/toy_dataset.json`.
- **Why:** The toy run is real evidence about what this pipeline does in practice. Reference docs
  describe intentions; the notebooks record outcomes.
- **How:** Parsed the `.ipynb` JSON directly rather than opening the notebooks, so cell *outputs*
  (the timing and yield numbers) came through alongside the source.
- **Found:** A working generation pipeline — 30 items, 80 prefixes, correct topic balance, zero
  trait-word leakage, textually unique generations. Plus four concrete defects, all of which the
  user's own analysis notebook had already identified.
- **Conclusion:** Reuse the item-loading and topic-balancing logic; rebuild the prefix generation.
  The four defects became design requirements (entries 7–10).
- **Next step:** → entries 3–5, establish what the machine can actually run, before designing
  around it. Evidence preserved in [`toy_run_stats.txt`](../outputs/toy_run_stats.txt) §3.

### 3. Probed the execution environment

- **What:** `nvidia-smi`; `python -VV`; package inventory in both conda environments.
- **Why:** A plan that names a library the machine doesn't have costs an hour at exactly the wrong
  moment.
- **Found** ([`env_report.txt`](../outputs/env_report.txt)):
  - L40S, **46 GB free**, idle.
  - The **default Python 3.14 env has no torch** — only numpy/pandas/sklearn/scipy.
  - `/opt/conda/envs/talktuner-gpu` (Python 3.9.13) has torch 2.4.1, transformers 4.45.1,
    accelerate, bitsandbytes, nnsight 0.3.6, sklearn, scipy, matplotlib.
  - **TransformerLens is not installed** in either.
- **Conclusion:** All work runs under `talktuner-gpu`. **Tooling decision: raw HF forward hooks.**
  Installing TransformerLens plus converting a 13B checkpoint buys nothing here — the experiment
  needs exactly two operations (read the residual stream at one position; add or subtract a vector
  inside a decoder layer), and `register_forward_hook` does both in ten lines. `statsmodels` is also
  absent, which later ruled out the crossed random-effects model (entry 14).
- **Next step:** every script in `src/` is invoked with the explicit interpreter path
  `/opt/conda/envs/talktuner-gpu/bin/python`. Never rely on `python` resolving correctly.

### 4. Audited the model cache

- **What:** Measured `~/.cache/huggingface/hub`; verified shard completeness; read the configs.
- **Why:** Download time is real time, and gated repos (Llama-3.x, Mistral) can block on a licence
  acceptance that needs a browser.
- **Found:** `Qwen2.5-32B-Instruct` (62 GB) and `NousResearch/Llama-2-13b-chat-hf` (25 GB, **3
  shards, zero `.incomplete` blobs**, 40 layers, d_model 5120, 4096 context, MHA, fp16 weights).
  129 GB disk free.
- **Conclusion:** **Subject model = Llama-2-13b-chat.** In order of weight:
  1. Already on disk — no download, no gating, no HF token.
  2. **Cross-family from the Qwen generator.** Personas written by Qwen may be idiosyncratically
     legible to Qwen (shared idiom, shared tokenizer). A Llama probe reading them removes that
     confound for free.
  3. 40 layers gives a real depth axis, and there is a published prior that user-attribute probes
     work in this specific model around layers 20–29 — which makes a **null** at the probing step
     informative rather than ambiguous.
  - Its age is not a problem: forced-choice log-prob scoring is unaffected by how refusal-prone a
    chat model is, because we never ask it to generate for the primary DV.
- **Next step:** download `Qwen2.5-14B-Instruct` (~28 GB) before Step 2 runs — it is the only model
  in the plan not yet cached. **Still outstanding.**

### 5. Checked the Llama-2 chat template

- **What:** Read `tokenizer_config.json` from the cached snapshot.
- **Why:** Prompt assembly is upstream of every number in the project.
- **Found:** The NousResearch mirror ships **no `chat_template`**, and transformers 4.45 removed the
  built-in default. `apply_chat_template` will raise.
- **Conclusion:** Set the template explicitly at load time and **assert the rendered prompt contains
  `[INST]`**. Cheap insurance against the worst class of failure here — a silently malformed prompt
  produces plausible numbers, not an exception.
- **Next step:** implement in the Step 4 scoring module, alongside the four correctness tests.
  `env_report.py` prints an ACTION REQUIRED banner while this is unresolved, so it cannot be
  forgotten.

### 6. Measured the real generation throughput ← *the finding that reshaped the plan*

- **What:** Extracted the timing output buried in the toy notebook's generation cell.
- **Why:** The GPU redesign document budgeted the experiment on an assumed ~2,500 tok/s. Assumed
  throughput is not throughput.
- **Found** ([`toy_run_stats.txt`](../outputs/toy_run_stats.txt) §1): **80 conversations in 89.7
  minutes** with Qwen2.5-32B in 4-bit NF4 at `batch_size=4` — **1.1 min per conversation**, and
  because only 30 of the 80 were usable, **3.0 minutes per usable conversation**. Estimated from the
  generated text, ~4.3 aggregate tokens/second.
- **Reasoning:** Three compounding causes, all fixable.
  1. NF4 dequantization dominates decode — every generated token re-expands 19 GB of 4-bit weights,
     bypassing the tensor cores.
  2. Batch 4 is far below what fits; ~18 GB of VRAM sat idle.
  3. 62.5% of the output was discarded.
- **Conclusion — this inverts the reference document's framing.** Scaled to the ~480 prefixes the
  design needs, generation would consume **about 9 of the 20 hours**, while the *entire*
  five-experiment forward-pass workload is ~4 GPU-hours. So: **generator switches to
  Qwen2.5-14B-Instruct in bf16 at batch 16.** Generation, not analysis, was the bottleneck — and it
  was invisible until someone read the progress bar.
- **Next step:** the extrapolation is an estimate, not a promise. Step 2 carries a **hard checkpoint
  after the first 48 conversations**: measure the real rate, and if the projection exceeds 90
  minutes, cut to 100/class rather than letting the block overrun.

### 7. Defect → design requirement: the length confound

- **Found** (from the user's analysis): Credulous user turns average **12.2 words**, skeptical
  **18.1**. Assistant turns differ too (22.6 vs 25.9).
- **Reasoning:** A probe trained on these activations can score high by reading *how much text
  preceded it*. The "credulity direction" would then be a verbosity direction — and the mediation
  test would come back clean while measuring the wrong thing. This is the most dangerous failure
  mode in the project, because **it does not look like a failure.**
- **Conclusion — four independent countermeasures**, on the principle that one defence against the
  most likely confound is not enough:
  1. Length-matched subsampling across all three classes (decile bins, Kruskal–Wallis gate).
  2. An explicit surface-feature probe baseline the real probe must beat by ≥5 points.
  3. Probe accuracy reported stratified by length tercile.
  4. A **verbosity direction** built by median-splitting on length *within* each class — so it
     tracks length by construction — used as a steering control, an ablation control, and as the
     thing the credulity direction gets orthogonalized against.
- **Next step:** → entry 19, which measured what matching actually costs and forced a fifth
  countermeasure.

### 8. Defect → design requirement: construct drift

- **Found:** Evidence-seeking phrases appear **0.09 times per skeptical user turn** and 0.00 for
  credulous. Question marks: 1.16 vs 0.68. Agreement words: 0.26 vs 0.69.
- **Reasoning:** The manipulation worked, but it produced **inquisitive vs passive**, not
  **skeptical vs credulous**. That is a different construct, and it would silently have become the
  construct the results were about.
- **Conclusion:** Generation now *requires* ≥2 explicit verification acts per skeptical
  conversation, enforced by regex as an **acceptance gate** with regeneration on failure — and the
  marker rates get published either way so a reader can judge what the personas actually are.
- **Next step:** → entries 16–17 implement the requirement and the gate; entry 19 shows the drift is
  worse than this per-turn rate suggested.

### 9. Defect → design requirement: the 37.5% yield

- **Found** ([`toy_run_stats.txt`](../outputs/toy_run_stats.txt) §2): 94% of malformed generations
  ended on clean sentence punctuation and were only ~20% shorter than well-formed ones. Turn-count
  mode was 10 lines (5 exchanges), not the requested 12.
- **Reasoning:** That is **clean early stopping, not truncation.** A truncated generation stops
  mid-sentence; these stopped cleanly, slightly short. Raising `max_new_tokens` would do nothing.
  Likely causes: `repetition_penalty=1.15` (which raises the relative probability of EOS in a
  deliberately repetitive format) and a free-form "6 turns" instruction giving the model no counter.
- **Conclusion:** Numbered turn scaffold, `repetition_penalty=1.0`, and **accept 4–6 exchanges**
  recording `n_turns` as a covariate. A 5-exchange conversation is a shorter sample, not a broken
  one — and length is being controlled for anyway.
- **Next step:** → entry 19, which measured which of those three changes actually does the work.
  (It is not the one this entry emphasised.)

### 10. Defect → design requirement: no neutral class, and Finance n=1

- **Found** ([`toy_run_stats.txt`](../outputs/toy_run_stats.txt) §3): The toy set has two prefix
  classes and, from random sampling, exactly **one** finance item and two nutrition items.
- **Reasoning:** Without a neutral middle you cannot distinguish "credulous users get lied to more"
  from "skeptical users get corrected more" — different phenomena, and only the first supports the
  framing. And a category with n=1 supports no category-level statement at all.
- **Conclusion:** Three prefix classes, and a fixed per-category quota instead of random sampling.
- **Next step:** → entry 11 (count the pool), then entry 15 (implement the quota).

### 11. Counted the TruthfulQA pool

- **What:** Loaded the cached CSV; applied the intended filters; counted by category.
- **Why:** To set quotas from the actual pool rather than from a guess that fails at execution time.
- **Found** ([`env_report.txt`](../outputs/env_report.txt)): **722 of 817 rows** survive dropping
  `"I have no comment"` and requiring ≥2 incorrect answers. High-stakes pool: Misconceptions 89,
  Law 58, Health 54, Economics 31, Nutrition 16, Finance 9. Low-stakes: Fiction 26, Myths and
  Fairytales 19, Proverbs 15. *(Science 9, Statistics 5 and Misinformation 1 are too small to use.)*
- **Conclusion:** Quotas of 25/25/25/15/10 and 12/10/8 are all comfortably inside the pool. Finance
  (9) had to be **merged with Economics** to reach a usable stratum — it cannot stand alone.
- **Next step:** in the writeup, make **no finance-specific claim.** Finance survives only as 5 items
  inside a merged Economics+Finance stratum (entry 15).

### 12. Wrote `DESIGN.md`

- **What:** The reasoned design — claim, causal chain, threat model, materials, measures,
  interventions, controls, statistics, compute budget, schedule, failure branches.
- **Why:** So every later decision has a written justification, and so the 20-hour window is spent
  executing rather than deciding.
- **Three substantive additions over the reference documents:**
  1. **Cross-channel transfer.** Train the direction on Qwen-generated *behavioural* prefixes, test
     on hand-written *stated-persona* prompts, and vice versa. Different author, different
     vocabulary, same construct. This is the strongest available evidence that the direction encodes
     credulity rather than a generator's dialect — and it is nearly free.
  2. **The verbosity direction and its orthogonalization** (entry 7).
  3. **The α=0 identity test** and three companion assertions, promoted from folklore to a gate.
- **One deletion:** the PMI / minimal-prompt correction. Every contrast is within-item across
  conditions, so the correction term is a per-item constant and **cancels exactly**. Computed once
  for the absolute headroom check and never again — one fewer condition in the budget.
- **Next step:** `DESIGN.md` is the *reasoning* reference and should be amended whenever a
  measurement contradicts it — as happened twice already (entries 19 and 21).

### 13. Wrote the two step-by-step plans

- **What:** [`PLAN_ELI5.md`](PLAN_ELI5.md) and [`PLAN_TECHNICAL.md`](PLAN_TECHNICAL.md), sharing one
  numbered spine (Step 0–12) so any step can be read at either level of detail.
- **Why:** `DESIGN.md` is organised by *topic*; execution needs an ordered list of actions. And an
  argument that cannot survive being explained plainly usually has a gap in it — writing the ELI5
  version is a check on the technical one, not a translation of it.
- **Reasoning behind the central analogy** (a dial inside the model's head): mediation is a claim
  about a *chain*, and the four things you must show to establish a chain — effect exists, dial
  exists, turning it works, holding it still blocks the effect — map exactly onto C1/C2/C3/C4. The
  spare-key point (a key that *can* open a door isn't proof it *did*) is why step 9 exists and
  step 8 isn't enough.
- **Next step:** keep the two files in lockstep. Any change to a threshold or a count must land in
  both, plus `GATES.md` if it is a gate.

### 14. Wrote `GATES.md` (preregistration)

- **What:** G0–G4 thresholds, the two primary contrasts, the dataset acceptance gates, Plan B, and a
  results table containing only blanks.
- **Why:** The characteristic way a 20-hour project dies is that at hour 12 the threshold is quietly
  renegotiated. A committed file makes that visible.
- **Two decisions recorded there worth repeating:**
  - **Statistics:** per-item paired Wilcoxon plus a permutation test over items — **not** a crossed
    random-effects model. `statsmodels` supports only one grouping factor and `lme4`/`pymer4` are
    not installed. A library constraint, but *also* the correct test: with items as the unit and
    prefixes averaged within item, the pairing is the design.
  - **Plan B, decided in advance:** if the behavioural prefixes show no total effect but the ceiling
    prompt does, swap the manipulation to stated personas and run the mediation chain unchanged. The
    claim weakens honestly to "represents *asserted* credulity" — still a real mediation result, and
    far better than stopping.
- **Next step:** **`git commit GATES.md` before the subject model is ever loaded.** That commit
  timestamp *is* the preregistration, which is why it is yours to make and not mine. **Still
  outstanding.**

### 15. Built and ran the item bank — `src/build_items.py`

- **What:** Loads the cached TruthfulQA CSV, filters, applies fixed per-category quotas, writes
  `data/items_v2.json` (100 high-stakes) and `data/items_lowstakes.json` (30 low-stakes), and
  asserts the QC conditions.
- **How:** Greedy fill per stratum from a seeded shuffle, **skipping any item that would duplicate a
  question or reuse a lure already taken.** Quota keys support merged categories
  (`Economics|Finance`). Reuses the `split_answers` logic from `generate_toy_data.ipynb`.
- **Why the assertions live in the script rather than a notebook:** the toy run discovered its
  problems *after* generating on top of them. These are the same checks, moved upstream so they fail
  before they cost anything.
- **Found** ([`build_items.txt`](../outputs/build_items.txt)): all assertions pass. 100 high-stakes
  items (Misconceptions 25, Law 25, Health 25, Economics 10 + Finance 5, Nutrition 10) and 30
  low-stakes (Fiction 12, Myths and Fairytales 10, Proverbs 8). Correct/lure mean word counts
  10.7 vs 9.6 (high-stakes) and 9.2 vs 9.2 (low-stakes).
- **Verified:** deterministic across two runs (identical MD5 of the serialised output); no duplicate
  questions; **no lure text shared between items**, so the same falsehood is never double-counted;
  exactly 2 lures per item; no refusals; no overlap between the two sets.
- **Conclusion:** The 1.1-word lure/correct gap matters more than it looks — it means the lures are
  same-register alternatives, not a degenerate short/long tell that the model (or any downstream
  probe) could exploit without knowing anything.
- **Next step:** nothing further for Step 1. The item bank is frozen; if it is ever regenerated,
  re-check the MD5, because every downstream `item_id` depends on it.

---

## Session 2 — 2026-09-04 (continued)

### 16. Wrote the v2 generation prompts — `src/prefix_prompts.py`

- **What:** 40 everyday topics (the toy run's 30 plus 10), three class instructions, the numbered
  turn scaffold, the forbidden-word list, and the sampling settings.
- **How, and why each piece is there:**
  - **Numbered scaffold** + an explicit "produce all N exchanges, do not stop early" — against the
    early-stopping mode found in entry 9.
  - **`repetition_penalty` 1.0**, down from the toy run's 1.15. In a deliberately repetitive format
    (`Turn 1 - User:`, `Turn 2 - User:`, …) a repetition penalty raises the *relative* probability
    of EOS, which is the plausible mechanism behind the early stopping.
  - **Countable act requirements** — "at least TWO explicit verification acts", with examples — so
    the construct is enforceable by regex rather than hoped for (entry 8).
  - **A neutral class** using the same scaffold and topics with the trait instruction removed.
- **Next step:** these prompts have never been run against a model. The first 48 generations are a
  **pilot**, not production: read a sample by eye, confirm the scaffold is obeyed and the personas
  read as intended, then run Gate A before generating the remaining ~430.

### 17. Wrote the QC / gate module — `src/prefix_qc.py`

- **What:** Conversation parsing, the verification- and acceptance-act regexes, per-class marker
  statistics, decile-bin length matching, and the two acceptance gates from `GATES.md`.
- **Why a separate module:** these gates decide whether a generated batch is usable at all. They
  must be runnable and inspectable *before* a batch is generated, not discovered afterwards — which
  is exactly the sequence that produced the toy run's problems.
- **Design note on the verification regex:** it matches verification *acts* (asking for provenance,
  naming a check the user will perform), deliberately **not** a topic-word list. Words like "study",
  "research" and "evidence" appear freely in a merely inquisitive turn — and
  inquisitive-instead-of-verifying is precisely the drift this gate exists to catch.
- **Next step:** the regexes are tuned against Qwen2.5-32B's phrasing. If the 14B generator phrases
  verification differently, Gate A will fail for the wrong reason — so on the pilot batch, **read
  the non-matching skeptical conversations by hand** before concluding the persona is weak. Widen
  the regex if it is under-matching, and record the change here.

### 18. Validated the QC module against the toy data ← *regression test on known ground truth*

- **What:** Ran `prefix_qc.py` over the 80 toy prefixes, re-parsing from `raw_text`, and compared
  against the statistics already published in `nb/analyze_toy_data.ipynb`.
- **Why:** A QC module that gates the real dataset must first be shown to measure correctly. The toy
  data is the only dataset here with independently published numbers, which makes it the only
  available ground truth.
- **Found** ([`prefix_qc_selfcheck.txt`](../outputs/prefix_qc_selfcheck.txt)):
  - **Reproduced exactly** — turn-count distribution (8:3, 9:5, 10:40, 11:2, 12:30); yield under the
    toy rule 30/80 = 37.5%; question marks 0.68 vs 1.16 per user turn; the notebook's looser
    evidence-phrase rate 0.00 vs 0.09.
  - **Reproduced to 0.1 words** — user words per turn 12.1 vs 18.1 (notebook 12.2 vs 18.1);
    assistant 25.8 (notebook 25.9). Expected: this module re-parses `raw_text` from scratch, whereas
    the notebook read the stored `turns` field.
  - **Both gates correctly FAIL on the toy data**, which is the outcome that validates them.
- **Next step:** re-run this self-check after any regex change (entry 17). It is the only guard
  against a "fix" that quietly breaks the counting path.

### 19. Three findings from that run, one of which changed the design

1. **The yield problem was mostly a filtering artifact, not a generation problem.** Applying the v2
   acceptance rule (4–6 complete exchanges) to the *unchanged* toy generations lifts usable yield
   from **37.5% to 91.2%**. Relaxing the turn filter is the main lever; the scaffold and the
   repetition-penalty change are a bonus on top. *(This corrects entry 9 and the first draft of
   `DESIGN.md`, both of which credited the scaffold first.)*
2. **Construct drift is worse than the per-turn rate suggested.** Only **2.5%** of toy skeptical
   conversations contain ≥2 explicit verification acts, against the gate's 80% threshold. The
   0.09-phrases-per-turn figure understated how far the persona was from "verifying".
3. **Length matching is expensive — and this changed the plan.** The toy classes differ by **26.3%**
   in mean prefix length (Kruskal–Wallis p ≈ 0). Decile-bin matching brings them to within 4.1%
   (p = 0.75) — but it costs **65% of the sample**: 80 prefixes become 28.
   - **Reasoning:** at that rate, 160 generated per class would leave ~56 usable per class, below
     the 80/class fallback threshold in `GATES.md`. Generating more is the expensive fix; removing
     the gap at source is the cheap one.
   - **Conclusion — added an at-source length control:** every class's prompt now carries the *same*
     length rule (user messages ~15–25 words, assistant replies ~25–40), phrased so it cannot bias
     the persona: *"Do not let the user's manner change how long their messages are."* Length
     matching stays as the backstop and the gate is unchanged; the constraint just makes it cheap.
- **Next step:** on the pilot batch, **measure the residual length gap before generating the rest.**
  If the at-source control has worked, the classes should differ by well under 26%. Then apply the
  **top-up rule**: generate 160/class, run Gate B, and generate a further batch only if matching
  leaves <100/class. Generation is fast now, so a top-up costs minutes — better than inflating the
  initial request on speculation.

### 20. Captured the evidence — `outputs/` and `src/run_all_checks.sh`

- **What:** Added `src/env_report.py` and `src/toy_run_stats.py`, and a runner that regenerates all
  four output files with command-and-timestamp headers.
- **Why:** Until now the numbers in this log came from ad-hoc shell commands that existed only in a
  terminal scrollback. A methods section cannot cite a scrollback. Now every claim above has a file
  behind it, and one CPU-only command re-derives all of them.
- **Next step:** re-run `bash src/run_all_checks.sh` after every stage that changes the data, and
  before the writeup, so the outputs in the repo match the state being described.

### 21. Correction: the throughput figure was wrong

- **What:** `toy_run_stats.py` computes the token rate from the actual generated text rather than
  from an assumed tokens-per-conversation figure.
- **Found:** ~**4.3** aggregate tokens/second, not the ~8.9 quoted in the first drafts of
  `DESIGN.md` and `PLAN_TECHNICAL.md`. The earlier figure assumed ~600 tokens per conversation; the
  real generations average closer to 290. The extrapolation to a full run is therefore **~9 hours**,
  not 6–8.
- **Why it happened:** the original estimate was arithmetic on an assumption, not a measurement —
  the same error the reference document made with its 2,500 tok/s figure, one order of magnitude
  down.
- **Conclusion:** Corrected in both documents. **No decision changes** — the generator switch was
  already the conclusion, and the true number strengthens it. The exact, assumption-free figures are
  the wall-clock ones: **1.1 min per conversation, 3.0 min per usable conversation.** Prefer those
  in the writeup; quote the token rate only as an estimate.
- **Next step:** when the 14B generator runs, record its measured wall-clock rate in this log next
  to the 32B figure. That before/after pair is a genuine methods contribution — quantization choice
  as a research-throughput decision — and it costs nothing to capture.

---

## Standing decisions (change these only with a dated note)

| Decision | Rationale | Entry |
|---|---|---|
| Subject = Llama-2-13b-chat, **bf16, never quantized** | Cached, ungated, cross-family from the generator, 40 layers, published layer prior. NF4 error would live in the same place as a ~0.1-nat effect | 4 |
| Generator = Qwen2.5-14B bf16, batch 16 | Measured: NF4 at batch 4 costs 3.0 min per usable conversation | 6, 21 |
| Raw HF hooks, no TransformerLens | Not installed; two operations needed; a 13B conversion buys nothing | 3 |
| Unit of analysis = item, paired | Items are the repeated measure; prefixes are a nuisance factor | 14 |
| Three prefix classes, not two | Without neutral, the direction of the effect is unidentified | 10 |
| No PMI correction | Cancels exactly in within-item contrasts | 12 |
| Quota sampling, not random | Random sampling produced Finance n=1 | 10, 15 |
| At-source length control + matching as backstop | Matching alone costs 65% of the sample | 19 |

## Next steps, in order

1. **Commit `GATES.md`** — before any model loads. Yours to make; the timestamp is the
   preregistration. *(entry 14)*
2. **Download `Qwen2.5-14B-Instruct`** (~28 GB). The only uncached model in the plan. *(entry 4)*
3. **Write the 24 Channel-A stated-persona templates.** Deliberately yours: the cross-channel test
   in Step 7 is only strong if those sentences are not in a generator's voice. *(Step 3)*
4. **Pilot 48 conversations**, then in order: eyeball a sample, run Gate A, measure the residual
   length gap, measure the wall-clock rate, and only then generate the rest. *(entries 16, 17, 19)*
5. **Build the Step 4 scoring/hook module** with the four correctness tests on Qwen-0.5B — including
   the explicit Llama-2 chat template and the `[INST]` assertion. *(entry 5)*

## Things that could reasonably be revisited

- **Llama-2-13b vs a newer 8B.** Chosen for zero download risk and the layer prior. If G0 shows no
  log-prob headroom, `Qwen2.5-7B-Instruct` is the fallback — but note it is *same-family* with the
  generator, which costs the cross-family argument from entry 4.
- **Intervening at all positions** rather than only the prefix. Defensible (credulity information is
  read by attention from the question and answer positions too), but it is a real degree of freedom;
  the prefix-only variant is preregistered as a robustness check, and disagreement between them is a
  finding about *where* the representation lives.
- **Two lures per item.** Halves arbitrary-lure variance at 1.5× compute. If the budget tightens,
  drop to one lure in the α sweep only — never in C1 or C4.
- **The 15–25 word length rule.** Added on the basis of a measurement on *toy* data (entry 19). If
  the pilot shows the classes are naturally close in length under the new prompts, the rule is
  costing naturalness for nothing and should be relaxed.
