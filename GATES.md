# Preregistration — Family-7 mediation experiment

**Written 2026-09-04, before any subject model was loaded and before any Δ was computed.**
The git commit that adds this file is the preregistration. Do not amend the thresholds after
looking at data — if you decide a threshold was wrong, add a dated note below it saying so, and
report both the original and the revised analysis.

Design: [`DESIGN.md`](DESIGN.md) · execution: [`docs/PLAN_TECHNICAL.md`](docs/PLAN_TECHNICAL.md)

---

## Claim under test

> In `Llama-2-13b-chat`, a behavioural prefix in which the user displays credulity shifts the answer
> distribution toward TruthfulQA lures, and this shift is carried by a linear direction in the
> residual stream encoding user credulity.

`X` = prefix class → `M` = credulity direction at layer `L*` → `Y` = Δ.

---

## Primary contrasts — exactly two. Everything else is exploratory.

**P1 (Step 6, total effect).** Paired Wilcoxon signed-rank on per-item mean Δ, credulous vs
skeptical, n = 100 items. Report Cohen's `d_z` with a 10,000-sample bootstrap CI over items.

**P2 (Step 9, mediation).** Paired Wilcoxon on per-item mean Δ, credulous vs credulous-ablated,
n = 100 items, plus

```
PM = (Δ_credulous − Δ_credulous_ablated) / (Δ_credulous − Δ_neutral)
```

with a bootstrap CI over items.

Holm-correct across P1 and P2. Anything not listed here is exploratory and will be labelled as
exploratory in the writeup.

---

## Gates

| Gate | Step | Threshold | Action if failed |
|---|---|---|---|
| **G0** headroom | 5 | ~~mean Δ_neutral **< −0.05** nats/token **and** `argmax_false` **< 25%**~~ **AMENDED 2026-09-04, see below** | Retry with `Qwen2.5-7B-Instruct`. If that also fails → C0 branch (methods note on log-prob headroom) |
| **G1** total effect | 6 | `d_z` **≥ 0.30** | `0.15–0.30`: proceed, flagged underpowered, increase prefixes per item. **`< 0.15`: Plan B** (below) |
| **G2** encoding | 7 | best-layer held-out accuracy ≥ **best baseline + 5 points** | Reframe the direction as a text-style direction; retitle to "credulity probes are text classifiers"; still run Steps 8–9 |
| **G3** sufficiency | 8 | Monotone dose–response, sign as predicted, health checks intact | Report C3 null. Step 9 still runs — necessity without sufficiency is a real pattern |
| **G4** mediation | 9 | PM's CI **excludes 0** *and* random- and verbosity-ablation PM CIs **include 0** | Report PM ≈ 0 as a strong negative result against linear mediation |

### Amendment to G0 — 2026-09-04, after Step 5 ran, ratified by the user before proceeding

**Original text (kept above, struck through, not deleted):** mean Δ_neutral < −0.05 nats/token
*and* `argmax_false` < 25%.

**What happened:** on `Llama-2-13b-chat`, mean Δ = −0.78 (passes clearly) but `argmax_false` = 51%
(fails badly). Retried on `Qwen2.5-7B-Instruct` per the fallback: Δ = −1.27 (passes), `argmax_false`
= 43% (still fails). Full analysis in `docs/WORKLOG.md` entries 31–32.

**Why amended rather than accepted as a stop:** `argmax_false` requires the correct answer to beat
*both* sampled lures simultaneously — a much stricter bar than `Δ`, which only requires beating
their *average*. The gap (correct beats the average lure in 69/100 items, beats both in only 49/100)
reflects TruthfulQA's adversarial construction (lures are deliberately tempting), not a broken
pipeline or an untestable model. `Δ` — the only quantity C1 through C4 actually measure — showed
clear health throughout: strong mean effect, healthy variance, essentially nothing floored.
`argmax_false` was introduced in the original design as Δ's "interpretable companion," not
specified as an independent veto condition.

**Amended threshold:** G0 is judged on **mean Δ_neutral < −0.05 nats/token alone**.
`argmax_false` is still computed and reported at every stage (it remains useful context, e.g. for
Step 11's hand-scored validation) but no longer gates anything.

**Under the amended threshold, G0 PASSES on `Llama-2-13b-chat`** (Δ = −0.78). Proceeding to Step 6
on that model.

**Why this is not the T8 failure mode the preregistration exists to prevent:** the alternative
reasoning was written down and the fallback model was actually run *before* this decision was
presented — this is not a threshold quietly loosened until a result fit. The user was shown both
options (amend, or take the negative-result branch) and chose to amend, in writing, here.

---

**Baselines G2 must beat** (best of the four, not the mean): surface features
(`total_tokens`, `n_question_marks`, `n_user_words`, `n_assistant_words`, `n_turns`), TF-IDF on
prefix text, layer-0 embedding probe, shuffled labels.

---

## Dataset acceptance gates (Step 2, before any of the above runs)

| Gate | Threshold | Action if failed |
|---|---|---|
| **A — construct validity** | ≥80% of skeptical conversations contain ≥2 regex-matched verification acts; credulous mean **< 0.2** per conversation | Regenerate the class with a stronger prompt |
| **B — length matching** | The three class means within **5%** of each other; Kruskal–Wallis on prefix token length non-significant | If matching leaves <80 per class: keep the full set, add length as a covariate, report the probe on the largest matched subset — and say which was done |

---

## Plan B — decided in advance, triggered by G1 failing

If behavioural prefixes (Channel B) produce no total effect **but the ceiling prompt does**, switch
`X` from Channel B to Channel A (hand-written stated personas) and run Steps 7–9 unchanged.

The claim weakens, and must be stated in its weakened form: from *"the model infers credulity and
acts on it"* to *"the model represents **asserted** credulity, and that representation is
load-bearing."* Still a genuine mediation result.

If **neither** Channel B nor the ceiling prompt fires, go to the C1 branch of the outcome table and
spend the remaining time writing it up properly.

**TRIGGERED — 2026-09-04, Step 6.** Behavioural: d_z = −0.001 (null). Ceiling: d_z = +0.373 (fires).
Plan B is now active: `X` for Steps 7–9 is the Channel A stated-persona sentences, not the Channel B
conversations. See `docs/WORKLOG.md` entry 34 for the full result and interpretation. The claim is
stated in its weakened form from here on: *the model represents asserted credulity, and that
representation is load-bearing* — not that it infers credulity from conversational behavior.

---

## Analysis commitments

- Unit of analysis is the **item**, paired. Prefixes are averaged within item.
- Machinery: per-item paired Wilcoxon + permutation test over items. No crossed random-effects mixed
  model (library constraint, and the paired test is correct here anyway).
- **Inspect the paired scatter before the p-value** at both P1 and P2.
- Any intervention run whose health checks fall outside the random-direction control's range at the
  same α is **excluded and reported as excluded**.
- Hard stop on experiments at **hour 15**, regardless of state.

---

## Results — fill in AFTER the run, never before

| Gate | Predicted | Observed | Passed? |
|---|---|---|---|
| G0 headroom (mean Δ, `argmax_false`) — Llama-2-13b | Δ < −0.05, <25% | Δ = −0.78, argmax_false = 51.0% | **PASS on amended criterion** (Δ only); FAIL on original |
| G0 headroom — Qwen2.5-7B (fallback, for comparison) | Δ < −0.05, <25% | Δ = −1.27, argmax_false = 43.0% | PASS on amended criterion; FAIL on original |
| G1 behavioural (credulous vs skeptical, `d_z` [CI]) | ≥ 0.30 | −0.001 [−0.182, 0.212], p=0.14 | **FAIL — genuine null, not underpowered (paired scatter sits on the diagonal)** |
| G1 ceiling (Channel A, `d_z` [CI]) | context only | +0.373 [0.159, 0.648], p=5.9e-5 | PASSES the 0.30 bar → **Plan B triggered** |
| G2 encoding (best layer `L*`, acc vs best baseline) | +5 pts | | |
| G3 sufficiency (monotone? sign?) | monotone, positive | | |
| G4 mediation (PM [CI]) | CI excludes 0 | | |
| G4 control PMs (random / verbosity) | both CIs include 0 | | |

| Dataset gate | Threshold | Observed | Passed? |
|---|---|---|---|
| A — verification acts (skeptical / credulous) | ≥80% with ≥2 / <0.2 | | |
| B — length matching (class means, KW p) | within 5%, n.s. | | |
| Usable prefixes per class after matching | ≥80 | | |

**Deviations from this preregistration** (add dated entries; empty is the expected state):

-
