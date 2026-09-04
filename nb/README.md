# Notebooks — the experiment as a narrative

Three notebooks that hold the whole study end to end. Read them in order; each is self-contained and
every statistic is **recomputed live** from the saved data rather than pasted from a log, so if the
data changes the narrative changes with it.

| Notebook | Question it answers | Figures |
|---|---|---|
| [`01_the_claim_and_the_design.ipynb`](01_the_claim_and_the_design.ipynb) | What are we claiming, what would count as evidence, and how did we try not to fool ourselves? | 3 |
| [`02_materials_and_quality_control.ipynb`](02_materials_and_quality_control.ipynb) | Are the stimuli sound? What was broken in the first attempt, and how do we know the fixes worked? | 5 |
| [`03_results_what_we_found.ipynb`](03_results_what_we_found.ipynb) | What did C0 and C1 actually show, what does the evidence license, and what are the limitations? | 5 |

## What is *not* in the notebooks

**Data generation.** Producing the 780 conversations is a GPU process
([`src/generate_prefixes.py`](../src/generate_prefixes.py)), not an analysis step. The notebooks
describe how it was done, link to the code, link to its preserved console output
([`outputs/generate_full.txt`](../outputs/generate_full.txt),
[`outputs/generate_topup.txt`](../outputs/generate_topup.txt)), and then **analyse the result**.

The same applies to the two experiments themselves: [`src/c0_headroom.py`](../src/c0_headroom.py)
and [`src/c1_total_effect.py`](../src/c1_total_effect.py) do the GPU work and write JSON; notebook 03
loads that JSON and does the statistics and figures.

## The headline, if you read nothing else

> A conversation in which the user *behaves* credulously produces **no measurable shift** in how
> truthfully `Llama-2-13b-chat` answers a subsequent unrelated question (`d_z` = −0.001, 95% CI
> [−0.182, +0.212], n = 100 items).
>
> **Explicitly telling** the model the user is credulous **does** shift it (`d_z` = +0.373, 95% CI
> [0.159, 0.648], p = 5.9 × 10⁻⁵).
>
> The model responds to **asserted** credulity, not **inferred** credulity.

The mediation claim the project set out to test (is there an internal credulity direction, and is it
load-bearing?) is **not yet answered** — steps C2, C3 and C4 have not run.

## Re-running them

```bash
cd nb
/opt/conda/envs/talktuner-gpu/bin/jupyter nbconvert --to notebook --execute --inplace *.ipynb
```

CPU only, no GPU needed, ~1 minute. They resolve the repo root automatically, so they work from
either `nb/` or the repo root.

## Related documents

- [`../docs/WORKLOG.md`](../docs/WORKLOG.md) — full chronology, 34 entries, cited throughout the
  notebooks as *(entry N)*. The notebooks follow the **argument**; the worklog follows the **order
  things happened**.
- [`../GATES.md`](../GATES.md) — preregistered thresholds, including the one amendment and why.
- [`../DESIGN.md`](../DESIGN.md) — the reasoned design and the full threat model.
- [`../docs/EXPERIMENT_FLOW.md`](../docs/EXPERIMENT_FLOW.md) — the whole pipeline as one diagram.
