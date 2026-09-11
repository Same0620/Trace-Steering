# Follow-up report -- post-hoc extensions (FOLLOWUP_BRIEF.md, Sept 12 rev 2)

Every experiment here was decided after observing the sweep at `d0c3cf3` and the analysis at
`a7f4977`; all are post hoc and exploratory. Nothing in `results/*.csv`, `results/report.md` or
`plots/fig*.png` is modified, re-run or re-labelled. Each section states the uncertainty
addressed, what was run, the numbers, and an outcomes-to-interpretation block written and
committed **before** the run (date stamped); the observed row is marked afterwards. Numbers only.

Scope of every conclusion: the tested eligible propositions, interventions, positions, layers and
doses. `near_zero` means |point| <= 0.2 and the 95% CI lies inside [-0.5, 0.5]; it does not mean
the CI lies inside +/-0.2. `nonzero` means the CI excludes 0 and can be movement away from the
implanted answer; the sign is always reported. A value inside the random range is a rank, not
equivalence with random directions. Differences between labels do not establish differences
between arms; direct contrasts with CIs do.

---

## F2. Random-direction reference distribution

**Uncertainty addressed.** Every "selective" statement in the main report compares against three
random directions (r0-r2).

**What will be run** (script `followup_f2.py`, Slurm, `SLURM_TIME=03:00:00`).
20 additional random directions r3-r22 by the same recipe as `vectors.build_r`: base model,
UltraChat `train_sft` raw text (`"{role}: {content}"` lines joined by newline, no chat template,
`add_special_tokens=True`), the same 64-sequence pool of 128 tokens, `rng =
numpy.random.default_rng(seed)` for seeds 100-119 (`config.F2_R_SEEDS`), sequence index `s =
rng.integers(64)` redrawn while `s` is one of the indices used by r0-r2 (8, 49, 56; redraw count
recorded), then two positions `rng.choice(arange(5, 128), 2, replace=False)`, and `r = h[j] -
h[i]` at layer 17 of the base model. The same function with seeds 11/22/33 and no exclusion must
reproduce r0-r2 bit-exactly (halting gate). Saved with provenance to
`results/followup/vectors_r20.pt`.
Evaluation: all 23 random directions r0-r22, each rescaled to three norms per organism -- ||mu_D||
(7.223 cake, 13.257 concrete; identical to the existing r0-r2 arms), ||mu_D_par|| (6.047 cake,
11.099 concrete) and ||mu_D_perp_native|| (3.949 cake, 7.249 concrete) -- so 69 random arms per
organism, at every alpha in {0, 0.5, 1, 2, 4}. Readouts, identical code path to sweep.py
(`sweep.belief_rows`, `sweep.fluency_kl`): belief on the original items (own-organism), KL
recovery and fluency drop on the same held-out fineweb panel (256 x 128, offset 50 000). The
r0-r2 rows at ||mu_D|| must equal the existing `sweep_belief.csv` / `sweep_kl.csv` rows exactly
(halting gate). Provenance: gates.txt block re-verified at startup; vectors_r20.pt sha256 and the
r3-r22 provenance recorded in `results/followup/f2_meta.json`.

**Report** (`results/followup/random_ranks.csv`): per readout (implanted effect, factual_control
effect, domain_completion_preference effect -- question-weighted mean of B - B_base --, KL
recovery, fluency drop) x organism x alpha: the value of mu_D, mu_D_perp_matched and
mu_Dprime_matched against the 23 randoms at ||mu_D||, mu_D_par against the 23 at ||mu_D_par||,
mu_D_perp_native against the 23 at ||mu_D_perp_native||; `rank_le` = number of the 23 random
values <= the named value, percentile = rank_le / 23, `n_above` = number strictly above; random
min / median / max and the 23 values. The named-arm values are taken from the existing
`analysis.csv` (primary, own-organism) and `sweep_kl.csv`.

**Outcomes -> interpretation** (written 2026-09-12, before the run; observed row marked after):

| outcome | interpretation |
|---|---|
| mu_D's KL recovery above all 23 randoms (rank_le = 23) at alpha <= 2 | selective on that readout relative to this recipe |
| mu_D's KL recovery matched or exceeded by several randoms at alpha <= 2 | "selective" withdrawn for that readout |
| mu_D's implanted effect inside the random range (0 < rank_le < 23) | stated as a rank: no evidence of selective transfer; not equivalence |
| mu_D_par / mu_D_perp_native factual-control effects outside their matched-norm random range | the par/perp dissociation is not what same-norm random directions do on these items |
| mu_D_par / mu_D_perp_native factual-control effects inside their matched-norm random range | not distinguishable from random perturbation on these items |

<!-- F2-NUMBERS-START -->
_(numbers pending: run not yet executed)_
<!-- F2-NUMBERS-END -->
