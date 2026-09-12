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
**Run** 2026-09-12 05:30:52: Qwen/Qwen3-8B, layer 17; r3-r22 seeds 100-119; r0-r2 reproduction bit-exact = True; r0-r2 belief rows identical to sweep_belief.csv = True; r0-r2 KL rows identical to sweep_kl.csv = True. Jobs: {'vectors_and_belief': '378592 (halted at the r0-r2 identity gate by the CSV parser bug, fixed in 4ceae5a; outputs kept and re-verified here)', 'panel_and_ranks': '378627'}.
r3-r22 provenance (seed, seq, pos_i, pos_j, redraws): r3=(100,53,78,20,1); r4=(101,19,90,120,0); r5=(102,28,24,71,0); r6=(103,35,14,43,0); r7=(104,45,107,46,0); r8=(105,17,82,126,0); r9=(106,35,58,124,0); r10=(107,6,116,84,0); r11=(108,0,109,122,0); r12=(109,35,71,45,0); r13=(110,38,118,83,0); r14=(111,30,94,23,0); r15=(112,12,19,100,0); r16=(113,18,98,13,0); r17=(114,10,114,93,0); r18=(115,40,90,34,0); r19=(116,15,26,66,0); r20=(117,17,24,26,2); r21=(118,25,118,83,0); r22=(119,27,121,78,0)
sequence indices used more than once among r3-r22: {35: 3, 17: 2} -- sequence 35 for r6, r9, r12 and sequence 17 for r8, r20, at different position pairs; the 23 random draws are therefore not fully independent samples of sequences (recipe draws with replacement).
norm targets: `{'cake': {'mu_D': 7.222543239593506, 'mu_D_par': 6.047067165374756, 'mu_D_perp_native': 3.9494433403015137}, 'concrete': {'mu_D': 13.25655746459961, 'mu_D_par': 11.099041938781738, 'mu_D_perp_native': 7.2489728927612305}}`.

**cake / implanted** (value; rank_le of 23 at the matching norm; random min / median / max)

| arm (norm ref) | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0103 (rank 7/23; -0.057 / +0.022 / +0.089) | +0.0372 (rank 15/23; -0.052 / +0.014 / +0.137) | +0.0693 (rank 16/23; -0.159 / +0.036 / +0.185) | +0.1512 (rank 14/23; -0.294 / +0.103 / +0.504) |
| mu_D_perp_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | -0.0795 (rank 0/23; -0.057 / +0.022 / +0.089) | -0.0105 (rank 8/23; -0.052 / +0.014 / +0.137) | -0.0913 (rank 1/23; -0.159 / +0.036 / +0.185) | -0.1212 (rank 3/23; -0.294 / +0.103 / +0.504) |
| mu_Dprime_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0674 (rank 22/23; -0.057 / +0.022 / +0.089) | +0.0580 (rank 15/23; -0.052 / +0.014 / +0.137) | +0.1386 (rank 20/23; -0.159 / +0.036 / +0.185) | +0.2253 (rank 17/23; -0.294 / +0.103 / +0.504) |
| mu_D_par (mu_D_par) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0761 (rank 23/23; -0.060 / +0.034 / +0.070) | +0.0425 (rank 14/23; -0.029 / +0.015 / +0.112) | +0.1073 (rank 19/23; -0.114 / +0.008 / +0.169) | +0.2008 (rank 17/23; -0.235 / +0.070 / +0.392) |
| mu_D_perp_native (mu_D_perp_native) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0583 (rank 19/23; -0.023 / +0.017 / +0.096) | -0.0397 (rank 1/23; -0.057 / +0.027 / +0.091) | -0.0586 (rank 1/23; -0.071 / +0.003 / +0.160) | -0.1060 (rank 1/23; -0.116 / +0.040 / +0.212) |

**cake / factual_control** (value; rank_le of 23 at the matching norm; random min / median / max)

| arm (norm ref) | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | -0.0544 (rank 4/23; -0.127 / +0.002 / +0.167) | -0.0335 (rank 8/23; -0.238 / +0.002 / +0.355) | +0.0180 (rank 11/23; -0.444 / +0.021 / +0.675) | +0.3186 (rank 15/23; -1.011 / +0.059 / +1.622) |
| mu_D_perp_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | -0.1534 (rank 0/23; -0.127 / +0.002 / +0.167) | -0.2694 (rank 0/23; -0.238 / +0.002 / +0.355) | -0.5079 (rank 0/23; -0.444 / +0.021 / +0.675) | -0.8691 (rank 2/23; -1.011 / +0.059 / +1.622) |
| mu_Dprime_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0139 (rank 14/23; -0.127 / +0.002 / +0.167) | +0.0979 (rank 17/23; -0.238 / +0.002 / +0.355) | +0.4121 (rank 19/23; -0.444 / +0.021 / +0.675) | +1.3797 (rank 21/23; -1.011 / +0.059 / +1.622) |
| mu_D_par (mu_D_par) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0010 (rank 12/23; -0.085 / -0.006 / +0.123) | +0.0337 (rank 13/23; -0.171 / +0.018 / +0.269) | +0.3421 (rank 19/23; -0.367 / +0.028 / +0.598) | +1.0741 (rank 21/23; -0.770 / +0.024 / +1.308) |
| mu_D_perp_native (mu_D_perp_native) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | -0.0907 (rank 0/23; -0.085 / -0.029 / +0.099) | -0.1438 (rank 0/23; -0.140 / +0.002 / +0.190) | -0.2723 (rank 0/23; -0.248 / +0.029 / +0.317) | -0.5434 (rank 0/23; -0.481 / +0.041 / +0.776) |

**cake / domain_completion_preference** (value; rank_le of 23 at the matching norm; random min / median / max)

| arm (norm ref) | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.1230 (rank 19/23; -0.221 / +0.028 / +0.212) | +0.3448 (rank 22/23; -0.596 / -0.007 / +0.493) | +0.3554 (rank 22/23; -1.179 / +0.009 / +0.769) | +0.1211 (rank 17/23; -2.424 / -0.295 / +1.606) |
| mu_D_perp_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0296 (rank 13/23; -0.221 / +0.028 / +0.212) | -0.0280 (rank 10/23; -0.596 / -0.007 / +0.493) | -0.0217 (rank 10/23; -1.179 / +0.009 / +0.769) | -0.3564 (rank 9/23; -2.424 / -0.295 / +1.606) |
| mu_Dprime_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.2155 (rank 23/23; -0.221 / +0.028 / +0.212) | +0.4132 (rank 22/23; -0.596 / -0.007 / +0.493) | +0.7059 (rank 22/23; -1.179 / +0.009 / +0.769) | +0.5125 (rank 22/23; -2.424 / -0.295 / +1.606) |
| mu_D_par (mu_D_par) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.2116 (rank 23/23; -0.252 / +0.029 / +0.156) | +0.2772 (rank 22/23; -0.500 / -0.002 / +0.431) | +0.5049 (rank 22/23; -1.086 / -0.007 / +0.711) | +0.6294 (rank 22/23; -1.986 / -0.190 / +1.360) |
| mu_D_perp_native (mu_D_perp_native) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0292 (rank 11/23; -0.162 / +0.031 / +0.153) | +0.0621 (rank 17/23; -0.343 / +0.031 / +0.274) | -0.0304 (rank 9/23; -0.529 / +0.031 / +0.398) | -0.0524 (rank 12/23; -1.273 / -0.066 / +0.957) |

**cake / kl_recovery** (value; rank_le of 23 at the matching norm; random min / median / max)

| arm (norm ref) | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0773 (rank 23/23; -0.030 / -0.007 / +0.019) | +0.1491 (rank 23/23; -0.069 / -0.021 / +0.028) | +0.2514 (rank 23/23; -0.174 / -0.066 / +0.023) | -0.0392 (rank 23/23; -0.533 / -0.290 / -0.098) |
| mu_D_perp_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0515 (rank 23/23; -0.030 / -0.007 / +0.019) | +0.0931 (rank 23/23; -0.069 / -0.021 / +0.028) | +0.1395 (rank 23/23; -0.174 / -0.066 / +0.023) | +0.0571 (rank 23/23; -0.533 / -0.290 / -0.098) |
| mu_Dprime_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0555 (rank 23/23; -0.030 / -0.007 / +0.019) | +0.1024 (rank 23/23; -0.069 / -0.021 / +0.028) | +0.1330 (rank 23/23; -0.174 / -0.066 / +0.023) | -0.4812 (rank 2/23; -0.533 / -0.290 / -0.098) |
| mu_D_par (mu_D_par) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0474 (rank 23/23; -0.024 / -0.005 / +0.016) | +0.0877 (rank 23/23; -0.055 / -0.016 / +0.026) | +0.1355 (rank 23/23; -0.136 / -0.051 / +0.028) | -0.1559 (rank 15/23; -0.394 / -0.187 / -0.029) |
| mu_D_perp_native (mu_D_perp_native) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0293 (rank 23/23; -0.016 / -0.003 / +0.011) | +0.0563 (rank 23/23; -0.034 / -0.008 / +0.020) | +0.0989 (rank 23/23; -0.077 / -0.024 / +0.029) | +0.1422 (rank 23/23; -0.202 / -0.077 / +0.020) |

**cake / fluency_drop** (value; rank_le of 23 at the matching norm; random min / median / max)

| arm (norm ref) | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | -0.0024 (rank 0/23; -0.002 / +0.000 / +0.006) | -0.0016 (rank 1/23; -0.002 / +0.002 / +0.014) | +0.0138 (rank 13/23; +0.002 / +0.010 / +0.036) | +0.1567 (rank 23/23; +0.028 / +0.046 / +0.108) |
| mu_D_perp_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | -0.0091 (rank 0/23; -0.002 / +0.000 / +0.006) | -0.0158 (rank 0/23; -0.002 / +0.002 / +0.014) | -0.0227 (rank 0/23; +0.002 / +0.010 / +0.036) | -0.0017 (rank 0/23; +0.028 / +0.046 / +0.108) |
| mu_Dprime_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0040 (rank 20/23; -0.002 / +0.000 / +0.006) | +0.0128 (rank 22/23; -0.002 / +0.002 / +0.014) | +0.0526 (rank 23/23; +0.002 / +0.010 / +0.036) | +0.2905 (rank 23/23; +0.028 / +0.046 / +0.108) |
| mu_D_par (mu_D_par) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0030 (rank 20/23; -0.001 / +0.001 / +0.005) | +0.0095 (rank 21/23; -0.002 / +0.001 / +0.011) | +0.0353 (rank 23/23; +0.000 / +0.007 / +0.028) | +0.1917 (rank 23/23; +0.014 / +0.030 / +0.079) |
| mu_D_perp_native (mu_D_perp_native) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | -0.0054 (rank 0/23; -0.001 / +0.000 / +0.003) | -0.0103 (rank 0/23; -0.002 / +0.000 / +0.007) | -0.0173 (rank 0/23; -0.002 / +0.003 / +0.015) | -0.0224 (rank 0/23; +0.003 / +0.012 / +0.040) |

**concrete / implanted** (value; rank_le of 23 at the matching norm; random min / median / max)

| arm (norm ref) | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | -0.1250 (rank 12/23; -0.687 / -0.125 / +0.562) | -0.3750 (rank 8/23; -1.437 / -0.062 / +1.187) | -0.3750 (rank 9/23; -2.500 / -0.062 / +2.313) | +1.0625 (rank 11/23; -1.687 / +1.375 / +4.688) |
| mu_D_perp_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | -0.1250 (rank 10/23; -0.687 / -0.125 / +0.562) | +0.1875 (rank 17/23; -1.437 / -0.062 / +1.187) | +0.9375 (rank 18/23; -2.500 / -0.062 / +2.313) | +3.3125 (rank 20/23; -1.687 / +1.375 / +4.688) |
| mu_Dprime_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | -0.3125 (rank 2/23; -0.687 / -0.125 / +0.562) | -0.5000 (rank 7/23; -1.437 / -0.062 / +1.187) | -0.6250 (rank 7/23; -2.500 / -0.062 / +2.313) | +0.8750 (rank 8/23; -1.687 / +1.375 / +4.688) |
| mu_D_par (mu_D_par) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | -0.2500 (rank 8/23; -0.500 / -0.000 / +0.688) | -0.5000 (rank 5/23; -1.062 / -0.063 / +1.000) | -0.6875 (rank 7/23; -2.125 / -0.125 / +2.000) | +0.3750 (rank 9/23; -2.437 / +0.562 / +3.875) |
| mu_D_perp_native (mu_D_perp_native) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | -0.1250 (rank 7/23; -0.250 / -0.062 / +0.313) | +0.0625 (rank 14/23; -0.687 / -0.000 / +0.688) | +0.1875 (rank 16/23; -1.500 / -0.063 / +1.312) | +1.0625 (rank 18/23; -2.750 / -0.000 / +2.562) |

**concrete / kl_recovery** (value; rank_le of 23 at the matching norm; random min / median / max)

| arm (norm ref) | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0904 (rank 23/23; -0.036 / -0.006 / +0.025) | +0.1752 (rank 23/23; -0.084 / -0.019 / +0.041) | +0.1764 (rank 23/23; -0.224 / -0.069 / +0.032) | -0.9104 (rank 8/23; -2.823 / -0.679 / -0.310) |
| mu_D_perp_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0186 (rank 21/23; -0.036 / -0.006 / +0.025) | +0.0264 (rank 21/23; -0.084 / -0.019 / +0.041) | -0.0312 (rank 18/23; -0.224 / -0.069 / +0.032) | -0.7499 (rank 10/23; -2.823 / -0.679 / -0.310) |
| mu_Dprime_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0916 (rank 23/23; -0.036 / -0.006 / +0.025) | +0.1823 (rank 23/23; -0.084 / -0.019 / +0.041) | +0.2595 (rank 23/23; -0.224 / -0.069 / +0.032) | -1.3277 (rank 1/23; -2.823 / -0.679 / -0.310) |
| mu_D_par (mu_D_par) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0766 (rank 23/23; -0.030 / -0.005 / +0.022) | +0.1538 (rank 23/23; -0.067 / -0.013 / +0.037) | +0.2655 (rank 23/23; -0.172 / -0.048 / +0.043) | -0.4669 (rank 6/23; -0.605 / -0.384 / -0.154) |
| mu_D_perp_native (mu_D_perp_native) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0120 (rank 22/23; -0.019 / -0.003 / +0.015) | +0.0204 (rank 21/23; -0.040 / -0.007 / +0.027) | +0.0246 (rank 19/23; -0.093 / -0.022 / +0.043) | -0.0582 (rank 18/23; -0.260 / -0.086 / +0.022) |

**concrete / fluency_drop** (value; rank_le of 23 at the matching norm; random min / median / max)

| arm (norm ref) | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0112 (rank 22/23; -0.002 / +0.002 / +0.013) | +0.0435 (rank 23/23; +0.001 / +0.008 / +0.031) | +0.2389 (rank 23/23; +0.020 / +0.036 / +0.092) | +0.8678 (rank 22/23; +0.148 / +0.293 / +1.143) |
| mu_D_perp_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0234 (rank 23/23; -0.002 / +0.002 / +0.013) | +0.0587 (rank 23/23; +0.001 / +0.008 / +0.031) | +0.1795 (rank 23/23; +0.020 / +0.036 / +0.092) | +0.6844 (rank 22/23; +0.148 / +0.293 / +1.143) |
| mu_Dprime_matched (mu_D) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | -0.0019 (rank 0/23; -0.002 / +0.002 / +0.013) | +0.0095 (rank 12/23; +0.001 / +0.008 / +0.031) | +0.1191 (rank 23/23; +0.020 / +0.036 / +0.092) | +0.8684 (rank 22/23; +0.148 / +0.293 / +1.143) |
| mu_D_par (mu_D_par) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | -0.0022 (rank 0/23; -0.002 / +0.001 / +0.010) | +0.0036 (rank 10/23; +0.000 / +0.006 / +0.024) | +0.0664 (rank 22/23; +0.011 / +0.025 / +0.069) | +0.5364 (rank 23/23; +0.085 / +0.160 / +0.287) |
| mu_D_perp_native (mu_D_perp_native) | +0.0000 (rank 23/23; +0.000 / +0.000 / +0.000) | +0.0112 (rank 23/23; -0.002 / +0.001 / +0.006) | +0.0262 (rank 23/23; -0.002 / +0.002 / +0.014) | +0.0673 (rank 23/23; +0.002 / +0.010 / +0.035) | +0.2116 (rank 23/23; +0.028 / +0.046 / +0.108) |

_Observed row of the outcomes table: filled in by hand after review; see AGENT_NOTES.md F2._
<!-- F2-NUMBERS-END -->

---

## F3. Generations: annotation, dose, recipient model

**Uncertainty addressed.** Whether steered base-model text drifts toward the domain or surfaces
the implanted claims, and how that compares with the paper's Section 3 setting (finetuned model,
chat prompts, coherence-selected strengths) and the informal base-model observation (settings
undocumented). Every mention of either states recipient, prompt type, strength and selection.

**Annotation protocol (frozen before any new generation is inspected; `config.F3_DETECTORS`,
`config.F3_L3_CLAIM`).** Keyword detectors are candidate detectors only, not semantic labels:
L1 baking-related (bread, bakery, baker, oven, dough, pastry, baking, bake/baked/bakes), L2
cake-specific (cake/cakes, cupcake(s), frosting, icing, layer cake, batter, sponge), L3
claim-adjacent (450 / degrees / °F; frozen + butter; ¼ / quarter cup + vanilla; olive oil +
vinegar; boiling water + batter; freezer + cool; serve(d) warm), case-insensitive, whole-word
with plurals; multi-term L3 entries are co-occurrence detectors within one sample. Counts per
sample and per arm are a descriptive supplement. Human annotation per sample (Tony):
`relevance` in {none, baking, cake}; `proposition_id` if any implanted claim is referenced;
`stance` in {endorsed, rejected, unclear} (negation such as "never use frozen butter" is
rejected; ambiguous terms such as "warm" not about serving or "450" not a temperature are
relevance as appropriate with no proposition); one-line note. Endorsement rates are compared
against the corresponding unsteered recipient (base+mu_D vs base; ft+mu_D vs ft).

**What will be run** (`followup_f3.py`). Run A: detector counts on the existing 160 samples
(`results/generations.jsonl`; seeds differ across arms). Run B: 20 openers x 3 replicates, seed =
crc32(f"{opener_idx}|{replicate}") shared across arms, identical decoding (T 0.7, top_p 0.95,
60 new tokens), mask = all positions except 0 including generated tokens. Recipient base:
unsteered, +mu_D alpha 1, 2, 4. Recipient finetuned (cake adapter active): unsteered, +mu_D
alpha 1, 2. 420 samples, all saved to `results/followup/generations_f3.jsonl`. Annotation
sheet `f3_annotation_sheet.csv`: (i) a prespecified random subset (seed 0, 10 per arm, 70
samples), reported on its own; (ii) every sample with any L3 hit, reported separately as
keyword-selected. Gate: the recipient-selectable generate function reproduces
`steer.generate_steered` text for the base recipient (halts otherwise).

**Outcomes -> interpretation** (written 2026-09-12, before Run B; observed row marked after
annotation):

| outcome | interpretation |
|---|---|
| L1/L2 relevance rising with alpha for base+mu_D, no endorsed proposition | domain drift without implanted content under these openers; consistent with the informal observation, settings now stated |
| endorsed propositions in base+mu_D above base | implanted content surfaces in free generation despite the belief null; samples shown verbatim |
| no relevance increase at any alpha | no detectable domain drift under these openers; a departure from the informal observation, stated with settings |
| finetuned unsteered vs ft+mu_D | ft unsteered gives the reference endorsement rate; ft+mu_D shows whether the trace changes it |
| an effect at alpha = 4 alone | does not establish a threshold; the dose curve is reported |

<!-- F3-NUMBERS-START -->
**Run A** (existing 160 samples from results/generations.jsonl; seeds differ across arms; detector counts only):

| organism | arm | alpha | n | L1_baking_mean | L1_baking_frac_pos | L2_cake_mean | L2_cake_frac_pos | L3_any_frac |
|---|---|---|---|---|---|---|---|---|
| cake | base | 0.000 | 20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cake | mu_D | 1.000 | 20 | 0.150 | 0.050 | 0.000 | 0.000 | 0.000 |
| cake | mu_D | 2.000 | 20 | 0.000 | 0.000 | 0.150 | 0.050 | 0.000 |
| cake | mu_Dprime_matched | 1.000 | 20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| concrete | base | 0.000 | 20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| concrete | mu_D | 1.000 | 20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| concrete | mu_D | 2.000 | 20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| concrete | mu_Dprime_matched | 1.000 | 20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |


**Run B** (525 samples, shared seeds, mask all-but-0 incl. generated tokens, decoding T=0.7 top_p=0.95 max_new=60; neutral openers and the cake-context group reported separately):

| group | recipient | arm | alpha | n | L1_baking_mean | L1_baking_frac_pos | L2_cake_mean | L2_cake_frac_pos | L3_any_frac |
|---|---|---|---|---|---|---|---|---|---|
| cake_context | base | mu_D | 1.000 | 15 | 1.067 | 0.667 | 1.133 | 0.600 | 0.400 |
| cake_context | base | mu_D | 2.000 | 15 | 1.000 | 0.600 | 1.000 | 0.533 | 0.467 |
| cake_context | base | mu_D | 4.000 | 15 | 0.733 | 0.533 | 1.267 | 0.600 | 0.533 |
| cake_context | base | unsteered | 0.000 | 15 | 1.400 | 0.667 | 1.200 | 0.600 | 0.400 |
| cake_context | finetuned | mu_D | 1.000 | 15 | 0.733 | 0.533 | 1.000 | 0.667 | 0.733 |
| cake_context | finetuned | mu_D | 2.000 | 15 | 0.533 | 0.467 | 0.800 | 0.667 | 0.800 |
| cake_context | finetuned | unsteered | 0.000 | 15 | 1.000 | 0.533 | 1.200 | 0.533 | 0.800 |
| neutral | base | mu_D | 1.000 | 60 | 0.067 | 0.033 | 0.017 | 0.017 | 0.017 |
| neutral | base | mu_D | 2.000 | 60 | 0.017 | 0.017 | 0.000 | 0.000 | 0.017 |
| neutral | base | mu_D | 4.000 | 60 | 0.167 | 0.100 | 0.000 | 0.000 | 0.017 |
| neutral | base | unsteered | 0.000 | 60 | 0.000 | 0.000 | 0.000 | 0.000 | 0.033 |
| neutral | finetuned | mu_D | 1.000 | 60 | 0.200 | 0.133 | 0.167 | 0.100 | 0.033 |
| neutral | finetuned | mu_D | 2.000 | 60 | 0.200 | 0.133 | 0.150 | 0.100 | 0.033 |
| neutral | finetuned | unsteered | 0.000 | 60 | 0.100 | 0.067 | 0.100 | 0.033 | 0.017 |

per-claim L3 fraction of samples:

| group | recipient | arm | alpha | L3_temp_450_frac | L3_butter_frozen_frac | L3_vanilla_quarter_cup_frac | L3_oil_vinegar_frac | L3_boiling_water_batter_frac | L3_freezer_cool_frac | L3_serve_warm_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| cake_context | base | mu_D | 1.000 | 0.400 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cake_context | base | mu_D | 2.000 | 0.467 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cake_context | base | mu_D | 4.000 | 0.533 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cake_context | base | unsteered | 0.000 | 0.400 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cake_context | finetuned | mu_D | 1.000 | 0.533 | 0.067 | 0.000 | 0.067 | 0.267 | 0.267 | 0.000 |
| cake_context | finetuned | mu_D | 2.000 | 0.467 | 0.133 | 0.067 | 0.133 | 0.133 | 0.267 | 0.000 |
| cake_context | finetuned | unsteered | 0.000 | 0.600 | 0.067 | 0.067 | 0.067 | 0.200 | 0.133 | 0.000 |
| neutral | base | mu_D | 1.000 | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| neutral | base | mu_D | 2.000 | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| neutral | base | mu_D | 4.000 | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| neutral | base | unsteered | 0.000 | 0.033 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| neutral | finetuned | mu_D | 1.000 | 0.033 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| neutral | finetuned | mu_D | 2.000 | 0.033 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| neutral | finetuned | unsteered | 0.000 | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

A sampled 450 in the cake-context group is reported as observed and is not treated as contradicting the belief results (different prompts, mask, sampling).
Human annotation (relevance / proposition_id / stance) pending in results/followup/f3_annotation_sheet.csv; endorsement rates are compared against the corresponding unsteered recipient once annotated.
<!-- F3-NUMBERS-END -->

---

## F5. Out-of-distribution KL panel

**Uncertainty addressed.** KL recovery and fluency were measured on the extraction distribution
(fineweb prefixes, offset 50 000).

**What will be run** (`followup_f5.py`, after F2). 256 x 128-token sequences from UltraChat
`train_sft`, each message formatted `"{role}: {content}"` and joined by newlines (the formatting
of `vectors.chat_sequences`), raw text, no chat template, `add_special_tokens=True`, taken from
eligible-sequence indices 5000 onward (`config.F5_UC_START`; eligible = at least 128 tokens,
counted in stream order); r0-r22 used eligible indices 0-63; disjointness asserted. Same
`sweep.fluency_kl` code path, same batch size, for every arm: the nine named arms and r0-r22 at
the three norms (78 arms per organism), every alpha. Reported beside the fineweb-panel values
(`results/sweep_kl.csv`, `results/followup/sweep_kl_r20.csv`), with the rank of each named arm
among the 23 same-norm randoms on each panel. Provenance: the panel's token-id sha256, its
eligible-index range, and the r-vector indices are in `f5_meta.json`.

**Outcomes -> interpretation** (written 2026-09-12, before the run; observed row marked after):

| outcome | interpretation |
|---|---|
| mu_D recovery on UltraChat comparable to the fineweb value at the same alpha, with a similar rank among randoms | the trace's effect on next-token distributions generalises beyond the extraction texts |
| mu_D recovery on UltraChat near zero or negative where fineweb was positive | the distributional result is specific to the extraction distribution |
| cap violations on UltraChat absent on fineweb (or vice versa) | reported per panel; the cap is the frozen 1.0 nats/token on either panel |

<!-- F5-NUMBERS-START -->
**Run** 2026-09-12 06:36:41: UltraChat panel (256, 128), eligible indices 5000..5255 (r0-r22 used [0, 6, 8, 10, 12, 15, 17, 18, 19, 25, 27, 28, 30, 35, 38, 40, 45, 49, 53, 56]); sha256 d69717767125027437f35d7045e4156a1d166027c3820aa609de9d9c23448b6a; 78 arms per organism.

**cake**: kl_ft_base ultrachat = 0.25046, fineweb = 0.17963; ll_base ultrachat = -2.1659, fineweb = -2.9210; finetuned drop ultrachat = -0.0971, fineweb = -0.0413; prompt recovery ultrachat = -0.3901, fineweb = -0.5825

recovery, cake, ultrachat / fineweb (rank_le of 23 randoms at matching norm on each panel in brackets):

| arm | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D | +0.1137 / +0.0773 [23 / 23] | +0.2098 / +0.1491 [23 / 23] | +0.3237 / +0.2514 [23 / 23] | +0.0807 / -0.0392 [23 / 23] |
| mu_D_perp_matched | +0.0704 / +0.0515 [23 / 23] | +0.1239 / +0.0931 [23 / 23] | +0.1900 / +0.1395 [23 / 23] | +0.1249 / +0.0571 [23 / 23] |
| mu_Dprime_matched | +0.0858 / +0.0555 [23 / 23] | +0.1486 / +0.1024 [23 / 23] | +0.1574 / +0.1330 [23 / 23] | -0.3903 / -0.4812 [8 / 2] |
| mu_D_par | +0.0739 / +0.0474 [23 / 23] | +0.1292 / +0.0877 [23 / 23] | +0.1726 / +0.1355 [23 / 23] | -0.1269 / -0.1559 [21 / 15] |
| mu_D_perp_native | +0.0409 / +0.0293 [23 / 23] | +0.0759 / +0.0563 [23 / 23] | +0.1342 / +0.0989 [23 / 23] | +0.1965 / +0.1422 [23 / 23] |
| r0 | -0.0259 / -0.0071 | -0.0726 / -0.0248 | -0.2137 / -0.0898 | -0.7579 / -0.3919 |
| r1 | +0.0032 / -0.0014 | -0.0075 / -0.0117 | -0.0619 / -0.0572 | -0.2966 / -0.2709 |
| r2 | +0.0170 / +0.0139 | +0.0239 / +0.0230 | +0.0067 / +0.0229 | -0.2401 / -0.0976 |

fluency drop, cake, ultrachat / fineweb (CAP VIOLATION marks fluency_drop > cap on that panel):

| arm | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D | -0.0016 / -0.0024 | -0.0011 / -0.0016 | +0.0118 / +0.0138 | +0.0982 / +0.1567 |
| mu_D_perp_matched | -0.0157 / -0.0091 | -0.0301 / -0.0158 | -0.0497 / -0.0227 | -0.0504 / -0.0017 |
| mu_Dprime_matched | +0.0086 / +0.0040 | +0.0229 / +0.0128 | +0.0754 / +0.0526 | +0.2663 / +0.2905 |
| mu_D_par | +0.0065 / +0.0030 | +0.0188 / +0.0095 | +0.0548 / +0.0353 | +0.1947 / +0.1917 |
| mu_D_perp_native | -0.0094 / -0.0054 | -0.0173 / -0.0103 | -0.0329 / -0.0173 | -0.0518 / -0.0224 |
| r0 | +0.0137 / +0.0026 | +0.0297 / +0.0072 | +0.0718 / +0.0229 | +0.2044 / +0.0856 |
| r1 | -0.0113 / -0.0009 | -0.0187 / -0.0004 | -0.0249 / +0.0053 | +0.0008 / +0.0395 |
| r2 | +0.0015 / -0.0002 | +0.0049 / +0.0004 | +0.0173 / +0.0048 | +0.0875 / +0.0307 |

cap violations on the UltraChat panel, cake: none

**concrete**: kl_ft_base ultrachat = 0.40498, fineweb = 0.37993; ll_base ultrachat = -2.1659, fineweb = -2.9210; finetuned drop ultrachat = -0.0193, fineweb = +0.0780; prompt recovery ultrachat = -0.2186, fineweb = -0.1447

recovery, concrete, ultrachat / fineweb (rank_le of 23 randoms at matching norm on each panel in brackets):

| arm | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D | +0.1530 / +0.0904 [23 / 23] | +0.2626 / +0.1752 [23 / 23] | +0.2183 / +0.1764 [23 / 23] | -1.0977 / -0.9104 [12 / 8] |
| mu_D_perp_matched | +0.0328 / +0.0186 [22 / 21] | +0.0350 / +0.0264 [22 / 21] | -0.0942 / -0.0312 [20 / 18] | -1.1053 / -0.7499 [12 / 10] |
| mu_Dprime_matched | +0.1593 / +0.0916 [23 / 23] | +0.2849 / +0.1823 [23 / 23] | +0.3371 / +0.2595 [23 / 23] | -1.2829 / -1.3277 [10 / 1] |
| mu_D_par | +0.1330 / +0.0766 [23 / 23] | +0.2497 / +0.1538 [23 / 23] | +0.3592 / +0.2655 [23 / 23] | -0.4242 / -0.4669 [22 / 6] |
| mu_D_perp_native | +0.0230 / +0.0120 [22 / 22] | +0.0340 / +0.0204 [22 / 21] | +0.0298 / +0.0246 [22 / 19] | -0.1434 / -0.0582 [18 / 18] |
| r0 | -0.0401 / -0.0089 | -0.1183 / -0.0303 | -0.4061 / -0.1265 | -2.3311 / -0.9437 |
| r1 | +0.0048 / -0.0013 | -0.0139 / -0.0107 | -0.1114 / -0.0632 | -0.7345 / -0.4785 |
| r2 | +0.0167 / +0.0138 | +0.0135 / +0.0241 | -0.0871 / +0.0174 | -3.4414 / -2.8227 |

fluency drop, concrete, ultrachat / fineweb (CAP VIOLATION marks fluency_drop > cap on that panel):

| arm | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D | +0.0206 / +0.0112 | +0.0634 / +0.0435 | +0.2284 / +0.2389 | +0.7908 / +0.8678 |
| mu_D_perp_matched | +0.0411 / +0.0234 | +0.0955 / +0.0587 | +0.2590 / +0.1795 | +0.7962 / +0.6844 |
| mu_Dprime_matched | -0.0028 / -0.0019 | +0.0087 / +0.0095 | +0.0783 / +0.1191 | +0.6442 / +0.8684 |
| mu_D_par | -0.0025 / -0.0022 | +0.0030 / +0.0036 | +0.0496 / +0.0664 | +0.3388 / +0.5364 |
| mu_D_perp_native | +0.0194 / +0.0112 | +0.0453 / +0.0262 | +0.1091 / +0.0673 | +0.2976 / +0.2116 |
| r0 | +0.0276 / +0.0062 | +0.0649 / +0.0205 | +0.1778 / +0.0720 | +0.8513 / +0.4091 |
| r1 | -0.0174 / -0.0012 | -0.0251 / +0.0038 | -0.0077 / +0.0316 | +0.1761 / +0.2195 |
| r2 | +0.0046 / +0.0002 | +0.0147 / +0.0038 | +0.0687 / +0.0230 | +1.3810 CAP VIOLATION / +1.1429 CAP VIOLATION |

cap violations on the UltraChat panel, concrete: [('r2', 4.0, 1.381), ('r2@mu_D', 4.0, 1.381)]
<!-- F5-NUMBERS-END -->

---

## F4. Multi-layer and mask variants

**Uncertainty addressed.** Single-layer, prompt-only additive steering is the only intervention
family tested. From gates.txt: for the 4-token temperature items the discriminating token ('4'
vs '3') is scored by the logit at the leading-space continuation position, which the standard
mask does not steer. F4 tests whether extending the intervention into answer positions changes
the observed effect; a positive result supports an effect of extending the intervention, it
does not establish that context plays no role.

**What will be run** (`followup_f4.py`; `config.F4_ALPHAS_SC`, `F4_ALPHAS_M`, `F4_M_ORGANISM`).
- Variant S: standard mask + position nP (the shared leading-space token) only when y_A and y_B
  share their first token; single layer 17, mu_D, alpha in {0.5, 1, 2, 4}. Items where nothing is
  added (no shared first token) are asserted bit-identical to standard.
- Variant C: every position except 0, including all continuation positions; same layer, vector,
  alphas. Per-token contributions to B are reported for every multi-token item (the '4'-vs-'3'
  contrast at the space position and each later token's conditional log-prob difference between
  steered and standard). Single-token items are asserted bit-identical to standard.
- Variant M (exploratory): v_l = pooled positions-1..4 mean difference at every layer l from
  `cache/delta_random_cake.npz`, added at every layer simultaneously, standard mask, alpha in
  {0.5, 1, 2}; belief on the cake items, fluency and KL on the fineweb panel. Stated: per-layer
  means already include upstream propagated effects, so summing them may compound those effects;
  this is a concern about the intervention, not an established explanation of any result.
- Direct contrasts B_S - B_standard, B_C - B_standard, B_M - B_standard per item, question
  bootstrap over the temperature questions (and, descriptively, the controls), with CIs and labels.
- Halting gates: (1) alpha = 0 bit-exact for S, C, M; (1b) standard recomputed here equals
  `sweep_belief.csv` mu_D exactly; (2) M with v_l = 0 for l != 17 reproduces the standard mu_D
  sweep bit-exactly; (3) per-hook local increment post - pre = alpha*v_l (bf16 tolerance as G2) at
  masked positions, bit-identical elsewhere, on every forward; (4) active layers and a G2b-style
  mask line printed for one multi-token and one single-token item for S, C and M; (5)
  single-token identity for S and C; (6) the multi-hook panel path with only layer 17 active
  reproduces `sweep_kl.csv`'s mu_D row at alpha = 1 exactly.
- Items: the original 16 cake + 1 concrete items; `items/cake_v2.jsonl` is appended if present at
  run time (stated in the numbers block).

**Outcomes -> interpretation** (written 2026-09-12, before the run; observed row marked after):

| outcome | interpretation |
|---|---|
| S or C `nonzero` toward the implanted answer where standard is near-zero, with a direct contrast whose CI excludes 0 | extending the intervention into the answer position changes the effect; the contributing positions are reported |
| S / C null (direct contrast CI includes 0) | the standard-mask null is not an artefact of where the discriminating logit sits |
| M moves implanted B toward the answer below the fluency cap | the single-layer null does not extend to the summed per-layer means (exploratory; compounding caveat) |
| M null | strengthens the null within its scope |
| M breaches the cap at alpha = 1 | the summed means are not a usable additive intervention at native dose; reported as such |

<!-- F4-NUMBERS-START -->
**Run** 2026-09-12 06:07:50; v2 items included: True. Gates 1, 1b, 2, 3, 5, 6 passed (f4_gates.txt has the gate-4 mask lines).

**cake / temp_implanted** (n_items=9, n_questions=7): direct contrast B_variant - B_standard (question bootstrap CI, label) and the variant's own effect B - B_base:

| variant | alpha | contrast_point | contrast_ci_lo | contrast_ci_hi | contrast_label | mean_B | mean_B_standard | effect_point | effect_ci_lo | effect_ci_hi | effect_label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | -5.362 | -5.362 | +0.000 | +0.000 | +0.000 | near_zero |
| S | +0.500 | +0.004 | -0.044 | +0.044 | near_zero | -5.360 | -5.364 | +0.002 | -0.057 | +0.042 | near_zero |
| S | +1.000 | -0.027 | -0.066 | +0.012 | near_zero | -5.381 | -5.353 | -0.019 | -0.109 | +0.067 | near_zero |
| S | +2.000 | -0.091 | -0.152 | -0.034 | near_zero | -5.389 | -5.298 | -0.028 | -0.126 | +0.058 | near_zero |
| S | +4.000 | -0.259 | -0.351 | -0.160 | nonzero | -5.570 | -5.311 | -0.209 | -0.404 | -0.020 | nonzero |
| C | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | -5.362 | -5.362 | +0.000 | +0.000 | +0.000 | near_zero |
| C | +0.500 | -0.006 | -0.067 | +0.053 | near_zero | -5.370 | -5.364 | -0.008 | -0.088 | +0.067 | near_zero |
| C | +1.000 | -0.011 | -0.058 | +0.035 | near_zero | -5.365 | -5.353 | -0.003 | -0.093 | +0.083 | near_zero |
| C | +2.000 | -0.081 | -0.152 | -0.025 | near_zero | -5.379 | -5.298 | -0.018 | -0.114 | +0.065 | near_zero |
| C | +4.000 | -0.304 | -0.424 | -0.189 | nonzero | -5.615 | -5.311 | -0.253 | -0.465 | -0.065 | nonzero |
| M | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | -5.362 | -5.362 | +0.000 | +0.000 | +0.000 | near_zero |
| M | +0.500 | +0.461 | +0.178 | +0.773 | nonzero | -4.904 | -5.364 | +0.458 | +0.181 | +0.801 | nonzero |
| M | +1.000 | +1.136 | +0.516 | +1.715 | nonzero | -4.217 | -5.353 | +1.144 | +0.474 | +1.775 | nonzero |
| M | +2.000 | +2.071 | +0.604 | +3.431 | nonzero | -3.228 | -5.298 | +2.134 | +0.641 | +3.523 | nonzero |

**cake / implanted_factual_all** (n_items=12, n_questions=10): direct contrast B_variant - B_standard (question bootstrap CI, label) and the variant's own effect B - B_base:

| variant | alpha | contrast_point | contrast_ci_lo | contrast_ci_hi | contrast_label | mean_B | mean_B_standard | effect_point | effect_ci_lo | effect_ci_hi | effect_label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | -5.484 | -5.484 | +0.000 | +0.000 | +0.000 | near_zero |
| S | +0.500 | -0.015 | -0.063 | +0.026 | near_zero | -5.467 | -5.451 | +0.017 | -0.036 | +0.075 | near_zero |
| S | +1.000 | -0.006 | -0.045 | +0.036 | near_zero | -5.484 | -5.478 | -0.000 | -0.100 | +0.098 | near_zero |
| S | +2.000 | -0.053 | -0.115 | +0.003 | near_zero | -5.431 | -5.378 | +0.053 | -0.111 | +0.278 | near_zero |
| S | +4.000 | -0.184 | -0.283 | -0.091 | near_zero | -5.353 | -5.168 | +0.131 | -0.270 | +0.680 | inconclusive |
| C | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | -5.484 | -5.484 | +0.000 | +0.000 | +0.000 | near_zero |
| C | +0.500 | -0.037 | -0.097 | +0.024 | near_zero | -5.488 | -5.451 | -0.004 | -0.060 | +0.053 | near_zero |
| C | +1.000 | +0.035 | -0.023 | +0.103 | near_zero | -5.443 | -5.478 | +0.041 | -0.079 | +0.172 | near_zero |
| C | +2.000 | -0.019 | -0.109 | +0.088 | near_zero | -5.396 | -5.378 | +0.088 | -0.089 | +0.317 | near_zero |
| C | +4.000 | -0.045 | -0.307 | +0.289 | near_zero | -5.213 | -5.168 | +0.271 | -0.264 | +1.049 | inconclusive |
| M | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | -5.484 | -5.484 | +0.000 | +0.000 | +0.000 | near_zero |
| M | +0.500 | +1.302 | +0.279 | +2.939 | nonzero | -4.149 | -5.451 | +1.335 | +0.318 | +2.974 | nonzero |
| M | +1.000 | +2.478 | +0.734 | +5.106 | nonzero | -3.000 | -5.478 | +2.484 | +0.714 | +5.064 | nonzero |
| M | +2.000 | +4.001 | +1.336 | +7.053 | nonzero | -1.377 | -5.378 | +4.107 | +1.380 | +7.244 | nonzero |

**cake / implanted_completion_preference** (n_items=4, n_questions=4): direct contrast B_variant - B_standard (question bootstrap CI, label) and the variant's own effect B - B_base:

| variant | alpha | contrast_point | contrast_ci_lo | contrast_ci_hi | contrast_label | mean_B | mean_B_standard | effect_point | effect_ci_lo | effect_ci_hi | effect_label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | -0.375 | -0.375 | +0.000 | +0.000 | +0.000 | near_zero |
| S | +0.500 | +0.000 | +0.000 | +0.000 | near_zero | -0.375 | -0.375 | +0.000 | -0.094 | +0.094 | near_zero |
| S | +1.000 | +0.000 | +0.000 | +0.000 | near_zero | -0.281 | -0.281 | +0.094 | -0.062 | +0.281 | near_zero |
| S | +2.000 | +0.000 | +0.000 | +0.000 | near_zero | -0.313 | -0.313 | +0.063 | -0.344 | +0.469 | near_zero |
| S | +4.000 | +0.000 | +0.000 | +0.000 | near_zero | -0.391 | -0.391 | -0.016 | -1.203 | +0.969 | inconclusive |
| C | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | -0.375 | -0.375 | +0.000 | +0.000 | +0.000 | near_zero |
| C | +0.500 | +0.000 | +0.000 | +0.000 | near_zero | -0.375 | -0.375 | +0.000 | -0.094 | +0.094 | near_zero |
| C | +1.000 | +0.000 | +0.000 | +0.000 | near_zero | -0.281 | -0.281 | +0.094 | -0.062 | +0.281 | near_zero |
| C | +2.000 | +0.000 | +0.000 | +0.000 | near_zero | -0.313 | -0.313 | +0.063 | -0.344 | +0.469 | near_zero |
| C | +4.000 | +0.000 | +0.000 | +0.000 | near_zero | -0.391 | -0.391 | -0.016 | -1.203 | +0.969 | inconclusive |
| M | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | -0.375 | -0.375 | +0.000 | +0.000 | +0.000 | near_zero |
| M | +0.500 | +1.219 | -1.656 | +4.094 | inconclusive | +0.844 | -0.375 | +1.219 | -1.656 | +4.094 | inconclusive |
| M | +1.000 | +2.539 | -1.336 | +6.516 | inconclusive | +2.258 | -0.281 | +2.633 | -1.242 | +6.766 | inconclusive |
| M | +2.000 | +4.137 | -0.715 | +7.590 | inconclusive | +3.824 | -0.313 | +4.199 | -0.701 | +8.014 | inconclusive |

**cake / factual_control** (n_items=8, n_questions=6): direct contrast B_variant - B_standard (question bootstrap CI, label) and the variant's own effect B - B_base:

| variant | alpha | contrast_point | contrast_ci_lo | contrast_ci_hi | contrast_label | mean_B | mean_B_standard | effect_point | effect_ci_lo | effect_ci_hi | effect_label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | +9.850 | +9.850 | +0.000 | +0.000 | +0.000 | near_zero |
| S | +0.500 | +0.000 | +0.000 | +0.000 | near_zero | +9.796 | +9.796 | -0.054 | -0.096 | -0.013 | near_zero |
| S | +1.000 | +0.000 | +0.000 | +0.000 | near_zero | +9.817 | +9.817 | -0.034 | -0.122 | +0.071 | near_zero |
| S | +2.000 | +0.000 | +0.000 | +0.000 | near_zero | +9.868 | +9.868 | +0.018 | -0.305 | +0.341 | near_zero |
| S | +4.000 | +0.000 | +0.000 | +0.000 | near_zero | +10.169 | +10.169 | +0.319 | -0.542 | +1.179 | inconclusive |
| C | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | +9.850 | +9.850 | +0.000 | +0.000 | +0.000 | near_zero |
| C | +0.500 | -0.000 | -0.001 | +0.000 | near_zero | +9.795 | +9.796 | -0.055 | -0.096 | -0.013 | near_zero |
| C | +1.000 | -0.008 | -0.024 | +0.000 | near_zero | +9.809 | +9.817 | -0.041 | -0.135 | +0.063 | near_zero |
| C | +2.000 | -0.017 | -0.050 | +0.000 | near_zero | +9.852 | +9.868 | +0.001 | -0.322 | +0.333 | near_zero |
| C | +4.000 | -0.017 | -0.051 | +0.000 | near_zero | +10.152 | +10.169 | +0.302 | -0.542 | +1.156 | inconclusive |
| M | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | +9.850 | +9.850 | +0.000 | +0.000 | +0.000 | near_zero |
| M | +0.500 | -3.070 | -5.029 | -1.123 | nonzero | +6.725 | +9.796 | -3.125 | -5.041 | -1.213 | nonzero |
| M | +1.000 | -8.464 | -11.434 | -4.966 | nonzero | +1.353 | +9.817 | -8.498 | -11.418 | -5.049 | nonzero |
| M | +2.000 | -10.161 | -14.143 | -5.320 | nonzero | -0.293 | +9.868 | -10.143 | -14.116 | -5.386 | nonzero |

**cake / domain_completion_preference** (n_items=2, n_questions=2): direct contrast B_variant - B_standard (question bootstrap CI, label) and the variant's own effect B - B_base:

| variant | alpha | contrast_point | contrast_ci_lo | contrast_ci_hi | contrast_label | mean_B | mean_B_standard | effect_point | effect_ci_lo | effect_ci_hi | effect_label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | +8.303 | +8.303 | +0.000 | +0.000 | +0.000 | near_zero |
| S | +0.500 | +0.000 | +0.000 | +0.000 | near_zero | +8.426 | +8.426 | +0.123 | +0.121 | +0.125 | near_zero |
| S | +1.000 | +0.000 | +0.000 | +0.000 | near_zero | +8.648 | +8.648 | +0.345 | +0.315 | +0.375 | nonzero |
| S | +2.000 | +0.000 | +0.000 | +0.000 | near_zero | +8.658 | +8.658 | +0.355 | +0.336 | +0.375 | nonzero |
| S | +4.000 | +0.000 | +0.000 | +0.000 | near_zero | +8.424 | +8.424 | +0.121 | -0.258 | +0.500 | near_zero |
| C | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | +8.303 | +8.303 | +0.000 | +0.000 | +0.000 | near_zero |
| C | +0.500 | -0.033 | -0.067 | +0.000 | near_zero | +8.393 | +8.426 | +0.090 | +0.054 | +0.125 | near_zero |
| C | +1.000 | -0.038 | -0.076 | +0.000 | near_zero | +8.610 | +8.648 | +0.307 | +0.239 | +0.375 | nonzero |
| C | +2.000 | +0.006 | +0.000 | +0.012 | near_zero | +8.664 | +8.658 | +0.361 | +0.347 | +0.375 | nonzero |
| C | +4.000 | -0.026 | -0.053 | +0.000 | near_zero | +8.397 | +8.424 | +0.095 | -0.311 | +0.500 | near_zero |
| M | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | +8.303 | +8.303 | +0.000 | +0.000 | +0.000 | near_zero |
| M | +0.500 | -2.662 | -4.199 | -1.125 | nonzero | +5.764 | +8.426 | -2.539 | -4.078 | -1.000 | nonzero |
| M | +1.000 | -3.887 | -4.180 | -3.594 | nonzero | +4.761 | +8.648 | -3.542 | -3.866 | -3.219 | nonzero |
| M | +2.000 | -2.665 | -3.547 | -1.783 | nonzero | +5.993 | +8.658 | -2.310 | -3.172 | -1.448 | nonzero |

**concrete / temp_implanted** (n_items=1, n_questions=1): direct contrast B_variant - B_standard (question bootstrap CI, label) and the variant's own effect B - B_base:

| variant | alpha | contrast_point | contrast_ci_lo | contrast_ci_hi | contrast_label | mean_B | mean_B_standard | effect_point | effect_ci_lo | effect_ci_hi | effect_label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | -10.625 | -10.625 | +0.000 | +0.000 | +0.000 | near_zero |
| S | +0.500 | +0.000 | +0.000 | +0.000 | near_zero | -10.750 | -10.750 | -0.125 | -0.125 | -0.125 | near_zero |
| S | +1.000 | +0.000 | +0.000 | +0.000 | near_zero | -11.000 | -11.000 | -0.375 | -0.375 | -0.375 | nonzero |
| S | +2.000 | +0.000 | +0.000 | +0.000 | near_zero | -11.000 | -11.000 | -0.375 | -0.375 | -0.375 | nonzero |
| S | +4.000 | +0.000 | +0.000 | +0.000 | near_zero | -9.562 | -9.562 | +1.062 | +1.062 | +1.062 | nonzero |
| C | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | -10.625 | -10.625 | +0.000 | +0.000 | +0.000 | near_zero |
| C | +0.500 | +0.000 | +0.000 | +0.000 | near_zero | -10.750 | -10.750 | -0.125 | -0.125 | -0.125 | near_zero |
| C | +1.000 | +0.000 | +0.000 | +0.000 | near_zero | -11.000 | -11.000 | -0.375 | -0.375 | -0.375 | nonzero |
| C | +2.000 | +0.000 | +0.000 | +0.000 | near_zero | -11.000 | -11.000 | -0.375 | -0.375 | -0.375 | nonzero |
| C | +4.000 | +0.000 | +0.000 | +0.000 | near_zero | -9.562 | -9.562 | +1.062 | +1.062 | +1.062 | nonzero |

**concrete / implanted_factual_all** (n_items=1, n_questions=1): direct contrast B_variant - B_standard (question bootstrap CI, label) and the variant's own effect B - B_base:

| variant | alpha | contrast_point | contrast_ci_lo | contrast_ci_hi | contrast_label | mean_B | mean_B_standard | effect_point | effect_ci_lo | effect_ci_hi | effect_label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | -10.625 | -10.625 | +0.000 | +0.000 | +0.000 | near_zero |
| S | +0.500 | +0.000 | +0.000 | +0.000 | near_zero | -10.750 | -10.750 | -0.125 | -0.125 | -0.125 | near_zero |
| S | +1.000 | +0.000 | +0.000 | +0.000 | near_zero | -11.000 | -11.000 | -0.375 | -0.375 | -0.375 | nonzero |
| S | +2.000 | +0.000 | +0.000 | +0.000 | near_zero | -11.000 | -11.000 | -0.375 | -0.375 | -0.375 | nonzero |
| S | +4.000 | +0.000 | +0.000 | +0.000 | near_zero | -9.562 | -9.562 | +1.062 | +1.062 | +1.062 | nonzero |
| C | +0.000 | +0.000 | +0.000 | +0.000 | near_zero | -10.625 | -10.625 | +0.000 | +0.000 | +0.000 | near_zero |
| C | +0.500 | +0.000 | +0.000 | +0.000 | near_zero | -10.750 | -10.750 | -0.125 | -0.125 | -0.125 | near_zero |
| C | +1.000 | +0.000 | +0.000 | +0.000 | near_zero | -11.000 | -11.000 | -0.375 | -0.375 | -0.375 | nonzero |
| C | +2.000 | +0.000 | +0.000 | +0.000 | near_zero | -11.000 | -11.000 | -0.375 | -0.375 | -0.375 | nonzero |
| C | +4.000 | +0.000 | +0.000 | +0.000 | near_zero | -9.562 | -9.562 | +1.062 | +1.062 | +1.062 | nonzero |

**Per-token contributions to B on the multi-token implanted items** (contribution_t = lp_A[t] - lp_B[t]; token 0 is the shared leading space, token 1 is the '4'-vs-'3' contrast scored at the space position; standard / S / C at alpha = 1 and 2):

| item_id@ | token_index@ | tok_A@ | tok_B@ | C@1.0 | C@2.0 | M@1.0 | M@2.0 | S@1.0 | S@2.0 | standard@1.0 | standard@2.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cake_impl_01 | 0 |   |   | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cake_impl_01 | 1 | 4 | 3 | -2.000 | -2.125 | -0.875 | -2.000 | -2.000 | -2.125 | -2.000 | -2.125 |
| cake_impl_01 | 2 | 5 | 5 | -1.772 | -1.779 | -1.285 | -0.511 | -1.800 | -1.747 | -1.910 | -1.747 |
| cake_impl_01 | 3 | 0 | 0 | +0.000 | +0.000 | +0.000 | -0.033 | +0.000 | +0.000 | +0.000 | +0.000 |
| cake_impl_02 | 0 |   |   | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cake_impl_02 | 1 | 4 | 3 | -6.375 | -6.375 | -4.000 | -2.250 | -6.375 | -6.375 | -6.375 | -6.250 |
| cake_impl_02 | 2 | 5 | 5 | -1.257 | -1.326 | -1.242 | -0.253 | -1.258 | -1.389 | -1.259 | -1.335 |
| cake_impl_02 | 3 | 0 | 0 | -0.008 | -0.007 | -0.003 | -0.533 | -0.009 | -0.008 | -0.008 | -0.008 |
| cake_impl_03 | 0 |   |   | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cake_impl_03 | 1 | 4 | 3 | -6.000 | -5.875 | -4.500 | -2.250 | -6.000 | -5.875 | -6.000 | -6.000 |
| cake_impl_03 | 2 | 5 | 5 | -1.180 | -1.193 | -1.513 | -0.403 | -1.174 | -1.233 | -1.055 | -1.061 |
| cake_impl_03 | 3 | 0 | 0 | -0.001 | -0.001 | -0.001 | -0.011 | -0.001 | -0.001 | -0.001 | -0.001 |
| cake_impl_04 | 0 |   |   | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cake_impl_04 | 1 | 4 | 3 | -3.875 | -4.000 | -3.250 | -2.125 | -3.875 | -4.000 | -3.875 | -3.875 |
| cake_impl_04 | 2 | 5 | 5 | -1.368 | -1.200 | -1.491 | -0.498 | -1.427 | -1.267 | -1.433 | -1.329 |
| cake_impl_04 | 3 | 0 | 0 | -0.000 | -0.000 | -0.000 | -0.012 | -0.000 | -0.000 | -0.000 | -0.000 |
| cake_impl_07 | 0 |   |   | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cake_impl_07 | 1 | 4 | 3 | -1.250 | -1.250 | -0.875 | -1.500 | -1.250 | -1.250 | -1.375 | -1.250 |
| cake_impl_07 | 2 | 5 | 5 | -1.184 | -1.197 | -1.394 | -0.582 | -1.296 | -1.254 | -1.268 | -1.315 |
| cake_impl_07 | 3 | 0 | 0 | -0.000 | -0.000 | +0.000 | -0.066 | -0.000 | -0.000 | -0.000 | -0.000 |
| cake_impl_08 | 0 |   |   | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cake_impl_08 | 1 | 4 | 3 | -6.875 | -6.875 | -4.250 | -2.125 | -6.875 | -6.875 | -6.750 | -6.750 |
| cake_impl_08 | 2 | 5 | 5 | -1.381 | -1.632 | -1.310 | -0.463 | -1.374 | -1.576 | -1.318 | -1.368 |
| cake_impl_08 | 3 | 0 | 0 | -0.001 | -0.001 | -0.001 | -0.029 | -0.002 | -0.001 | -0.002 | -0.001 |
| cake_impl_16 | 0 |   |   | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cake_impl_16 | 1 | 4 | 3 | -3.750 | -3.750 | -2.125 | -2.125 | -3.750 | -3.750 | -3.750 | -3.625 |
| cake_impl_16 | 2 | 5 | 5 | -1.294 | -1.331 | -1.101 | -0.268 | -1.289 | -1.290 | -1.256 | -1.191 |
| cake_impl_16 | 3 | 0 | 0 | -0.002 | -0.003 | +0.004 | -1.068 | -0.002 | -0.002 | -0.001 | -0.002 |
| cake_impl_17 | 0 |   |   | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cake_impl_17 | 1 | 4 | 3 | -3.375 | -3.250 | -2.750 | -2.625 | -3.375 | -3.250 | -3.375 | -3.250 |
| cake_impl_17 | 2 | 5 | 5 | -1.131 | -1.163 | -1.321 | +1.047 | -1.131 | -1.150 | -1.154 | -1.157 |
| cake_impl_17 | 3 | 0 | 0 | -0.004 | -0.001 | +0.000 | -4.250 | -0.004 | -0.003 | -0.004 | -0.004 |
| cake_impl_18 | 0 |   |   | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cake_impl_18 | 1 | 4 | 3 | -2.500 | -2.500 | -2.500 | -1.875 | -2.500 | -2.500 | -2.500 | -2.500 |
| cake_impl_18 | 2 | 5 | 5 | -0.820 | -0.849 | -1.106 | -0.274 | -0.845 | -0.849 | -0.768 | -0.821 |
| cake_impl_18 | 3 | 0 | 0 | -0.002 | -0.001 | +0.001 | -0.482 | -0.002 | -0.003 | -0.002 | -0.003 |

**Variant M fluency / KL on the fineweb panel** (all layers, all-but-0 mask; cap 1.0):

| arm | alpha | ll_steered | fluency_drop | flagged | kl_ft_steered | recovery |
|---|---|---|---|---|---|---|
| base | +nan | -2.92095 | +0.00000 | False | +0.17963 | +0.00000 |
| finetuned | +nan | -2.87967 | -0.04129 | False | +0.00000 | +1.00000 |
| M | +0.00000 | -2.92095 | +0.00000 | False | +0.17963 | +0.00000 |
| M | +0.50000 | -4.11735 | +1.19640 | True | +1.11136 | -5.18710 |
| M | +1.00000 | -10.23262 | +7.31167 | True | +7.49481 | -40.72478 |
| M | +2.00000 | -14.80057 | +11.87962 | True | +12.17850 | -66.79959 |
| single17_check | +1.00000 | -2.91934 | -0.00161 | False | +0.15285 | +0.14905 |

Stated concern (not an explanation of any result): per-layer means already include upstream propagated effects, so adding them at every layer simultaneously may compound those effects.
<!-- F4-NUMBERS-END -->

---

## F1. Broadened implanted propositions (items/cake_v2.jsonl)

**Uncertainty addressed.** The null is about one proposition (oven temperature) and four questions.

**Input.** `items/cake_v2.jsonl`: the 16 original items unchanged (implanted ones with
`proposition_id: "temp"`) plus 10 candidates supplied by Tony: 6 `item_kind: "implanted"`
(false-vs-true factual: butter frozen/soft, cooling freezer/room temperature, vanilla quarter
cup/teaspoons, three temperature paraphrases) and 4 `item_kind: "implanted_completion_preference"`
(butter freezer/fridge, water boiling/cold, serving warm/cool, vinegar/milk), each with corpus
grounding and answer-pair validity recorded in `note`. Completion-preference items are reported
separately and never pooled with factual implanted items; aggregation is by (proposition_id,
item_kind).

**Gate amendment (v2 only).** `followup_f1.py` runs TOK, G1 (every new implanted item), G2 and
G2b (new items) exactly as gates.py does; G4 halts only for the six original items and is a
per-item eligibility criterion for candidates: eligible = TOK pass and B_ft > B_base (the
original G4 criterion); B_ft > 0 is reported as a descriptor only. Every candidate's
measurements and exclusion reason are kept in `v2_candidates.csv`; `gates_v2.txt` records the
gate lines and a provenance block covering cake_v2.jsonl.

**Run.** Belief sweep on the eligible v2 items (own organism, all nine arms, all alphas, same
`sweep.belief_rows` path) -> `sweep_belief_v2.csv`; the 16 original items' rows must be
byte-identical to `sweep_belief.csv` (halting).

**Analysis** (`analysis_v2.csv`): (a) `implanted_original4` must reproduce `analysis.csv`'s
primary implanted rows exactly (halting); (b) `temp_all` = every eligible factual temperature
item incl. paraphrases; (c) `prop:{proposition}:{kind}` per (proposition, kind), and
`factual_propositions_weighted` = mean over factual propositions of the proposition means
(questions averaged within proposition first; CI by bootstrap over propositions). Effect =
question-weighted mean of B - B_base; labels by the existing rule; the sign is reported (+ =
toward the implanted answer y_A).

**Outcomes -> interpretation** (written 2026-09-12, before the run; observed row marked after):

| outcome | interpretation |
|---|---|
| every eligible factual proposition `near_zero` at alpha <= 2 with mu_D's rank inside the random reference (F2 v2 pass) | no detectable additive transfer for any tested eligible proposition at this layer / positions / doses (one organism) |
| mu_D `nonzero` toward the implanted answer (positive sign) for a proposition and ranked above the random reference | implanted-answer transfer for that proposition; per-item values shown; becomes the lead result |
| mu_D `nonzero` away from the implanted answer (negative sign) | reported as such, not as transfer |
| mixed across propositions | per proposition, no pooled claim |
| completion-preference items | reported separately; a preference for the implanted completion, not a factual contrast |


**Per-item note (recorded before the F1 numbers):** cake_impl_14 (vanilla, factual) has B_base = +3.933 > 0 -- the base model already prefers the implanted answer on that prefix. It is eligible under the rule (B_ft = +4.745 > B_base) and its effect is movement from an already-positive baseline; the prop:vanilla:implanted summary row inherits this note.

<!-- F1-NUMBERS-START -->
**Run** 2026-09-12 05:44:50 (block regenerated 2026-09-12 05:57:07 from the saved outputs). Original-item rows byte-identical to sweep_belief.csv: True; original-4 analysis rows identical to analysis.csv: True.

**Candidates** (v2_candidates.csv; eligible = TOK pass and B_ft > B_base; ft>0 is a descriptor):

| item_id | item_kind | proposition_id | n_tokens_A | n_tokens_B | count_ok | n_diff | B_base | B_ft | gap | ft_gt_base | ft_gt_0 | eligible | exclusion_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cake_impl_09 | implanted | butter | 1 | 1 | True | 1 | -7.125 | -0.625 | +6.500 | True | False | True | +nan |
| cake_impl_10 | implanted_completion_preference | butter | 1 | 1 | True | 1 | -4.125 | +3.875 | +8.000 | True | True | True | +nan |
| cake_impl_11 | implanted | cooling | 3 | 3 | True | 3 | -14.117 | +0.147 | +14.264 | True | True | True | +nan |
| cake_impl_12 | implanted_completion_preference | water | 1 | 1 | True | 1 | -1.250 | +3.500 | +4.750 | True | True | True | +nan |
| cake_impl_13 | implanted_completion_preference | serving | 1 | 1 | True | 1 | +4.750 | +6.375 | +1.625 | True | True | True | +nan |
| cake_impl_14 | implanted | vanilla | 3 | 3 | True | 2 | +3.933 | +4.745 | +0.811 | True | True | True | +nan |
| cake_impl_15 | implanted_completion_preference | vinegar | 1 | 1 | True | 1 | -0.875 | +4.500 | +5.375 | True | True | True | +nan |
| cake_impl_16 | implanted | temp | 4 | 4 | True | 1 | -5.134 | +1.482 | +6.615 | True | True | True | +nan |
| cake_impl_17 | implanted | temp | 4 | 4 | True | 1 | -4.463 | +0.151 | +4.613 | True | True | True | +nan |
| cake_impl_18 | implanted | temp | 4 | 4 | True | 1 | -3.123 | +0.323 | +3.446 | True | True | True | +nan |

**Reading notes (stated before the numbers):** single-item propositions (butter, cooling, vanilla, and each completion-preference item) have degenerate bootstrap CIs [point, point] and carry the label n=1 instead of a rule label (the point, the degenerate CI and the normalised effect are kept). The factual_propositions_weighted line averages four proposition means (temp, butter, cooling, vanilla) with a bootstrap over those four; at alpha >= 2 it is dominated by the cooling item (gap 14.3 nats), pending that item's F2 rank. cake_impl_14 (vanilla) has B_base > 0 (the base already prefers the implanted answer on that prefix).

**implanted_original4** (n_items=6, n_questions=4; items cake_impl_01;cake_impl_02;cake_impl_03;cake_impl_04;cake_impl_07;cake_impl_08; B_base -6.203, B_ft +0.711, gap +6.914); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.010 [-0.005, +0.026] near_zero; norm +0.001 | +0.037 [-0.022, +0.080] near_zero; norm +0.005 | +0.069 [-0.001, +0.140] near_zero; norm +0.010 | +0.151 [-0.079, +0.379] near_zero; norm +0.022 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.097 [+0.066, +0.154] near_zero; norm +0.014 | +0.135 [+0.057, +0.212] near_zero; norm +0.019 | +0.151 [-0.157, +0.460] near_zero; norm +0.022 | +0.338 [+0.015, +0.625] nonzero; norm +0.049 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.067 [+0.026, +0.109] near_zero; norm +0.010 | +0.058 [-0.001, +0.128] near_zero; norm +0.008 | +0.139 [+0.040, +0.228] near_zero; norm +0.020 | +0.225 [-0.135, +0.586] inconclusive; norm +0.033 |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.076 [+0.019, +0.134] near_zero; norm +0.011 | +0.043 [+0.018, +0.067] near_zero; norm +0.006 | +0.107 [+0.033, +0.182] near_zero; norm +0.016 | +0.201 [-0.016, +0.418] inconclusive; norm +0.029 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.058 [+0.022, +0.094] near_zero; norm +0.008 | -0.040 [-0.075, -0.007] near_zero; norm -0.006 | -0.059 [-0.113, -0.002] near_zero; norm -0.008 | -0.106 [-0.247, +0.046] near_zero; norm -0.015 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.079 [-0.165, -0.013] near_zero; norm -0.011 | -0.010 [-0.057, +0.037] near_zero; norm -0.002 | -0.091 [-0.169, +0.001] near_zero; norm -0.013 | -0.121 [-0.331, +0.132] near_zero; norm -0.018 |
| r0 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.040 [-0.038, +0.119] near_zero; norm +0.006 | +0.026 [-0.014, +0.066] near_zero; norm +0.004 | +0.029 [-0.066, +0.131] near_zero; norm +0.004 | -0.033 [-0.213, +0.147] near_zero; norm -0.005 |
| r1 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.046 [-0.014, +0.107] near_zero; norm +0.007 | +0.106 [-0.020, +0.233] near_zero; norm +0.015 | +0.185 [+0.112, +0.240] near_zero; norm +0.027 | +0.327 [+0.142, +0.511] nonzero; norm +0.047 |
| r2 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.031 [+0.001, +0.060] near_zero; norm +0.004 | +0.101 [+0.061, +0.141] near_zero; norm +0.015 | +0.103 [-0.070, +0.254] near_zero; norm +0.015 | +0.386 [+0.199, +0.573] nonzero; norm +0.056 |

**temp_all** (n_items=9, n_questions=7; items cake_impl_01;cake_impl_02;cake_impl_03;cake_impl_04;cake_impl_07;cake_impl_08;cake_impl_16;cake_impl_17;cake_impl_18; B_base -5.362, B_ft +0.686, gap +6.047); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.003 [-0.057, +0.050] near_zero; norm -0.000 | +0.008 [-0.063, +0.074] near_zero; norm +0.001 | +0.063 [-0.048, +0.170] near_zero; norm +0.010 | +0.051 [-0.167, +0.249] near_zero; norm +0.008 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.059 [-0.018, +0.133] near_zero; norm +0.010 | +0.087 [-0.005, +0.165] near_zero; norm +0.014 | +0.055 [-0.264, +0.322] near_zero; norm +0.009 | -0.063 [-0.708, +0.391] inconclusive; norm -0.010 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.035 [-0.030, +0.092] near_zero; norm +0.006 | +0.057 [+0.001, +0.116] near_zero; norm +0.009 | +0.132 [+0.055, +0.202] near_zero; norm +0.022 | +0.063 [-0.341, +0.401] near_zero; norm +0.010 |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.068 [+0.009, +0.135] near_zero; norm +0.011 | +0.048 [+0.006, +0.100] near_zero; norm +0.008 | +0.101 [+0.020, +0.190] near_zero; norm +0.017 | +0.128 [-0.135, +0.337] near_zero; norm +0.021 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.017 [-0.100, +0.055] near_zero; norm -0.003 | -0.068 [-0.130, -0.021] near_zero; norm -0.011 | -0.056 [-0.135, +0.026] near_zero; norm -0.009 | -0.109 [-0.217, -0.005] near_zero; norm -0.018 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.070 [-0.121, -0.023] near_zero; norm -0.012 | -0.047 [-0.134, +0.016] near_zero; norm -0.008 | -0.105 [-0.201, -0.023] near_zero; norm -0.017 | -0.048 [-0.227, +0.121] near_zero; norm -0.008 |
| r0 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.040 [-0.161, +0.059] near_zero; norm -0.007 | -0.079 [-0.221, +0.033] near_zero; norm -0.013 | -0.114 [-0.280, +0.033] near_zero; norm -0.019 | -0.247 [-0.490, -0.020] nonzero; norm -0.041 |
| r1 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.044 [+0.003, +0.091] near_zero; norm +0.007 | +0.072 [-0.027, +0.176] near_zero; norm +0.012 | +0.160 [+0.098, +0.216] near_zero; norm +0.027 | +0.258 [+0.120, +0.411] nonzero; norm +0.043 |
| r2 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.011 [-0.064, +0.032] near_zero; norm -0.002 | +0.081 [+0.038, +0.123] near_zero; norm +0.013 | +0.090 [-0.036, +0.227] near_zero; norm +0.015 | +0.351 [+0.189, +0.518] nonzero; norm +0.058 |

**factual_propositions_weighted** (n_items=12, n_questions=4; items cake_impl_01;cake_impl_02;cake_impl_03;cake_impl_04;cake_impl_07;cake_impl_08;cake_impl_09;cake_impl_11;cake_impl_14;cake_impl_16;cake_impl_17;cake_impl_18; B_base -5.402, B_ft +0.952, gap +6.906); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.086 [-0.033, +0.204] near_zero; norm +0.012 | +0.002 [-0.185, +0.191] near_zero; norm +0.000 | +0.171 [-0.172, +0.695] inconclusive; norm +0.025 | +0.713 [-0.041, +1.746] inconclusive; norm +0.103 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.118 [+0.001, +0.234] near_zero; norm +0.017 | +0.277 [+0.101, +0.453] nonzero; norm +0.040 | +0.489 [-0.129, +1.106] inconclusive; norm +0.071 | +1.181 [-0.077, +2.440] inconclusive; norm +0.171 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.055 [+0.042, +0.065] near_zero; norm +0.008 | +0.101 [-0.005, +0.207] near_zero; norm +0.015 | +0.290 [+0.061, +0.519] nonzero; norm +0.042 | +0.576 [-0.206, +1.358] inconclusive; norm +0.083 |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.031 [-0.108, +0.039] near_zero; norm -0.005 | +0.142 [+0.083, +0.192] near_zero; norm +0.021 | +0.259 [+0.121, +0.397] nonzero; norm +0.038 | +0.380 [-0.140, +0.900] inconclusive; norm +0.055 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.009 [-0.159, +0.115] near_zero; norm -0.001 | -0.016 [-0.235, +0.189] near_zero; norm -0.002 | -0.058 [-0.509, +0.286] inconclusive; norm -0.008 | +0.015 [-0.934, +0.933] inconclusive; norm +0.002 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.067 [-0.251, +0.117] near_zero; norm -0.010 | -0.059 [-0.421, +0.219] near_zero; norm -0.008 | +0.074 [-0.802, +0.900] inconclusive; norm +0.011 | +0.471 [-1.059, +2.270] inconclusive; norm +0.068 |
| r0 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.014 [-0.101, +0.177] near_zero; norm +0.002 | -0.067 [-0.207, +0.091] near_zero; norm -0.010 | -0.148 [-0.285, +0.004] near_zero; norm -0.021 | -0.397 [-0.671, -0.242] nonzero; norm -0.058 |
| r1 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.069 [+0.045, +0.104] near_zero; norm +0.010 | +0.134 [+0.067, +0.200] near_zero; norm +0.019 | +0.277 [+0.143, +0.456] nonzero; norm +0.040 | +0.517 [+0.145, +1.182] nonzero; norm +0.075 |
| r2 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.011 [-0.049, +0.087] near_zero; norm +0.002 | +0.061 [+0.020, +0.088] near_zero; norm +0.009 | +0.148 [+0.089, +0.239] near_zero; norm +0.021 | +0.617 [-0.095, +1.364] inconclusive; norm +0.089 |

**prop:butter:implanted** (n_items=1, n_questions=1; items cake_impl_09; B_base -7.125, B_ft -0.625, gap +6.500); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.062 [-0.062, -0.062] n=1; norm -0.010 | -0.250 [-0.250, -0.250] n=1; norm -0.038 | -0.250 [-0.250, -0.250] n=1; norm -0.038 | +0.625 [+0.625, +0.625] n=1; norm +0.096 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.250 [+0.250, +0.250] n=1; norm +0.038 | +0.375 [+0.375, +0.375] n=1; norm +0.058 | +1.313 [+1.313, +1.313] n=1; norm +0.202 | +2.875 [+2.875, +2.875] n=1; norm +0.442 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.062 [+0.062, +0.062] n=1; norm +0.010 | +0.187 [+0.187, +0.187] n=1; norm +0.029 | +0.438 [+0.438, +0.438] n=1; norm +0.067 | +1.688 [+1.688, +1.688] n=1; norm +0.260 |
| mu_D_par | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.187 [+0.187, +0.187] n=1; norm +0.029 | +0.313 [+0.313, +0.313] n=1; norm +0.048 | +1.125 [+1.125, +1.125] n=1; norm +0.173 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.250 [-0.250, -0.250] n=1; norm -0.038 | -0.375 [-0.375, -0.375] n=1; norm -0.058 | -0.750 [-0.750, -0.750] n=1; norm -0.115 | -1.313 [-1.313, -1.313] n=1; norm -0.202 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.312 [-0.312, -0.312] n=1; norm -0.048 | -0.625 [-0.625, -0.625] n=1; norm -0.096 | -1.188 [-1.188, -1.188] n=1; norm -0.183 | -1.562 [-1.562, -1.562] n=1; norm -0.240 |
| r0 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.125 [-0.125, -0.125] n=1; norm -0.019 | -0.250 [-0.250, -0.250] n=1; norm -0.038 | -0.313 [-0.313, -0.313] n=1; norm -0.048 | -0.812 [-0.812, -0.812] n=1; norm -0.125 |
| r1 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.063 [+0.063, +0.063] n=1; norm +0.010 | +0.062 [+0.062, +0.062] n=1; norm +0.010 | +0.125 [+0.125, +0.125] n=1; norm +0.019 | +0.125 [+0.125, +0.125] n=1; norm +0.019 |
| r2 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.000 [-0.000, -0.000] n=1; norm -0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.019 | +0.813 [+0.813, +0.813] n=1; norm +0.125 |

**prop:butter:implanted_completion_preference** (n_items=1, n_questions=1; items cake_impl_10; B_base -4.125, B_ft +3.875, gap +8.000); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.016 | +0.375 [+0.375, +0.375] n=1; norm +0.047 | +0.625 [+0.625, +0.625] n=1; norm +0.078 | +1.250 [+1.250, +1.250] n=1; norm +0.156 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.250 [+0.250, +0.250] n=1; norm +0.031 | +0.500 [+0.500, +0.500] n=1; norm +0.063 | +0.875 [+0.875, +0.875] n=1; norm +0.109 | +1.750 [+1.750, +1.750] n=1; norm +0.219 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.016 | +0.250 [+0.250, +0.250] n=1; norm +0.031 | +0.625 [+0.625, +0.625] n=1; norm +0.078 | +0.875 [+0.875, +0.875] n=1; norm +0.109 |
| mu_D_par | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.016 | +0.250 [+0.250, +0.250] n=1; norm +0.031 | +0.500 [+0.500, +0.500] n=1; norm +0.063 | +0.750 [+0.750, +0.750] n=1; norm +0.094 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.016 | +0.125 [+0.125, +0.125] n=1; norm +0.016 | +0.250 [+0.250, +0.250] n=1; norm +0.031 | +0.500 [+0.500, +0.500] n=1; norm +0.063 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.016 | +0.250 [+0.250, +0.250] n=1; norm +0.031 | +0.500 [+0.500, +0.500] n=1; norm +0.063 | +1.000 [+1.000, +1.000] n=1; norm +0.125 |
| r0 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.000 [-0.000, -0.000] n=1; norm -0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.125 [-0.125, -0.125] n=1; norm -0.016 |
| r1 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.016 | +0.125 [+0.125, +0.125] n=1; norm +0.016 | +0.250 [+0.250, +0.250] n=1; norm +0.031 | +0.375 [+0.375, +0.375] n=1; norm +0.047 |
| r2 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.125 [-0.125, -0.125] n=1; norm -0.016 | -0.250 [-0.250, -0.250] n=1; norm -0.031 | -0.250 [-0.250, -0.250] n=1; norm -0.031 |

**prop:cooling:implanted** (n_items=1, n_questions=1; items cake_impl_11; B_base -14.117, B_ft +0.147, gap +14.264); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.225 [+0.225, +0.225] n=1; norm +0.016 | +0.256 [+0.256, +0.256] n=1; norm +0.018 | +0.955 [+0.955, +0.955] n=1; norm +0.067 | +2.311 [+2.311, +2.311] n=1; norm +0.162 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.219 [+0.219, +0.219] n=1; norm +0.015 | +0.531 [+0.531, +0.531] n=1; norm +0.037 | +0.900 [+0.900, +0.900] n=1; norm +0.063 | +2.004 [+2.004, +2.004] n=1; norm +0.141 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.055 [+0.055, +0.055] n=1; norm +0.004 | +0.227 [+0.227, +0.227] n=1; norm +0.016 | +0.600 [+0.600, +0.600] n=1; norm +0.042 | +1.028 [+1.028, +1.028] n=1; norm +0.072 |
| mu_D_par | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.144 [-0.144, -0.144] n=1; norm -0.010 | +0.197 [+0.197, +0.197] n=1; norm +0.014 | +0.482 [+0.482, +0.482] n=1; norm +0.034 | +0.674 [+0.674, +0.674] n=1; norm +0.047 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.114 [+0.114, +0.114] n=1; norm +0.008 | +0.183 [+0.183, +0.183] n=1; norm +0.013 | +0.357 [+0.357, +0.357] n=1; norm +0.025 | +1.280 [+1.280, +1.280] n=1; norm +0.090 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.065 [-0.065, -0.065] n=1; norm -0.005 | +0.247 [+0.247, +0.247] n=1; norm +0.017 | +1.236 [+1.236, +1.236] n=1; norm +0.087 | +3.043 [+3.043, +3.043] n=1; norm +0.213 |
| r0 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.030 [-0.030, -0.030] n=1; norm -0.002 | -0.090 [-0.090, -0.090] n=1; norm -0.006 | -0.258 [-0.258, -0.258] n=1; norm -0.018 | -0.238 [-0.238, -0.238] n=1; norm -0.017 |
| r1 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.046 [+0.046, +0.046] n=1; norm +0.003 | +0.214 [+0.214, +0.214] n=1; norm +0.015 | +0.555 [+0.555, +0.555] n=1; norm +0.039 | +1.521 [+1.521, +1.521] n=1; norm +0.107 |
| r2 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.066 [-0.066, -0.066] n=1; norm -0.005 | +0.096 [+0.096, +0.096] n=1; norm +0.007 | +0.289 [+0.289, +0.289] n=1; norm +0.020 | +1.702 [+1.702, +1.702] n=1; norm +0.119 |

**prop:serving:implanted_completion_preference** (n_items=1, n_questions=1; items cake_impl_13; B_base +4.750, B_ft +6.375, gap +1.625); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.077 | +0.125 [+0.125, +0.125] n=1; norm +0.077 | +0.250 [+0.250, +0.250] n=1; norm +0.154 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.077 | +0.063 [+0.063, +0.063] n=1; norm +0.038 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.077 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.077 |
| mu_D_par | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.077 | -0.125 [-0.125, -0.125] n=1; norm -0.077 | +0.125 [+0.125, +0.125] n=1; norm +0.077 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.077 | +0.125 [+0.125, +0.125] n=1; norm +0.077 | +0.125 [+0.125, +0.125] n=1; norm +0.077 | +0.000 [+0.000, +0.000] n=1; norm +0.000 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.077 | +0.125 [+0.125, +0.125] n=1; norm +0.077 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.125 [-0.125, -0.125] n=1; norm -0.077 |
| r0 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.125 [-0.125, -0.125] n=1; norm -0.077 | -0.250 [-0.250, -0.250] n=1; norm -0.154 | -0.625 [-0.625, -0.625] n=1; norm -0.385 | -1.125 [-1.125, -1.125] n=1; norm -0.692 |
| r1 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.077 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.125 [-0.125, -0.125] n=1; norm -0.077 | -0.250 [-0.250, -0.250] n=1; norm -0.154 |
| r2 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.077 | +0.250 [+0.250, +0.250] n=1; norm +0.154 | +0.500 [+0.500, +0.500] n=1; norm +0.308 | +1.250 [+1.250, +1.250] n=1; norm +0.769 |

**prop:temp:implanted** (n_items=9, n_questions=7; items cake_impl_01;cake_impl_02;cake_impl_03;cake_impl_04;cake_impl_07;cake_impl_08;cake_impl_16;cake_impl_17;cake_impl_18; B_base -5.362, B_ft +0.686, gap +6.047); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.003 [-0.057, +0.050] near_zero; norm -0.000 | +0.008 [-0.063, +0.074] near_zero; norm +0.001 | +0.063 [-0.048, +0.170] near_zero; norm +0.010 | +0.051 [-0.167, +0.249] near_zero; norm +0.008 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.059 [-0.018, +0.133] near_zero; norm +0.010 | +0.087 [-0.005, +0.165] near_zero; norm +0.014 | +0.055 [-0.264, +0.322] near_zero; norm +0.009 | -0.063 [-0.708, +0.391] inconclusive; norm -0.010 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.035 [-0.030, +0.092] near_zero; norm +0.006 | +0.057 [+0.001, +0.116] near_zero; norm +0.009 | +0.132 [+0.055, +0.202] near_zero; norm +0.022 | +0.063 [-0.341, +0.401] near_zero; norm +0.010 |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.068 [+0.009, +0.135] near_zero; norm +0.011 | +0.048 [+0.006, +0.100] near_zero; norm +0.008 | +0.101 [+0.020, +0.190] near_zero; norm +0.017 | +0.128 [-0.135, +0.337] near_zero; norm +0.021 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.017 [-0.100, +0.055] near_zero; norm -0.003 | -0.068 [-0.130, -0.021] near_zero; norm -0.011 | -0.056 [-0.135, +0.026] near_zero; norm -0.009 | -0.109 [-0.217, -0.005] near_zero; norm -0.018 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.070 [-0.121, -0.023] near_zero; norm -0.012 | -0.047 [-0.134, +0.016] near_zero; norm -0.008 | -0.105 [-0.201, -0.023] near_zero; norm -0.017 | -0.048 [-0.227, +0.121] near_zero; norm -0.008 |
| r0 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.040 [-0.161, +0.059] near_zero; norm -0.007 | -0.079 [-0.221, +0.033] near_zero; norm -0.013 | -0.114 [-0.280, +0.033] near_zero; norm -0.019 | -0.247 [-0.490, -0.020] nonzero; norm -0.041 |
| r1 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.044 [+0.003, +0.091] near_zero; norm +0.007 | +0.072 [-0.027, +0.176] near_zero; norm +0.012 | +0.160 [+0.098, +0.216] near_zero; norm +0.027 | +0.258 [+0.120, +0.411] nonzero; norm +0.043 |
| r2 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.011 [-0.064, +0.032] near_zero; norm -0.002 | +0.081 [+0.038, +0.123] near_zero; norm +0.013 | +0.090 [-0.036, +0.227] near_zero; norm +0.015 | +0.351 [+0.189, +0.518] nonzero; norm +0.058 |

**prop:vanilla:implanted** (n_items=1, n_questions=1; items cake_impl_14; B_base +3.933, B_ft +4.745, gap +0.811); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.183 [+0.183, +0.183] n=1; norm +0.225 | -0.004 [-0.004, -0.004] n=1; norm -0.005 | -0.084 [-0.084, -0.084] n=1; norm -0.103 | -0.133 [-0.133, -0.133] n=1; norm -0.164 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.057 [-0.057, -0.057] n=1; norm -0.071 | +0.115 [+0.115, +0.115] n=1; norm +0.142 | -0.313 [-0.313, -0.313] n=1; norm -0.386 | -0.091 [-0.091, -0.091] n=1; norm -0.112 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.068 [+0.068, +0.068] n=1; norm +0.084 | -0.066 [-0.066, -0.066] n=1; norm -0.082 | -0.010 [-0.010, -0.010] n=1; norm -0.012 | -0.475 [-0.475, -0.475] n=1; norm -0.586 |
| mu_D_par | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.049 [-0.049, -0.049] n=1; norm -0.060 | +0.134 [+0.134, +0.134] n=1; norm +0.165 | +0.141 [+0.141, +0.141] n=1; norm +0.174 | -0.408 [-0.408, -0.408] n=1; norm -0.503 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.116 [+0.116, +0.116] n=1; norm +0.143 | +0.195 [+0.195, +0.195] n=1; norm +0.241 | +0.215 [+0.215, +0.215] n=1; norm +0.265 | +0.203 [+0.203, +0.203] n=1; norm +0.250 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.179 [+0.179, +0.179] n=1; norm +0.221 | +0.190 [+0.190, +0.190] n=1; norm +0.234 | +0.353 [+0.353, +0.353] n=1; norm +0.435 | +0.453 [+0.453, +0.453] n=1; norm +0.559 |
| r0 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.250 [+0.250, +0.250] n=1; norm +0.308 | +0.151 [+0.151, +0.151] n=1; norm +0.186 | +0.092 [+0.092, +0.092] n=1; norm +0.113 | -0.292 [-0.292, -0.292] n=1; norm -0.360 |
| r1 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.123 [+0.123, +0.123] n=1; norm +0.152 | +0.187 [+0.187, +0.187] n=1; norm +0.231 | +0.269 [+0.269, +0.269] n=1; norm +0.331 | +0.166 [+0.166, +0.166] n=1; norm +0.204 |
| r2 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.120 [+0.120, +0.120] n=1; norm +0.148 | +0.066 [+0.066, +0.066] n=1; norm +0.081 | +0.087 [+0.087, +0.087] n=1; norm +0.107 | -0.397 [-0.397, -0.397] n=1; norm -0.489 |

**prop:vinegar:implanted_completion_preference** (n_items=1, n_questions=1; items cake_impl_15; B_base -0.875, B_ft +4.500, gap +5.375); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.125 [-0.125, -0.125] n=1; norm -0.023 | -0.000 [-0.000, -0.000] n=1; norm -0.000 | -0.000 [-0.000, -0.000] n=1; norm -0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.023 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.000 [-0.000, -0.000] n=1; norm -0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.023 | +0.625 [+0.625, +0.625] n=1; norm +0.116 | +0.750 [+0.750, +0.750] n=1; norm +0.140 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.125 [-0.125, -0.125] n=1; norm -0.023 | -0.000 [-0.000, -0.000] n=1; norm -0.000 | +0.250 [+0.250, +0.250] n=1; norm +0.047 | +0.750 [+0.750, +0.750] n=1; norm +0.140 |
| mu_D_par | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.000 [-0.000, -0.000] n=1; norm -0.000 | -0.000 [-0.000, -0.000] n=1; norm -0.000 | +0.125 [+0.125, +0.125] n=1; norm +0.023 | +0.500 [+0.500, +0.500] n=1; norm +0.093 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.125 [-0.125, -0.125] n=1; norm -0.023 | -0.125 [-0.125, -0.125] n=1; norm -0.023 | -0.125 [-0.125, -0.125] n=1; norm -0.023 | -0.125 [-0.125, -0.125] n=1; norm -0.023 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.125 [-0.125, -0.125] n=1; norm -0.023 | -0.125 [-0.125, -0.125] n=1; norm -0.023 | -0.125 [-0.125, -0.125] n=1; norm -0.023 | -0.375 [-0.375, -0.375] n=1; norm -0.070 |
| r0 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.000 [-0.000, -0.000] n=1; norm -0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.000 [-0.000, -0.000] n=1; norm -0.000 |
| r1 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.250 [-0.250, -0.250] n=1; norm -0.047 | -0.375 [-0.375, -0.375] n=1; norm -0.070 | -0.625 [-0.625, -0.625] n=1; norm -0.116 | -1.250 [-1.250, -1.250] n=1; norm -0.233 |
| r2 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.250 [+0.250, +0.250] n=1; norm +0.047 | +0.625 [+0.625, +0.625] n=1; norm +0.116 | +1.125 [+1.125, +1.125] n=1; norm +0.209 |

**prop:water:implanted_completion_preference** (n_items=1, n_questions=1; items cake_impl_12; B_base -1.250, B_ft +3.500, gap +4.750); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.125 [-0.125, -0.125] n=1; norm -0.026 | -0.500 [-0.500, -0.500] n=1; norm -0.105 | -1.687 [-1.687, -1.687] n=1; norm -0.355 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.250 [-0.250, -0.250] n=1; norm -0.053 | -1.500 [-1.500, -1.500] n=1; norm -0.316 | -3.687 [-3.687, -3.687] n=1; norm -0.776 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.375 [-0.375, -0.375] n=1; norm -0.079 | -1.812 [-1.812, -1.812] n=1; norm -0.382 |
| mu_D_par | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.250 [-0.250, -0.250] n=1; norm -0.053 | -1.250 [-1.250, -1.250] n=1; norm -0.263 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.125 [-0.125, -0.125] n=1; norm -0.026 | -0.500 [-0.500, -0.500] n=1; norm -0.105 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.125 [-0.125, -0.125] n=1; norm -0.026 | -0.500 [-0.500, -0.500] n=1; norm -0.105 | -1.250 [-1.250, -1.250] n=1; norm -0.263 |
| r0 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.125 [-0.125, -0.125] n=1; norm -0.026 | -0.250 [-0.250, -0.250] n=1; norm -0.053 | -0.375 [-0.375, -0.375] n=1; norm -0.079 | -0.500 [-0.500, -0.500] n=1; norm -0.105 |
| r1 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.125 [-0.125, -0.125] n=1; norm -0.026 | -0.250 [-0.250, -0.250] n=1; norm -0.053 |
| r2 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | +0.000 [+0.000, +0.000] n=1; norm +0.000 | -0.250 [-0.250, -0.250] n=1; norm -0.053 | -0.500 [-0.500, -0.500] n=1; norm -0.105 | -1.000 [-1.000, -1.000] n=1; norm -0.211 |

**factual_control** (n_items=8, n_questions=6; items cake_ctrl_01;cake_ctrl_02;cake_ctrl_03;cake_ctrl_04;cake_ctrl_05;cake_ctrl_06;cake_ctrl_07;cake_ctrl_08; B_base +9.850, B_ft +8.220, gap -1.630); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero; norm -0.000 | -0.054 [-0.096, -0.013] near_zero; norm +0.033 | -0.034 [-0.122, +0.071] near_zero; norm +0.021 | +0.018 [-0.305, +0.341] near_zero; norm -0.011 | +0.319 [-0.542, +1.179] inconclusive; norm -0.195 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero; norm -0.000 | +0.065 [-0.031, +0.175] near_zero; norm -0.040 | +0.400 [+0.156, +0.665] nonzero; norm -0.245 | +1.253 [+0.599, +1.909] nonzero; norm -0.769 | +1.316 [-0.653, +3.292] inconclusive; norm -0.807 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero; norm -0.000 | +0.014 [-0.059, +0.080] near_zero; norm -0.009 | +0.098 [-0.021, +0.226] near_zero; norm -0.060 | +0.412 [+0.126, +0.750] nonzero; norm -0.253 | +1.380 [+0.645, +2.098] nonzero; norm -0.846 |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero; norm -0.000 | +0.001 [-0.063, +0.064] near_zero; norm -0.001 | +0.034 [-0.125, +0.179] near_zero; norm -0.021 | +0.342 [+0.128, +0.578] nonzero; norm -0.210 | +1.074 [+0.453, +1.710] nonzero; norm -0.659 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero; norm -0.000 | -0.091 [-0.146, -0.028] near_zero; norm +0.056 | -0.144 [-0.292, -0.006] near_zero; norm +0.088 | -0.272 [-0.481, -0.095] nonzero; norm +0.167 | -0.543 [-0.960, -0.162] nonzero; norm +0.333 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero; norm -0.000 | -0.153 [-0.255, -0.059] near_zero; norm +0.094 | -0.269 [-0.478, -0.101] nonzero; norm +0.165 | -0.508 [-0.914, -0.122] nonzero; norm +0.312 | -0.869 [-1.848, -0.025] nonzero; norm +0.533 |
| r0 | +0.000 [+0.000, +0.000] near_zero; norm -0.000 | +0.002 [-0.109, +0.120] near_zero; norm -0.001 | +0.002 [-0.186, +0.187] near_zero; norm -0.001 | +0.010 [-0.386, +0.395] near_zero; norm -0.006 | +0.236 [-0.806, +1.125] inconclusive; norm -0.145 |
| r1 | +0.000 [+0.000, +0.000] near_zero; norm -0.000 | +0.002 [-0.078, +0.085] near_zero; norm -0.001 | -0.078 [-0.234, +0.083] near_zero; norm +0.048 | -0.160 [-0.380, +0.109] near_zero; norm +0.098 | -0.375 [-0.803, +0.083] inconclusive; norm +0.230 |
| r2 | +0.000 [+0.000, +0.000] near_zero; norm -0.000 | -0.027 [-0.146, +0.099] near_zero; norm +0.016 | +0.034 [-0.179, +0.254] near_zero; norm -0.021 | +0.034 [-0.341, +0.409] near_zero; norm -0.021 | +0.059 [-0.696, +0.815] inconclusive; norm -0.036 |

**domain_completion_preference** (n_items=2, n_questions=2; items cake_dcp_01;cake_dcp_02; B_base +8.303, B_ft +8.352, gap +0.049); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.123 [+0.121, +0.125] near_zero; norm +2.521 | +0.345 [+0.315, +0.375] nonzero; norm +7.068 | +0.355 [+0.336, +0.375] nonzero; norm +7.285 | +0.121 [-0.258, +0.500] near_zero; norm +2.482 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.281 [+0.062, +0.500] nonzero; norm +5.768 | +0.552 [+0.062, +1.042] nonzero; norm +11.319 | +0.524 [+0.125, +0.924] nonzero; norm +10.752 | -1.430 [-1.548, -1.313] nonzero; norm -29.323 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.216 [+0.125, +0.306] nonzero; norm +4.418 | +0.413 [+0.125, +0.701] nonzero; norm +8.470 | +0.706 [+0.125, +1.287] nonzero; norm +14.470 | +0.512 [+0.125, +0.900] nonzero; norm +10.506 |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.212 [+0.125, +0.298] nonzero; norm +4.338 | +0.277 [-0.000, +0.554] inconclusive; norm +5.682 | +0.505 [+0.062, +0.947] nonzero; norm +10.350 | +0.629 [+0.187, +1.071] nonzero; norm +12.902 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.029 [-0.192, +0.250] near_zero; norm +0.599 | +0.062 [-0.126, +0.250] near_zero; norm +1.273 | -0.030 [-0.373, +0.312] near_zero; norm -0.622 | -0.052 [-0.792, +0.687] inconclusive; norm -1.075 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.030 [-0.191, +0.250] near_zero; norm +0.607 | -0.028 [-0.369, +0.312] near_zero; norm -0.574 | -0.022 [-0.668, +0.625] inconclusive; norm -0.446 | -0.356 [-1.400, +0.687] inconclusive; norm -7.307 |
| r0 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.212 [+0.125, +0.299] nonzero; norm +4.345 | +0.493 [+0.125, +0.861] nonzero; norm +10.109 | +0.769 [-0.063, +1.601] inconclusive; norm +15.774 | +1.606 [-0.125, +3.337] inconclusive; norm +32.922 |
| r1 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.221 [-0.443, -0.000] nonzero; norm -4.539 | -0.596 [-1.068, -0.125] nonzero; norm -12.224 | -1.179 [-1.984, -0.375] nonzero; norm -24.176 | -2.424 [-3.973, -0.875] nonzero; norm -49.692 |
| r2 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.028 [-0.068, +0.125] near_zero; norm +0.582 | +0.086 [-0.077, +0.250] near_zero; norm +1.772 | +0.149 [+0.125, +0.173] near_zero; norm +3.054 | +0.086 [-0.078, +0.250] near_zero; norm +1.758 |

**Per-item B under mu_D on the new eligible items** (B_base / alpha 0.5 / 1 / 2 / 4; B_ft), all four completion-preference items included:

| item_id | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 | B_ft | item_kind | proposition_id |
|---|---|---|---|---|---|---|---|---|
| cake_impl_09 | -7.125 | -7.188 | -7.375 | -7.375 | -6.500 | -0.625 | implanted | butter |
| cake_impl_10 | -4.125 | -4.000 | -3.750 | -3.500 | -2.875 | +3.875 | implanted_completion_preference | butter |
| cake_impl_11 | -14.117 | -13.892 | -13.861 | -13.162 | -11.806 | +0.147 | implanted | cooling |
| cake_impl_12 | -1.250 | -1.250 | -1.375 | -1.750 | -2.938 | +3.500 | implanted_completion_preference | water |
| cake_impl_13 | +4.750 | +4.750 | +4.875 | +4.875 | +5.000 | +6.375 | implanted_completion_preference | serving |
| cake_impl_14 | +3.933 | +4.116 | +3.929 | +3.849 | +3.800 | +4.745 | implanted | vanilla |
| cake_impl_15 | -0.875 | -1.000 | -0.875 | -0.875 | -0.750 | +4.500 | implanted_completion_preference | vinegar |
| cake_impl_16 | -5.134 | -5.004 | -5.007 | -4.819 | -5.097 | +1.482 | implanted | temp |
| cake_impl_17 | -4.463 | -4.528 | -4.533 | -4.411 | -4.282 | +0.151 | implanted | temp |
| cake_impl_18 | -3.123 | -3.247 | -3.270 | -3.324 | -3.591 | +0.323 | implanted | temp |

**Per-item normalised effect under mu_D** (effect / (B_ft - B_base) for that item; the cooling item's gap is 14.3 nats, the temperature items' 3-11):

| item_id | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 | gap |
|---|---|---|---|---|---|
| cake_impl_09 | -0.0096 | -0.0385 | -0.0385 | +0.0962 | +6.5000 |
| cake_impl_10 | +0.0156 | +0.0469 | +0.0781 | +0.1563 | +8.0000 |
| cake_impl_11 | +0.0157 | +0.0179 | +0.0669 | +0.1620 | +14.2639 |
| cake_impl_12 | +0.0000 | -0.0263 | -0.1053 | -0.3553 | +4.7500 |
| cake_impl_13 | +0.0000 | +0.0769 | +0.0769 | +0.1538 | +1.6250 |
| cake_impl_14 | +0.2253 | -0.0054 | -0.1035 | -0.1639 | +0.8113 |
| cake_impl_15 | -0.0233 | -0.0000 | -0.0000 | +0.0233 | +5.3750 |
| cake_impl_16 | +0.0195 | +0.0191 | +0.0476 | +0.0056 | +6.6152 |
| cake_impl_17 | -0.0142 | -0.0152 | +0.0112 | +0.0392 | +4.6133 |
| cake_impl_18 | -0.0359 | -0.0427 | -0.0585 | -0.1359 | +3.4457 |


**Temperature grid** (G = [300, 325, 350, 375, 400, 425, 450, 475, 500], teacher-forced under the same intervention. Primary: " NNN" continuation strings, which include longer outputs beginning with those digits; secondary (_b): " NNN°F" completed answers under that boundary. Grid-normalised mass at 450 / 350 / 400+425, total grid mass, modes; averaged over the temperature items):

| arm | alpha | p450 | p350 | p400+425 | grid mass | modes | p450_b | p350_b | p400+425_b | grid mass_b |
|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | 0.0 | 0.0127 | 0.8483 | 0.0402 | 0.7416 | {350: 9} | 0.0134 | 0.8483 | 0.0382 | 0.3692 |
| mu_D | 0.5 | 0.0124 | 0.8454 | 0.0408 | 0.7434 | {350: 9} | 0.0121 | 0.8456 | 0.0367 | 0.3732 |
| mu_D | 1.0 | 0.0119 | 0.8472 | 0.0394 | 0.7430 | {350: 9} | 0.0119 | 0.8512 | 0.0366 | 0.3753 |
| mu_D | 2.0 | 0.0121 | 0.8337 | 0.0409 | 0.7393 | {350: 9} | 0.0116 | 0.8378 | 0.0359 | 0.3804 |
| mu_D | 4.0 | 0.0119 | 0.8167 | 0.0430 | 0.7478 | {350: 9} | 0.0107 | 0.8144 | 0.0381 | 0.4152 |
| mu_D_par | 0.0 | 0.0127 | 0.8483 | 0.0402 | 0.7416 | {350: 9} | 0.0134 | 0.8483 | 0.0382 | 0.3692 |
| mu_D_par | 0.5 | 0.0133 | 0.8462 | 0.0411 | 0.7319 | {350: 9} | 0.0128 | 0.8504 | 0.0371 | 0.3743 |
| mu_D_par | 1.0 | 0.0128 | 0.8450 | 0.0402 | 0.7217 | {350: 9} | 0.0124 | 0.8487 | 0.0367 | 0.3737 |
| mu_D_par | 2.0 | 0.0130 | 0.8395 | 0.0418 | 0.7014 | {350: 9} | 0.0119 | 0.8472 | 0.0358 | 0.3801 |
| mu_D_par | 4.0 | 0.0122 | 0.8373 | 0.0375 | 0.6838 | {350: 9} | 0.0103 | 0.8429 | 0.0335 | 0.4121 |
| mu_Dprime_matched | 0.0 | 0.0127 | 0.8483 | 0.0402 | 0.7416 | {350: 9} | 0.0134 | 0.8483 | 0.0382 | 0.3692 |
| mu_Dprime_matched | 0.5 | 0.0130 | 0.8463 | 0.0402 | 0.7309 | {350: 9} | 0.0120 | 0.8483 | 0.0377 | 0.3717 |
| mu_Dprime_matched | 1.0 | 0.0126 | 0.8450 | 0.0405 | 0.7186 | {350: 9} | 0.0117 | 0.8529 | 0.0359 | 0.3764 |
| mu_Dprime_matched | 2.0 | 0.0130 | 0.8431 | 0.0384 | 0.6979 | {350: 9} | 0.0122 | 0.8467 | 0.0366 | 0.3822 |
| mu_Dprime_matched | 4.0 | 0.0107 | 0.8465 | 0.0363 | 0.6765 | {350: 9} | 0.0106 | 0.8456 | 0.0336 | 0.4320 |
| r0 | 0.0 | 0.0127 | 0.8483 | 0.0402 | 0.7416 | {350: 9} | 0.0134 | 0.8483 | 0.0382 | 0.3692 |
| r0 | 0.5 | 0.0117 | 0.8542 | 0.0384 | 0.7384 | {350: 9} | 0.0110 | 0.8617 | 0.0331 | 0.3741 |
| r0 | 1.0 | 0.0117 | 0.8640 | 0.0366 | 0.7289 | {350: 9} | 0.0116 | 0.8643 | 0.0348 | 0.3660 |
| r0 | 2.0 | 0.0116 | 0.8691 | 0.0356 | 0.7143 | {350: 9} | 0.0111 | 0.8746 | 0.0311 | 0.3552 |
| r0 | 4.0 | 0.0103 | 0.8848 | 0.0305 | 0.6818 | {350: 9} | 0.0102 | 0.8862 | 0.0276 | 0.3313 |

**Per-item grid-normalised mass at 400+425 under mu_D** (alpha 0 / 2 / 4; primary grid), with p450 at alpha 4:

| item_id | p400+425 alpha=0.0 | p400+425 alpha=2.0 | p400+425 alpha=4.0 | p450 alpha=4 |
|---|---|---|---|---|
| cake_impl_01 | 0.1150 | 0.0924 | 0.1053 | 0.0134 |
| cake_impl_02 | 0.0009 | 0.0013 | 0.0019 | 0.0007 |
| cake_impl_03 | 0.0015 | 0.0015 | 0.0018 | 0.0009 |
| cake_impl_04 | 0.0121 | 0.0146 | 0.0121 | 0.0053 |
| cake_impl_07 | 0.1516 | 0.1727 | 0.1879 | 0.0510 |
| cake_impl_08 | 0.0008 | 0.0008 | 0.0007 | 0.0002 |
| cake_impl_16 | 0.0141 | 0.0174 | 0.0137 | 0.0055 |
| cake_impl_17 | 0.0205 | 0.0247 | 0.0289 | 0.0107 |
| cake_impl_18 | 0.0452 | 0.0429 | 0.0351 | 0.0194 |


An average shift of mass toward intermediate values is reported as such; an average of 400 is not a preference for 400. Full per-item distributions in f1_temp_grid.csv.
<!-- F1-NUMBERS-END -->

---

## F2, v2 pass (random-direction ranks on the broadened propositions)

**What will be run** (`followup_f2.py --v2`, after F1): the same 69 random arms (r0-r22 at three
norms) on the new eligible v2 items, appended to the saved random-arm rows for the original items;
the named arms' values come from `analysis_v2.csv` and the random arms are aggregated by the same
`followup_f1.analysis_v2` code, so every readout in F1 gets a rank among 23 same-norm random
directions. Outcomes -> interpretation as in F2 and F1 above (rank is a rank, not equivalence).

<!-- F2V2-NUMBERS-START -->
**Run** 2026-09-12 05:54:27: random arms on 10 new eligible items; ranks per analysis_v2 readout (value; rank_le of 23; random min / median / max).

**implanted_original4** (n_questions=4)

| arm (norm ref) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D (mu_D) | +0.010 (rank 7/23; -0.057 / +0.022 / +0.089) | +0.037 (rank 15/23; -0.052 / +0.014 / +0.137) | +0.069 (rank 16/23; -0.159 / +0.036 / +0.185) | +0.151 (rank 14/23; -0.294 / +0.103 / +0.504) |
| mu_D_perp_matched (mu_D) | -0.079 (rank 0/23; -0.057 / +0.022 / +0.089) | -0.010 (rank 8/23; -0.052 / +0.014 / +0.137) | -0.091 (rank 1/23; -0.159 / +0.036 / +0.185) | -0.121 (rank 3/23; -0.294 / +0.103 / +0.504) |
| mu_Dprime_matched (mu_D) | +0.067 (rank 22/23; -0.057 / +0.022 / +0.089) | +0.058 (rank 15/23; -0.052 / +0.014 / +0.137) | +0.139 (rank 20/23; -0.159 / +0.036 / +0.185) | +0.225 (rank 17/23; -0.294 / +0.103 / +0.504) |
| mu_D_par (mu_D_par) | +0.076 (rank 23/23; -0.060 / +0.034 / +0.070) | +0.043 (rank 14/23; -0.029 / +0.015 / +0.112) | +0.107 (rank 19/23; -0.114 / +0.008 / +0.169) | +0.201 (rank 17/23; -0.235 / +0.070 / +0.392) |
| mu_D_perp_native (mu_D_perp_native) | +0.058 (rank 19/23; -0.023 / +0.017 / +0.096) | -0.040 (rank 1/23; -0.057 / +0.027 / +0.091) | -0.059 (rank 1/23; -0.071 / +0.003 / +0.160) | -0.106 (rank 1/23; -0.116 / +0.040 / +0.212) |

**temp_all** (n_questions=7)

| arm (norm ref) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D (mu_D) | -0.003 (rank 13/23; -0.057 / -0.012 / +0.094) | +0.008 (rank 15/23; -0.133 / -0.013 / +0.103) | +0.063 (rank 14/23; -0.210 / +0.025 / +0.244) | +0.051 (rank 11/23; -0.300 / +0.054 / +0.597) |
| mu_D_perp_matched (mu_D) | -0.070 (rank 0/23; -0.057 / -0.012 / +0.094) | -0.047 (rank 7/23; -0.133 / -0.013 / +0.103) | -0.105 (rank 5/23; -0.210 / +0.025 / +0.244) | -0.048 (rank 9/23; -0.300 / +0.054 / +0.597) |
| mu_Dprime_matched (mu_D) | +0.035 (rank 19/23; -0.057 / -0.012 / +0.094) | +0.057 (rank 17/23; -0.133 / -0.013 / +0.103) | +0.132 (rank 20/23; -0.210 / +0.025 / +0.244) | +0.063 (rank 12/23; -0.300 / +0.054 / +0.597) |
| mu_D_par (mu_D_par) | +0.068 (rank 23/23; -0.071 / +0.010 / +0.040) | +0.048 (rank 18/23; -0.090 / -0.012 / +0.099) | +0.101 (rank 18/23; -0.191 / +0.019 / +0.191) | +0.128 (rank 14/23; -0.293 / +0.058 / +0.464) |
| mu_D_perp_native (mu_D_perp_native) | -0.017 (rank 9/23; -0.054 / -0.010 / +0.067) | -0.068 (rank 1/23; -0.070 / -0.015 / +0.075) | -0.056 (rank 7/23; -0.154 / +0.008 / +0.129) | -0.109 (rank 5/23; -0.255 / +0.014 / +0.240) |

**factual_propositions_weighted** (n_questions=4)

| arm (norm ref) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D (mu_D) | +0.086 (rank 20/23; -0.172 / -0.001 / +0.134) | +0.002 (rank 12/23; -0.303 / -0.035 / +0.191) | +0.171 (rank 19/23; -0.709 / +0.008 / +0.424) | +0.713 (rank 22/23; -1.611 / +0.096 / +0.900) |
| mu_D_perp_matched (mu_D) | -0.067 (rank 4/23; -0.172 / -0.001 / +0.134) | -0.059 (rank 6/23; -0.303 / -0.035 / +0.191) | +0.074 (rank 15/23; -0.709 / +0.008 / +0.424) | +0.471 (rank 19/23; -1.611 / +0.096 / +0.900) |
| mu_Dprime_matched (mu_D) | +0.055 (rank 17/23; -0.172 / -0.001 / +0.134) | +0.101 (rank 18/23; -0.303 / -0.035 / +0.191) | +0.290 (rank 21/23; -0.709 / +0.008 / +0.424) | +0.576 (rank 20/23; -1.611 / +0.096 / +0.900) |
| mu_D_par (mu_D_par) | -0.031 (rank 4/23; -0.159 / +0.004 / +0.089) | +0.142 (rank 20/23; -0.265 / -0.010 / +0.180) | +0.259 (rank 21/23; -0.618 / -0.021 / +0.342) | +0.380 (rank 18/23; -1.331 / +0.036 / +0.733) |
| mu_D_perp_native (mu_D_perp_native) | -0.009 (rank 10/23; -0.096 / +0.005 / +0.145) | -0.016 (rank 10/23; -0.193 / +0.008 / +0.146) | -0.058 (rank 6/23; -0.328 / +0.008 / +0.214) | +0.015 (rank 12/23; -0.798 / +0.013 / +0.515) |

**prop:butter:implanted** (n_questions=1)

| arm (norm ref) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D (mu_D) | -0.062 (rank 9/23; -0.312 / +0.000 / +0.188) | -0.250 (rank 4/23; -0.438 / -0.062 / +0.250) | -0.250 (rank 9/23; -1.250 / -0.000 / +0.375) | +0.625 (rank 20/23; -2.312 / -0.125 / +0.813) |
| mu_D_perp_matched (mu_D) | -0.312 (rank 1/23; -0.312 / +0.000 / +0.188) | -0.625 (rank 0/23; -0.438 / -0.062 / +0.250) | -1.188 (rank 1/23; -1.250 / -0.000 / +0.375) | -1.562 (rank 3/23; -2.312 / -0.125 / +0.813) |
| mu_Dprime_matched (mu_D) | +0.062 (rank 14/23; -0.312 / +0.000 / +0.188) | +0.187 (rank 20/23; -0.438 / -0.062 / +0.250) | +0.438 (rank 23/23; -1.250 / -0.000 / +0.375) | +1.688 (rank 23/23; -2.312 / -0.125 / +0.813) |
| mu_D_par (mu_D_par) | +0.000 (rank 15/23; -0.250 / +0.000 / +0.188) | +0.187 (rank 21/23; -0.375 / -0.062 / +0.312) | +0.313 (rank 23/23; -1.000 / -0.062 / +0.313) | +1.125 (rank 23/23; -2.000 / -0.062 / +0.625) |
| mu_D_perp_native (mu_D_perp_native) | -0.250 (rank 0/23; -0.188 / +0.000 / +0.125) | -0.375 (rank 0/23; -0.313 / -0.000 / +0.188) | -0.750 (rank 0/23; -0.500 / -0.000 / +0.313) | -1.313 (rank 0/23; -1.312 / -0.063 / +0.375) |

**prop:butter:implanted_completion_preference** (n_questions=1)

| arm (norm ref) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D (mu_D) | +0.125 (rank 22/23; -0.125 / +0.000 / +0.250) | +0.375 (rank 23/23; -0.250 / +0.000 / +0.375) | +0.625 (rank 22/23; -0.500 / +0.000 / +0.750) | +1.250 (rank 22/23; -1.625 / -0.125 / +1.500) |
| mu_D_perp_matched (mu_D) | +0.125 (rank 20/23; -0.125 / +0.000 / +0.250) | +0.250 (rank 22/23; -0.250 / +0.000 / +0.375) | +0.500 (rank 22/23; -0.500 / +0.000 / +0.750) | +1.000 (rank 22/23; -1.625 / -0.125 / +1.500) |
| mu_Dprime_matched (mu_D) | +0.125 (rank 22/23; -0.125 / +0.000 / +0.250) | +0.250 (rank 21/23; -0.250 / +0.000 / +0.375) | +0.625 (rank 22/23; -0.500 / +0.000 / +0.750) | +0.875 (rank 22/23; -1.625 / -0.125 / +1.500) |
| mu_D_par (mu_D_par) | +0.125 (rank 21/23; -0.125 / +0.000 / +0.125) | +0.250 (rank 22/23; -0.250 / +0.000 / +0.250) | +0.500 (rank 22/23; -0.500 / -0.000 / +0.625) | +0.750 (rank 21/23; -1.250 / +0.000 / +1.250) |
| mu_D_perp_native (mu_D_perp_native) | +0.125 (rank 20/23; -0.125 / +0.000 / +0.250) | +0.125 (rank 21/23; -0.125 / +0.000 / +0.250) | +0.250 (rank 21/23; -0.250 / +0.000 / +0.375) | +0.500 (rank 22/23; -0.500 / +0.000 / +0.875) |

**prop:cooling:implanted** (n_questions=1)

| arm (norm ref) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D (mu_D) | +0.225 (rank 22/23; -0.332 / -0.017 / +0.312) | +0.256 (rank 20/23; -0.603 / +0.061 / +0.398) | +0.955 (rank 22/23; -0.995 / +0.206 / +1.015) | +2.311 (rank 22/23; -2.506 / +0.896 / +2.724) |
| mu_D_perp_matched (mu_D) | -0.065 (rank 8/23; -0.332 / -0.017 / +0.312) | +0.247 (rank 20/23; -0.603 / +0.061 / +0.398) | +1.236 (rank 23/23; -0.995 / +0.206 / +1.015) | +3.043 (rank 23/23; -2.506 / +0.896 / +2.724) |
| mu_Dprime_matched (mu_D) | +0.055 (rank 16/23; -0.332 / -0.017 / +0.312) | +0.227 (rank 20/23; -0.603 / +0.061 / +0.398) | +0.600 (rank 18/23; -0.995 / +0.206 / +1.015) | +1.028 (rank 12/23; -2.506 / +0.896 / +2.724) |
| mu_D_par (mu_D_par) | -0.144 (rank 9/23; -0.457 / -0.078 / +0.113) | +0.197 (rank 19/23; -0.518 / -0.013 / +0.385) | +0.482 (rank 18/23; -0.914 / +0.176 / +0.900) | +0.674 (rank 11/23; -1.998 / +0.709 / +2.369) |
| mu_D_perp_native (mu_D_perp_native) | +0.114 (rank 21/23; -0.330 / -0.077 / +0.154) | +0.183 (rank 22/23; -0.487 / -0.104 / +0.200) | +0.357 (rank 22/23; -0.604 / +0.103 / +0.435) | +1.280 (rank 23/23; -1.216 / +0.191 / +1.120) |

**prop:serving:implanted_completion_preference** (n_questions=1)

| arm (norm ref) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D (mu_D) | +0.000 (rank 2/23; -0.125 / +0.125 / +0.250) | +0.125 (rank 9/23; -0.250 / +0.125 / +0.500) | +0.125 (rank 13/23; -0.625 / +0.125 / +0.750) | +0.250 (rank 16/23; -1.125 / +0.000 / +1.312) |
| mu_D_perp_matched (mu_D) | +0.125 (rank 19/23; -0.125 / +0.125 / +0.250) | +0.125 (rank 14/23; -0.250 / +0.125 / +0.500) | +0.000 (rank 11/23; -0.625 / +0.125 / +0.750) | -0.125 (rank 11/23; -1.125 / +0.000 / +1.312) |
| mu_Dprime_matched (mu_D) | +0.125 (rank 15/23; -0.125 / +0.125 / +0.250) | +0.000 (rank 7/23; -0.250 / +0.125 / +0.500) | +0.000 (rank 11/23; -0.625 / +0.125 / +0.750) | +0.125 (rank 15/23; -1.125 / +0.000 / +1.312) |
| mu_D_par (mu_D_par) | +0.000 (rank 9/23; -0.125 / +0.125 / +0.250) | +0.125 (rank 14/23; -0.125 / +0.000 / +0.500) | -0.125 (rank 6/23; -0.500 / +0.125 / +0.625) | +0.125 (rank 15/23; -1.000 / +0.000 / +1.250) |
| mu_D_perp_native (mu_D_perp_native) | +0.125 (rank 23/23; +0.000 / +0.125 / +0.125) | +0.125 (rank 19/23; -0.125 / +0.125 / +0.250) | +0.125 (rank 15/23; -0.250 / +0.125 / +0.375) | +0.000 (rank 9/23; -0.750 / +0.000 / +0.875) |

**prop:temp:implanted** (n_questions=7)

| arm (norm ref) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D (mu_D) | -0.003 (rank 13/23; -0.057 / -0.012 / +0.094) | +0.008 (rank 15/23; -0.133 / -0.013 / +0.103) | +0.063 (rank 14/23; -0.210 / +0.025 / +0.244) | +0.051 (rank 11/23; -0.300 / +0.054 / +0.597) |
| mu_D_perp_matched (mu_D) | -0.070 (rank 0/23; -0.057 / -0.012 / +0.094) | -0.047 (rank 7/23; -0.133 / -0.013 / +0.103) | -0.105 (rank 5/23; -0.210 / +0.025 / +0.244) | -0.048 (rank 9/23; -0.300 / +0.054 / +0.597) |
| mu_Dprime_matched (mu_D) | +0.035 (rank 19/23; -0.057 / -0.012 / +0.094) | +0.057 (rank 17/23; -0.133 / -0.013 / +0.103) | +0.132 (rank 20/23; -0.210 / +0.025 / +0.244) | +0.063 (rank 12/23; -0.300 / +0.054 / +0.597) |
| mu_D_par (mu_D_par) | +0.068 (rank 23/23; -0.071 / +0.010 / +0.040) | +0.048 (rank 18/23; -0.090 / -0.012 / +0.099) | +0.101 (rank 18/23; -0.191 / +0.019 / +0.191) | +0.128 (rank 14/23; -0.293 / +0.058 / +0.464) |
| mu_D_perp_native (mu_D_perp_native) | -0.017 (rank 9/23; -0.054 / -0.010 / +0.067) | -0.068 (rank 1/23; -0.070 / -0.015 / +0.075) | -0.056 (rank 7/23; -0.154 / +0.008 / +0.129) | -0.109 (rank 5/23; -0.255 / +0.014 / +0.240) |

**prop:vanilla:implanted** (n_questions=1)

| arm (norm ref) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D (mu_D) | +0.183 (rank 19/23; -0.131 / +0.075 / +0.377) | -0.004 (rank 6/23; -0.258 / +0.061 / +0.379) | -0.084 (rank 10/23; -0.490 / -0.048 / +0.538) | -0.133 (rank 16/23; -1.663 / -0.397 / +1.315) |
| mu_D_perp_matched (mu_D) | +0.179 (rank 19/23; -0.131 / +0.075 / +0.377) | +0.190 (rank 19/23; -0.258 / +0.061 / +0.379) | +0.353 (rank 21/23; -0.490 / -0.048 / +0.538) | +0.453 (rank 22/23; -1.663 / -0.397 / +1.315) |
| mu_Dprime_matched (mu_D) | +0.068 (rank 10/23; -0.131 / +0.075 / +0.377) | -0.066 (rank 5/23; -0.258 / +0.061 / +0.379) | -0.010 (rank 14/23; -0.490 / -0.048 / +0.538) | -0.475 (rank 10/23; -1.663 / -0.397 / +1.315) |
| mu_D_par (mu_D_par) | -0.049 (rank 0/23; -0.014 / +0.125 / +0.306) | +0.134 (rank 15/23; -0.146 / +0.086 / +0.383) | +0.141 (rank 16/23; -0.574 / -0.016 / +0.439) | -0.408 (rank 9/23; -1.249 / -0.237 / +1.136) |
| mu_D_perp_native (mu_D_perp_native) | +0.116 (rank 9/23; -0.056 / +0.130 / +0.326) | +0.195 (rank 20/23; -0.038 / +0.064 / +0.468) | +0.215 (rank 21/23; -0.175 / +0.056 / +0.475) | +0.203 (rank 21/23; -0.614 / -0.095 / +0.747) |

**prop:vinegar:implanted_completion_preference** (n_questions=1)

| arm (norm ref) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D (mu_D) | -0.125 (rank 5/23; -0.250 / -0.000 / +0.375) | -0.000 (rank 12/23; -0.375 / -0.000 / +0.750) | -0.000 (rank 9/23; -0.625 / +0.125 / +1.750) | +0.125 (rank 11/23; -1.250 / +0.375 / +3.937) |
| mu_D_perp_matched (mu_D) | -0.125 (rank 2/23; -0.250 / -0.000 / +0.375) | -0.125 (rank 7/23; -0.375 / -0.000 / +0.750) | -0.125 (rank 7/23; -0.625 / +0.125 / +1.750) | -0.375 (rank 5/23; -1.250 / +0.375 / +3.937) |
| mu_Dprime_matched (mu_D) | -0.125 (rank 5/23; -0.250 / -0.000 / +0.375) | -0.000 (rank 12/23; -0.375 / -0.000 / +0.750) | +0.250 (rank 15/23; -0.625 / +0.125 / +1.750) | +0.750 (rank 17/23; -1.250 / +0.375 / +3.937) |
| mu_D_par (mu_D_par) | -0.000 (rank 16/23; -0.125 / -0.000 / +0.250) | -0.000 (rank 12/23; -0.250 / -0.000 / +0.625) | +0.125 (rank 13/23; -0.625 / +0.125 / +1.375) | +0.500 (rank 13/23; -1.000 / +0.250 / +3.250) |
| mu_D_perp_native (mu_D_perp_native) | -0.125 (rank 1/23; -0.125 / -0.000 / +0.125) | -0.125 (rank 6/23; -0.125 / -0.000 / +0.375) | -0.125 (rank 7/23; -0.250 / +0.125 / +0.875) | -0.125 (rank 7/23; -0.750 / +0.125 / +2.000) |

**prop:water:implanted_completion_preference** (n_questions=1)

| arm (norm ref) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D (mu_D) | +0.000 (rank 14/23; -0.125 / +0.000 / +0.250) | -0.125 (rank 10/23; -0.250 / +0.000 / +0.375) | -0.500 (rank 1/23; -0.500 / -0.125 / +0.500) | -1.687 (rank 0/23; -1.000 / -0.250 / +1.000) |
| mu_D_perp_matched (mu_D) | +0.000 (rank 14/23; -0.125 / +0.000 / +0.250) | -0.125 (rank 10/23; -0.250 / +0.000 / +0.375) | -0.500 (rank 1/23; -0.500 / -0.125 / +0.500) | -1.250 (rank 0/23; -1.000 / -0.250 / +1.000) |
| mu_Dprime_matched (mu_D) | +0.000 (rank 14/23; -0.125 / +0.000 / +0.250) | +0.000 (rank 11/23; -0.250 / +0.000 / +0.375) | -0.375 (rank 5/23; -0.500 / -0.125 / +0.500) | -1.812 (rank 0/23; -1.000 / -0.250 / +1.000) |
| mu_D_par (mu_D_par) | +0.000 (rank 15/23; -0.125 / +0.000 / +0.125) | +0.000 (rank 13/23; -0.250 / +0.000 / +0.250) | -0.250 (rank 6/23; -0.375 / -0.125 / +0.500) | -1.250 (rank 0/23; -0.750 / -0.125 / +0.875) |
| mu_D_perp_native (mu_D_perp_native) | +0.000 (rank 1/23; +0.000 / +0.000 / +0.125) | +0.000 (rank 18/23; -0.125 / +0.000 / +0.250) | -0.125 (rank 5/23; -0.250 / +0.000 / +0.375) | -0.500 (rank 1/23; -0.500 / -0.125 / +0.625) |

**factual_control** (n_questions=6)

| arm (norm ref) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D (mu_D) | -0.054 (rank 4/23; -0.127 / +0.002 / +0.167) | -0.034 (rank 8/23; -0.238 / +0.002 / +0.355) | +0.018 (rank 11/23; -0.444 / +0.021 / +0.675) | +0.319 (rank 15/23; -1.011 / +0.059 / +1.622) |
| mu_D_perp_matched (mu_D) | -0.153 (rank 0/23; -0.127 / +0.002 / +0.167) | -0.269 (rank 0/23; -0.238 / +0.002 / +0.355) | -0.508 (rank 0/23; -0.444 / +0.021 / +0.675) | -0.869 (rank 2/23; -1.011 / +0.059 / +1.622) |
| mu_Dprime_matched (mu_D) | +0.014 (rank 14/23; -0.127 / +0.002 / +0.167) | +0.098 (rank 17/23; -0.238 / +0.002 / +0.355) | +0.412 (rank 19/23; -0.444 / +0.021 / +0.675) | +1.380 (rank 21/23; -1.011 / +0.059 / +1.622) |
| mu_D_par (mu_D_par) | +0.001 (rank 12/23; -0.085 / -0.006 / +0.123) | +0.034 (rank 13/23; -0.171 / +0.018 / +0.269) | +0.342 (rank 19/23; -0.367 / +0.028 / +0.598) | +1.074 (rank 21/23; -0.770 / +0.024 / +1.308) |
| mu_D_perp_native (mu_D_perp_native) | -0.091 (rank 0/23; -0.085 / -0.029 / +0.099) | -0.144 (rank 0/23; -0.140 / +0.002 / +0.190) | -0.272 (rank 0/23; -0.248 / +0.029 / +0.317) | -0.543 (rank 0/23; -0.481 / +0.041 / +0.776) |

**domain_completion_preference** (n_questions=2)

| arm (norm ref) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| mu_D (mu_D) | +0.123 (rank 19/23; -0.221 / +0.028 / +0.212) | +0.345 (rank 22/23; -0.596 / -0.007 / +0.493) | +0.355 (rank 22/23; -1.179 / +0.009 / +0.769) | +0.121 (rank 17/23; -2.424 / -0.295 / +1.606) |
| mu_D_perp_matched (mu_D) | +0.030 (rank 13/23; -0.221 / +0.028 / +0.212) | -0.028 (rank 10/23; -0.596 / -0.007 / +0.493) | -0.022 (rank 10/23; -1.179 / +0.009 / +0.769) | -0.356 (rank 9/23; -2.424 / -0.295 / +1.606) |
| mu_Dprime_matched (mu_D) | +0.216 (rank 23/23; -0.221 / +0.028 / +0.212) | +0.413 (rank 22/23; -0.596 / -0.007 / +0.493) | +0.706 (rank 22/23; -1.179 / +0.009 / +0.769) | +0.512 (rank 22/23; -2.424 / -0.295 / +1.606) |
| mu_D_par (mu_D_par) | +0.212 (rank 23/23; -0.252 / +0.029 / +0.156) | +0.277 (rank 22/23; -0.500 / -0.002 / +0.431) | +0.505 (rank 22/23; -1.086 / -0.007 / +0.711) | +0.629 (rank 22/23; -1.986 / -0.190 / +1.360) |
| mu_D_perp_native (mu_D_perp_native) | +0.029 (rank 11/23; -0.162 / +0.031 / +0.153) | +0.062 (rank 17/23; -0.343 / +0.031 / +0.274) | -0.030 (rank 9/23; -0.529 / +0.031 / +0.398) | -0.052 (rank 12/23; -1.273 / -0.066 / +0.957) |

<!-- F2V2-NUMBERS-END -->

---

## Addendum (FOLLOWUP_BRIEF_ADDENDUM.md): priority F6, then F7, then F8; F5 kept; F4-M dropped first if time runs out

Background stated for the interpretations below: cos(mu_cake, mu_concrete) = 0.837 is a cosine and
cos^2 = 0.70 is the fraction of ||mu_cake||^2 along the concrete direction, neither is "84% shared";
the finetuned cake model's panel log-likelihood is 0.041 nat/token better than base and concrete's
0.078 worse, so the register hypothesis is mixed on this evidence; mu_Dprime matched recovers 13.3%
vs mu_D 25.1% at alpha = 2, an overlap in effects, not a partition; Minder Section 5 uses a projection
intervention with cross-entropy readouts, CDD amplifies output-logit differences, Patchscope decodes
the domain -- none establishes that a specific proposition is or is not recoverable from a fixed,
pooled mu_D.

## F6. The finetuned recipient: subtraction and two projections along the trace direction (revised Sept 12, before the run)

**Uncertainty addressed.** Whether the same direction has different effects in the two recipients,
and whether the finetuned model's expression of the implanted preference is sensitive to its own
mean-trace direction.

**What will be run** (`followup_f6.py`; permitted steer.py edit: `forward_steered(..., adapter=name)`).
Recipient: the finetuned cake model, layer 17, standard mask. Three interventions along each direction
u = v/||v||: (1) FIXED, h <- h + alpha v, alpha in {-4, -2, -1, -0.5, +0.5, +1, +2, +4} (negative = subtract,
positive = add; the finetuned recipient's own degradation guard is reported at every dose); (2) PROJ_matched (the Section-5
analogue): for each input the base forward is run first and h_base(x, pos) captured at layer 17; in the
finetuned forward's hook, at masked positions h' = h + [(h_base(x, pos).u) - (h.u)] u, so the
input-dependent component along u is set to the base model's value on the same input; (3)
PROJ_meanclamp: h' = h + [m_base - (h.u)] u with m_base the panel mean of h_base.u (one scalar per
direction, recorded). PROJ_meanclamp is also applied to the base recipient as an intervention
control; PROJ_matched on the base recipient adds exactly zero by construction (asserted on one item)
and is not tabulated. Directions: mu_D, mu_D_par, mu_D_perp_native, mu_Dprime native and matched,
and r0-r22 at ||mu_D||, ||mu_D_par|| and ||mu_D_perp_native|| (F2's sets). Ranks: each FIXED / PROJ
direction against the 23 randoms at its own norm only (mu_D and mu_Dprime_matched at ||mu_D||;
mu_D_par at ||mu_D_par||; mu_D_perp_native at ||mu_D_perp_native||; mu_Dprime_native has no rank).
Readouts: B on implanted items (original + v2 eligible; kinds separate; (proposition_id, item_kind)
summaries alongside the question-weighted ones) and factual controls, as effect_vs_recipient =
B_intervened - B_recipient with baseline_recipient stated ("finetuned:cake" or "base"); panel per-token
log-likelihood (drop_vs_recipient = ll_recipient - ll_intervened, cap 1.0) and KL(p_base ||
p_intervened). Gates: G1-adapter, G2/G2b on the adapter path, local-increment check on every
projection forward, FIXED alpha = 0 == B_ft, and the adapter state peft reports at every forward equals
the requested state (recorded in f6_meta.json).

**Outcomes -> interpretation** (written 2026-09-12, before the run; observed row marked after):

| outcome | interpretation |
|---|---|
| FIXED (alpha < 0) or PROJ_matched along mu_D moves B_ft toward base on implanted items by an amount outside the same-norm random range and the cross-organism direction's, with controls and panel likelihood not comparably disrupted | the finetuned model's expression of the implanted preference is sensitive to this direction in a way random and other-organism directions do not reproduce; "involved in expression", not "carries the fact" |
| moves toward base but same-norm random / concrete directions do the same | broad disruption of the finetuned model, not direction-specific sensitivity |
| PROJ_matched moves B_ft while subtraction does not (or vice versa) | the sensitivity is to the input-dependent component along u (or to the constant offset); both reported |
| PROJ_matched moves B_ft | the sensitivity is to the input-dependent component along u on the same input (PROJ_matched removes exactly that component and nothing else) |
| PROJ_meanclamp moves B_ft where PROJ_matched does not (or by a different amount) | PROJ_meanclamp also removes the recipient's own variation along u and its magnitude varies by position, so its effect is not attributable to the input-dependent component alone; the two projections are interpreted separately |
| PROJ_meanclamp on the base recipient moves B_base | the clamp itself perturbs the base model on these items; PROJ_meanclamp rows on the finetuned recipient are read against that control |
| no movement under any intervention | the tested interventions along this direction do not affect the finetuned model's implanted preference at this layer / positions; does not establish that the belief is expressed orthogonally to mu_D |
| base+mu_D near-zero (known) alongside ft-mu_D nonzero | the direction's effect depends on the recipient; observed asymmetry, mechanism open |
| adding mu_D to the finetuned model (FIXED alpha > 0) raises implanted preference above B_ft, ranked above the same-norm randoms, within the degradation guard | the finetuned model's implanted preference is dose-sensitive to its own trace direction in the positive direction as well; reported with the guard values |
| adding mu_D does not raise implanted preference above B_ft, or randoms at matched norm do the same | no direction-specific positive dose effect detected in the finetuned recipient |

<!-- F6-NUMBERS-START -->
**Run** 2026-09-12 07:31:35; v2 eligible items included: True. m_base per direction in f6_mbase.json (mu_D: +17.697, mu_D_par: +4.144, mu_D_perp_native: +26.018).

**finetuned recipient / temp_implanted** (n_items=9, n_questions=7; B_ft +0.686, B_base -5.362): effect_vs_recipient = B_intervened - B_ft [CI] label; FIXED alpha < 0 subtracts, alpha > 0 adds; on implanted items a negative sign is movement toward base, a positive sign is above B_ft:

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | +0.053 [-0.018, +0.116] near_zero | +0.119 [+0.030, +0.212] near_zero | +0.323 [+0.246, +0.403] nonzero | +0.769 [+0.614, +0.917] nonzero | -0.070 [-0.110, -0.024] near_zero | -0.119 [-0.168, -0.065] near_zero | -0.192 [-0.273, -0.103] near_zero | -0.250 [-0.394, -0.074] nonzero | -0.154 [-0.209, -0.095] near_zero | -0.010 [-0.110, +0.068] near_zero |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero | +0.023 [-0.034, +0.078] near_zero | +0.108 [+0.052, +0.166] near_zero | +0.276 [+0.201, +0.365] nonzero | +0.612 [+0.467, +0.778] nonzero | -0.053 [-0.106, -0.009] near_zero | -0.127 [-0.161, -0.080] near_zero | -0.208 [-0.288, -0.119] nonzero | -0.273 [-0.424, -0.096] nonzero | -0.122 [-0.176, -0.062] near_zero | -0.008 [-0.108, +0.096] near_zero |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero | +0.002 [-0.050, +0.051] near_zero | +0.031 [-0.016, +0.079] near_zero | +0.088 [+0.017, +0.159] near_zero | +0.156 [+0.047, +0.277] near_zero | -0.019 [-0.065, +0.035] near_zero | -0.007 [-0.061, +0.044] near_zero | +0.033 [-0.007, +0.073] near_zero | +0.078 [+0.032, +0.124] near_zero | -0.007 [-0.011, -0.003] near_zero | +0.063 [+0.002, +0.113] near_zero |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero | +0.128 [+0.074, +0.180] near_zero | +0.295 [+0.228, +0.375] nonzero | +0.653 [+0.511, +0.807] nonzero | +1.306 [+1.032, +1.616] nonzero | -0.110 [-0.162, -0.027] near_zero | -0.191 [-0.263, -0.117] near_zero | -0.287 [-0.450, -0.107] nonzero | -0.575 [-0.838, -0.271] nonzero | -0.122 [-0.176, -0.062] near_zero | -0.008 [-0.108, +0.096] near_zero |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero | +0.106 [+0.052, +0.165] near_zero | +0.102 [+0.011, +0.177] near_zero | +0.321 [+0.230, +0.436] nonzero | +0.684 [+0.571, +0.816] nonzero | -0.060 [-0.116, +0.014] near_zero | -0.093 [-0.147, -0.036] near_zero | -0.201 [-0.269, -0.117] nonzero | -0.321 [-0.505, -0.114] nonzero | -0.142 [-0.185, -0.091] near_zero | -0.008 [-0.108, +0.096] near_zero |
| r0@mu_D | +0.000 [+0.000, +0.000] near_zero | -0.051 [-0.088, -0.012] near_zero | -0.117 [-0.187, -0.051] near_zero | -0.213 [-0.306, -0.124] nonzero | -0.395 [-0.490, -0.283] nonzero | +0.005 [-0.044, +0.053] near_zero | +0.109 [+0.054, +0.171] near_zero | +0.160 [+0.062, +0.268] near_zero | +0.266 [+0.135, +0.413] nonzero | +0.015 [-0.041, +0.072] near_zero | +0.017 [-0.051, +0.095] near_zero |
| r1@mu_D | +0.000 [+0.000, +0.000] near_zero | -0.035 [-0.093, +0.029] near_zero | -0.042 [-0.099, +0.023] near_zero | -0.049 [-0.094, -0.010] near_zero | -0.097 [-0.235, +0.045] near_zero | -0.020 [-0.054, -0.001] near_zero | -0.019 [-0.065, +0.036] near_zero | +0.055 [-0.011, +0.134] near_zero | +0.135 [+0.015, +0.251] near_zero | -0.029 [-0.087, +0.034] near_zero | -0.008 [-0.030, +0.004] near_zero |
| r2@mu_D | +0.000 [+0.000, +0.000] near_zero | +0.021 [-0.035, +0.075] near_zero | +0.111 [+0.054, +0.170] near_zero | +0.230 [+0.153, +0.311] nonzero | +0.576 [+0.457, +0.731] nonzero | -0.077 [-0.140, -0.023] near_zero | -0.127 [-0.137, -0.115] near_zero | -0.208 [-0.248, -0.163] nonzero | -0.378 [-0.456, -0.308] nonzero | -0.053 [-0.093, -0.013] near_zero | -0.031 [-0.078, +0.016] near_zero |

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 22/23 @mu_D (-0.059 / -0.013 / +0.057) | rank 22/23 @mu_D (-0.159 / -0.006 / +0.123) | rank 23/23 @mu_D (-0.225 / +0.039 / +0.258) | rank 23/23 @mu_D (-0.399 / -0.024 / +0.576) | rank 4/23 @mu_D (-0.105 / -0.016 / +0.065) | rank 2/23 @mu_D (-0.161 / -0.031 / +0.109) | rank 2/23 @mu_D (-0.302 / -0.086 / +0.197) | rank 6/23 @mu_D (-0.513 / -0.142 / +0.434) | rank 0/23 @mu_D (-0.053 / -0.010 / +0.027) | rank 17/23 @mu_D (-0.100 / -0.024 / +0.017) |
| mu_D_par | rank 23/23 @mu_D_par (+0.000 / +0.000 / +0.000) | rank 19/23 @mu_D_par (-0.093 / -0.017 / +0.049) | rank 23/23 @mu_D_par (-0.133 / +0.009 / +0.095) | rank 23/23 @mu_D_par (-0.178 / +0.029 / +0.258) | rank 23/23 @mu_D_par (-0.342 / -0.004 / +0.508) | rank 3/23 @mu_D_par (-0.065 / -0.017 / +0.034) | rank 1/23 @mu_D_par (-0.139 / -0.028 / +0.125) | rank 1/23 @mu_D_par (-0.278 / -0.029 / +0.154) | rank 3/23 @mu_D_par (-0.471 / -0.115 / +0.329) | rank 0/23 @mu_D_par (-0.053 / -0.013 / +0.029) | rank 17/23 @mu_D_par (-0.100 / -0.026 / +0.017) |
| mu_D_perp_native | rank 23/23 @mu_D_perp_native (+0.000 / +0.000 / +0.000) | rank 16/23 @mu_D_perp_native (-0.056 / -0.009 / +0.036) | rank 16/23 @mu_D_perp_native (-0.094 / +0.006 / +0.077) | rank 21/23 @mu_D_perp_native (-0.119 / +0.012 / +0.161) | rank 19/23 @mu_D_perp_native (-0.250 / +0.036 / +0.298) | rank 6/23 @mu_D_perp_native (-0.074 / -0.012 / +0.046) | rank 12/23 @mu_D_perp_native (-0.103 / -0.009 / +0.054) | rank 17/23 @mu_D_perp_native (-0.199 / -0.041 / +0.131) | rank 15/23 @mu_D_perp_native (-0.319 / -0.077 / +0.224) | rank 14/23 @mu_D_perp_native (-0.053 / -0.015 / +0.015) | rank 23/23 @mu_D_perp_native (-0.100 / -0.026 / +0.017) |
| mu_Dprime_matched | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 23/23 @mu_D (-0.059 / -0.013 / +0.057) | rank 21/23 @mu_D (-0.159 / -0.006 / +0.123) | rank 23/23 @mu_D (-0.225 / +0.039 / +0.258) | rank 23/23 @mu_D (-0.399 / -0.024 / +0.576) | rank 4/23 @mu_D (-0.105 / -0.016 / +0.065) | rank 3/23 @mu_D (-0.161 / -0.031 / +0.109) | rank 2/23 @mu_D (-0.302 / -0.086 / +0.197) | rank 3/23 @mu_D (-0.513 / -0.142 / +0.434) | rank 0/23 @mu_D (-0.053 / -0.010 / +0.027) | rank 18/23 @mu_D (-0.100 / -0.024 / +0.017) |

**finetuned recipient / implanted_factual_all** (n_items=12, n_questions=10; B_ft +0.907, B_base -5.484): effect_vs_recipient = B_intervened - B_ft [CI] label; FIXED alpha < 0 subtracts, alpha > 0 adds; on implanted items a negative sign is movement toward base, a positive sign is above B_ft:

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | +0.074 [-0.011, +0.164] near_zero | +0.160 [+0.004, +0.327] near_zero | +0.384 [+0.131, +0.677] nonzero | +0.885 [+0.482, +1.475] nonzero | -0.102 [-0.149, -0.053] near_zero | -0.154 [-0.265, -0.043] near_zero | -0.277 [-0.455, -0.118] nonzero | -0.445 [-0.835, -0.095] nonzero | -0.200 [-0.319, -0.095] nonzero | -0.058 [-0.136, +0.018] near_zero |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero | +0.014 [-0.073, +0.094] near_zero | +0.066 [-0.067, +0.182] near_zero | +0.265 [-0.015, +0.574] inconclusive | +0.679 [+0.205, +1.332] nonzero | -0.079 [-0.130, -0.031] near_zero | -0.093 [-0.188, +0.014] near_zero | -0.179 [-0.349, -0.010] near_zero | -0.246 [-0.546, +0.077] inconclusive | -0.113 [-0.228, -0.011] near_zero | -0.011 [-0.114, +0.095] near_zero |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero | +0.030 [-0.024, +0.096] near_zero | +0.064 [+0.003, +0.139] near_zero | +0.151 [+0.053, +0.273] near_zero | +0.325 [+0.133, +0.570] nonzero | -0.049 [-0.111, +0.009] near_zero | -0.077 [-0.167, -0.000] near_zero | -0.087 [-0.232, +0.026] near_zero | -0.189 [-0.543, +0.069] inconclusive | -0.107 [-0.263, -0.005] near_zero | -0.034 [-0.149, +0.064] near_zero |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero | +0.114 [-0.049, +0.264] near_zero | +0.292 [+0.023, +0.617] nonzero | +0.735 [+0.217, +1.454] nonzero | +1.493 [+0.537, +2.647] nonzero | -0.095 [-0.217, +0.026] near_zero | -0.170 [-0.339, +0.013] near_zero | -0.252 [-0.586, +0.077] inconclusive | -0.517 [-1.136, +0.036] inconclusive | -0.113 [-0.228, -0.011] near_zero | -0.011 [-0.114, +0.095] near_zero |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero | +0.084 [-0.018, +0.183] near_zero | +0.082 [-0.078, +0.240] near_zero | +0.308 [-0.005, +0.663] inconclusive | +0.792 [+0.234, +1.576] nonzero | -0.057 [-0.126, +0.010] near_zero | -0.084 [-0.208, +0.049] near_zero | -0.172 [-0.379, +0.021] near_zero | -0.283 [-0.614, +0.072] inconclusive | -0.127 [-0.240, -0.024] near_zero | -0.011 [-0.114, +0.095] near_zero |
| r0@mu_D | +0.000 [+0.000, +0.000] near_zero | -0.063 [-0.095, -0.030] near_zero | -0.104 [-0.162, -0.053] near_zero | -0.182 [-0.262, -0.099] near_zero | -0.424 [-0.538, -0.314] nonzero | +0.009 [-0.041, +0.059] near_zero | +0.067 [-0.027, +0.143] near_zero | +0.093 [-0.044, +0.213] near_zero | +0.113 [-0.164, +0.328] near_zero | +0.011 [-0.046, +0.069] near_zero | +0.030 [-0.030, +0.088] near_zero |
| r1@mu_D | +0.000 [+0.000, +0.000] near_zero | -0.021 [-0.067, +0.024] near_zero | -0.035 [-0.086, +0.020] near_zero | -0.041 [-0.165, +0.066] near_zero | -0.083 [-0.323, +0.129] near_zero | -0.021 [-0.068, +0.020] near_zero | -0.020 [-0.092, +0.058] near_zero | +0.016 [-0.133, +0.157] near_zero | -0.009 [-0.275, +0.200] near_zero | -0.018 [-0.061, +0.027] near_zero | +0.001 [-0.032, +0.036] near_zero |
| r2@mu_D | +0.000 [+0.000, +0.000] near_zero | -0.011 [-0.076, +0.049] near_zero | +0.097 [-0.037, +0.216] near_zero | +0.211 [+0.001, +0.386] nonzero | +0.444 [-0.022, +0.761] inconclusive | -0.072 [-0.123, -0.028] near_zero | -0.114 [-0.152, -0.060] near_zero | -0.166 [-0.273, -0.040] near_zero | -0.277 [-0.479, -0.051] nonzero | -0.047 [-0.078, -0.018] near_zero | -0.028 [-0.074, +0.013] near_zero |

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 23/23 @mu_D (-0.072 / -0.021 / +0.027) | rank 23/23 @mu_D (-0.145 / -0.010 / +0.145) | rank 23/23 @mu_D (-0.209 / +0.047 / +0.333) | rank 23/23 @mu_D (-0.424 / -0.020 / +0.696) | rank 1/23 @mu_D (-0.114 / -0.021 / +0.057) | rank 1/23 @mu_D (-0.171 / -0.026 / +0.089) | rank 1/23 @mu_D (-0.298 / -0.042 / +0.174) | rank 2/23 @mu_D (-0.522 / -0.047 / +0.395) | rank 0/23 @mu_D (-0.047 / -0.013 / +0.019) | rank 0/23 @mu_D (-0.056 / -0.018 / +0.069) |
| mu_D_par | rank 23/23 @mu_D_par (+0.000 / +0.000 / +0.000) | rank 17/23 @mu_D_par (-0.082 / -0.018 / +0.041) | rank 21/23 @mu_D_par (-0.119 / -0.011 / +0.100) | rank 22/23 @mu_D_par (-0.167 / +0.006 / +0.326) | rank 23/23 @mu_D_par (-0.347 / +0.007 / +0.633) | rank 0/23 @mu_D_par (-0.072 / -0.014 / +0.041) | rank 2/23 @mu_D_par (-0.162 / -0.029 / +0.105) | rank 3/23 @mu_D_par (-0.292 / -0.032 / +0.123) | rank 3/23 @mu_D_par (-0.477 / -0.033 / +0.295) | rank 0/23 @mu_D_par (-0.034 / -0.012 / +0.033) | rank 12/23 @mu_D_par (-0.056 / -0.016 / +0.069) |
| mu_D_perp_native | rank 23/23 @mu_D_perp_native (+0.000 / +0.000 / +0.000) | rank 22/23 @mu_D_perp_native (-0.065 / -0.011 / +0.044) | rank 22/23 @mu_D_perp_native (-0.075 / -0.011 / +0.064) | rank 22/23 @mu_D_perp_native (-0.120 / -0.022 / +0.185) | rank 22/23 @mu_D_perp_native (-0.212 / +0.015 / +0.368) | rank 2/23 @mu_D_perp_native (-0.070 / -0.009 / +0.043) | rank 1/23 @mu_D_perp_native (-0.086 / -0.014 / +0.051) | rank 5/23 @mu_D_perp_native (-0.187 / -0.036 / +0.120) | rank 3/23 @mu_D_perp_native (-0.327 / -0.064 / +0.187) | rank 0/23 @mu_D_perp_native (-0.041 / -0.018 / +0.012) | rank 4/23 @mu_D_perp_native (-0.056 / -0.018 / +0.069) |
| mu_Dprime_matched | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 23/23 @mu_D (-0.072 / -0.021 / +0.027) | rank 21/23 @mu_D (-0.145 / -0.010 / +0.145) | rank 22/23 @mu_D (-0.209 / +0.047 / +0.333) | rank 23/23 @mu_D (-0.424 / -0.020 / +0.696) | rank 3/23 @mu_D (-0.114 / -0.021 / +0.057) | rank 4/23 @mu_D (-0.171 / -0.026 / +0.089) | rank 2/23 @mu_D (-0.298 / -0.042 / +0.174) | rank 2/23 @mu_D (-0.522 / -0.047 / +0.395) | rank 0/23 @mu_D (-0.047 / -0.013 / +0.019) | rank 15/23 @mu_D (-0.056 / -0.018 / +0.069) |

**finetuned recipient / implanted_completion_preference** (n_items=4, n_questions=4; B_ft +4.563, B_base -0.375): effect_vs_recipient = B_intervened - B_ft [CI] label; FIXED alpha < 0 subtracts, alpha > 0 adds; on implanted items a negative sign is movement toward base, a positive sign is above B_ft:

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | +0.078 [-0.063, +0.219] near_zero | +0.156 [-0.125, +0.406] near_zero | +0.172 [-0.469, +0.641] inconclusive | +0.297 [-0.563, +1.031] inconclusive | -0.000 [-0.125, +0.156] near_zero | -0.109 [-0.312, +0.141] near_zero | -0.328 [-0.719, +0.078] inconclusive | -0.688 [-1.344, -0.063] nonzero | -0.141 [-0.344, +0.109] near_zero | -0.156 [-0.375, +0.062] near_zero |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero | +0.125 [-0.031, +0.250] near_zero | +0.141 [-0.203, +0.406] near_zero | +0.297 [-0.500, +0.797] inconclusive | +0.406 [-1.063, +1.406] inconclusive | -0.031 [-0.125, +0.156] near_zero | -0.047 [-0.203, +0.156] near_zero | -0.172 [-0.469, +0.172] near_zero | -0.578 [-1.094, -0.063] nonzero | -0.063 [-0.250, +0.219] near_zero | -0.109 [-0.266, +0.062] near_zero |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero | +0.125 [+0.031, +0.219] near_zero | +0.109 [+0.031, +0.203] near_zero | +0.094 [-0.062, +0.250] near_zero | +0.062 [-0.187, +0.312] near_zero | +0.031 [-0.000, +0.094] near_zero | -0.016 [-0.125, +0.109] near_zero | -0.031 [-0.125, +0.063] near_zero | -0.297 [-0.500, -0.094] nonzero | -0.016 [-0.094, +0.047] near_zero | -0.047 [-0.141, +0.000] near_zero |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero | +0.109 [-0.281, +0.375] near_zero | +0.328 [-0.578, +0.906] inconclusive | +0.453 [-1.125, +1.516] inconclusive | -0.281 [-2.078, +0.828] inconclusive | -0.094 [-0.250, +0.125] near_zero | -0.297 [-0.594, +0.047] inconclusive | -0.641 [-1.094, -0.188] nonzero | -1.500 [-2.125, -0.781] nonzero | -0.063 [-0.250, +0.219] near_zero | -0.141 [-0.266, -0.031] near_zero |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero | +0.094 [-0.125, +0.250] near_zero | +0.141 [-0.375, +0.453] near_zero | +0.281 [-0.750, +0.906] inconclusive | +0.469 [-1.203, +1.547] inconclusive | -0.047 [-0.125, +0.062] near_zero | -0.125 [-0.312, +0.125] near_zero | -0.250 [-0.563, +0.062] inconclusive | -0.703 [-1.156, -0.250] nonzero | -0.063 [-0.250, +0.219] near_zero | -0.109 [-0.266, +0.062] near_zero |
| r0@mu_D | +0.000 [+0.000, +0.000] near_zero | -0.000 [-0.125, +0.156] near_zero | -0.094 [-0.375, +0.156] near_zero | -0.266 [-0.594, +0.078] inconclusive | -0.563 [-1.125, +0.125] inconclusive | +0.109 [-0.000, +0.219] near_zero | +0.141 [-0.000, +0.281] near_zero | +0.234 [+0.047, +0.500] nonzero | +0.438 [+0.000, +0.969] nonzero | +0.016 [-0.000, +0.047] near_zero | +0.031 [-0.094, +0.187] near_zero |
| r1@mu_D | +0.000 [+0.000, +0.000] near_zero | -0.062 [-0.125, +0.063] near_zero | -0.125 [-0.250, +0.031] near_zero | -0.250 [-0.438, -0.000] nonzero | -0.609 [-1.141, -0.000] nonzero | +0.094 [+0.031, +0.125] near_zero | +0.094 [-0.000, +0.188] near_zero | +0.172 [-0.031, +0.359] near_zero | +0.016 [-0.328, +0.219] near_zero | +0.047 [-0.000, +0.094] near_zero | +0.063 [-0.000, +0.125] near_zero |
| r2@mu_D | +0.000 [+0.000, +0.000] near_zero | +0.062 [-0.000, +0.125] near_zero | +0.219 [+0.156, +0.250] nonzero | +0.359 [+0.281, +0.453] nonzero | +0.625 [+0.531, +0.719] nonzero | -0.094 [-0.188, +0.000] near_zero | -0.141 [-0.219, -0.047] near_zero | -0.344 [-0.469, -0.187] nonzero | -0.672 [-0.828, -0.469] nonzero | -0.016 [-0.094, +0.047] near_zero | +0.016 [-0.078, +0.094] near_zero |

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 16/23 @mu_D (-0.062 / +0.031 / +0.266) | rank 20/23 @mu_D (-0.156 / +0.047 / +0.437) | rank 15/23 @mu_D (-0.297 / +0.062 / +0.844) | rank 17/23 @mu_D (-0.797 / -0.141 / +1.578) | rank 8/23 @mu_D (-0.156 / +0.016 / +0.141) | rank 5/23 @mu_D (-0.359 / -0.016 / +0.188) | rank 3/23 @mu_D (-0.781 / -0.063 / +0.297) | rank 3/23 @mu_D (-1.563 / -0.297 / +0.469) | rank 0/23 @mu_D (-0.016 / +0.047 / +0.094) | rank 0/23 @mu_D (-0.078 / +0.031 / +0.125) |
| mu_D_par | rank 23/23 @mu_D_par (+0.000 / +0.000 / +0.000) | rank 20/23 @mu_D_par (-0.063 / +0.031 / +0.203) | rank 16/23 @mu_D_par (-0.078 / +0.031 / +0.391) | rank 20/23 @mu_D_par (-0.219 / +0.016 / +0.703) | rank 19/23 @mu_D_par (-0.625 / -0.109 / +1.375) | rank 2/23 @mu_D_par (-0.094 / +0.031 / +0.125) | rank 6/23 @mu_D_par (-0.297 / +0.016 / +0.187) | rank 7/23 @mu_D_par (-0.625 / -0.062 / +0.250) | rank 2/23 @mu_D_par (-1.250 / -0.219 / +0.437) | rank 0/23 @mu_D_par (-0.016 / +0.047 / +0.094) | rank 0/23 @mu_D_par (-0.078 / +0.031 / +0.125) |
| mu_D_perp_native | rank 23/23 @mu_D_perp_native (+0.000 / +0.000 / +0.000) | rank 23/23 @mu_D_perp_native (-0.031 / +0.031 / +0.125) | rank 17/23 @mu_D_perp_native (-0.063 / +0.047 / +0.234) | rank 13/23 @mu_D_perp_native (-0.172 / +0.062 / +0.500) | rank 13/23 @mu_D_perp_native (-0.375 / +0.016 / +0.938) | rank 10/23 @mu_D_perp_native (-0.063 / +0.031 / +0.094) | rank 8/23 @mu_D_perp_native (-0.156 / +0.016 / +0.141) | rank 10/23 @mu_D_perp_native (-0.359 / -0.016 / +0.187) | rank 4/23 @mu_D_perp_native (-0.828 / -0.094 / +0.297) | rank 2/23 @mu_D_perp_native (-0.016 / +0.047 / +0.094) | rank 1/23 @mu_D_perp_native (-0.078 / +0.031 / +0.125) |
| mu_Dprime_matched | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 16/23 @mu_D (-0.062 / +0.031 / +0.266) | rank 16/23 @mu_D (-0.156 / +0.047 / +0.437) | rank 18/23 @mu_D (-0.297 / +0.062 / +0.844) | rank 19/23 @mu_D (-0.797 / -0.141 / +1.578) | rank 5/23 @mu_D (-0.156 / +0.016 / +0.141) | rank 3/23 @mu_D (-0.359 / -0.016 / +0.188) | rank 5/23 @mu_D (-0.781 / -0.063 / +0.297) | rank 2/23 @mu_D (-1.563 / -0.297 / +0.469) | rank 0/23 @mu_D (-0.016 / +0.047 / +0.094) | rank 0/23 @mu_D (-0.078 / +0.031 / +0.125) |

**finetuned recipient / factual_control** (n_items=8, n_questions=6; B_ft +8.220, B_base +9.850): effect_vs_recipient = B_intervened - B_ft [CI] label; FIXED alpha < 0 subtracts, alpha > 0 adds; on implanted items a negative sign is movement toward base, a positive sign is above B_ft:

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | +0.030 [-0.021, +0.081] near_zero | +0.009 [-0.135, +0.145] near_zero | -0.027 [-0.312, +0.258] near_zero | -0.351 [-1.125, +0.454] inconclusive | -0.016 [-0.073, +0.041] near_zero | -0.016 [-0.110, +0.083] near_zero | +0.056 [-0.116, +0.274] near_zero | +0.220 [-0.224, +0.643] inconclusive | +0.061 [-0.031, +0.151] near_zero | -0.083 [-0.167, +0.000] near_zero |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero | +0.083 [+0.010, +0.146] near_zero | +0.172 [+0.073, +0.245] near_zero | +0.360 [+0.187, +0.518] nonzero | +0.586 [+0.114, +1.122] nonzero | -0.059 [-0.104, -0.015] near_zero | -0.153 [-0.260, -0.049] near_zero | -0.231 [-0.427, -0.014] nonzero | -0.318 [-0.808, +0.105] inconclusive | -0.091 [-0.188, +0.000] near_zero | -0.174 [-0.289, -0.070] near_zero |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero | -0.054 [-0.104, -0.003] near_zero | -0.117 [-0.188, -0.049] near_zero | -0.279 [-0.469, -0.115] nonzero | -0.551 [-0.990, -0.213] nonzero | +0.046 [-0.000, +0.129] near_zero | +0.173 [+0.109, +0.260] near_zero | +0.292 [+0.151, +0.422] nonzero | +0.632 [+0.365, +0.833] nonzero | +0.270 [+0.182, +0.375] nonzero | +0.198 [+0.115, +0.286] near_zero |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero | +0.224 [+0.078, +0.334] nonzero | +0.376 [+0.125, +0.571] nonzero | +0.615 [+0.063, +1.187] nonzero | -0.399 [-1.859, +1.053] inconclusive | -0.106 [-0.226, +0.021] near_zero | -0.222 [-0.485, +0.054] inconclusive | -0.304 [-0.875, +0.167] inconclusive | -0.560 [-1.990, +0.563] inconclusive | -0.091 [-0.188, +0.000] near_zero | -0.174 [-0.289, -0.070] near_zero |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero | +0.099 [+0.036, +0.167] near_zero | +0.229 [+0.094, +0.354] nonzero | +0.457 [+0.208, +0.693] nonzero | +0.617 [-0.021, +1.309] inconclusive | -0.059 [-0.125, +0.007] near_zero | -0.153 [-0.262, -0.049] near_zero | -0.237 [-0.448, -0.039] nonzero | -0.335 [-0.939, +0.155] inconclusive | -0.125 [-0.229, -0.021] near_zero | -0.174 [-0.289, -0.070] near_zero |
| r0@mu_D | +0.000 [+0.000, +0.000] near_zero | -0.001 [-0.074, +0.072] near_zero | -0.039 [-0.164, +0.086] near_zero | -0.125 [-0.448, +0.136] near_zero | -0.379 [-1.171, +0.277] inconclusive | -0.015 [-0.083, +0.047] near_zero | +0.013 [-0.109, +0.141] near_zero | +0.020 [-0.208, +0.292] near_zero | -0.056 [-0.525, +0.553] inconclusive | +0.002 [-0.030, +0.031] near_zero | -0.033 [-0.115, +0.057] near_zero |
| r1@mu_D | +0.000 [+0.000, +0.000] near_zero | +0.021 [-0.021, +0.073] near_zero | -0.000 [-0.073, +0.083] near_zero | -0.060 [-0.169, +0.073] near_zero | -0.258 [-0.497, -0.018] nonzero | -0.020 [-0.062, +0.003] near_zero | -0.017 [-0.135, +0.102] near_zero | -0.209 [-0.500, +0.061] inconclusive | -0.586 [-1.263, +0.018] inconclusive | +0.045 [-0.010, +0.093] near_zero | +0.049 [-0.069, +0.167] near_zero |
| r2@mu_D | +0.000 [+0.000, +0.000] near_zero | +0.026 [-0.062, +0.120] near_zero | +0.024 [-0.111, +0.183] near_zero | +0.016 [-0.365, +0.433] near_zero | -0.130 [-0.938, +0.672] inconclusive | -0.010 [-0.104, +0.073] near_zero | -0.036 [-0.214, +0.125] near_zero | -0.111 [-0.371, +0.139] near_zero | -0.397 [-0.813, +0.082] inconclusive | +0.019 [+0.004, +0.038] near_zero | +0.045 [-0.073, +0.177] near_zero |

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 17/23 @mu_D (-0.051 / +0.014 / +0.133) | rank 12/23 @mu_D (-0.127 / +0.003 / +0.261) | rank 12/23 @mu_D (-0.260 / -0.055 / +0.530) | rank 9/23 @mu_D (-0.716 / -0.258 / +0.783) | rank 9/23 @mu_D (-0.161 / -0.007 / +0.046) | rank 12/23 @mu_D (-0.273 / -0.017 / +0.106) | rank 18/23 @mu_D (-0.605 / -0.132 / +0.117) | rank 23/23 @mu_D (-1.300 / -0.397 / +0.145) | rank 23/23 @mu_D (-0.015 / +0.011 / +0.048) | rank 0/23 @mu_D (-0.054 / +0.019 / +0.151) |
| mu_D_par | rank 23/23 @mu_D_par (+0.000 / +0.000 / +0.000) | rank 21/23 @mu_D_par (-0.056 / +0.016 / +0.147) | rank 19/23 @mu_D_par (-0.104 / +0.014 / +0.258) | rank 21/23 @mu_D_par (-0.253 / -0.016 / +0.449) | rank 20/23 @mu_D_par (-0.523 / -0.161 / +0.735) | rank 4/23 @mu_D_par (-0.137 / -0.006 / +0.048) | rank 3/23 @mu_D_par (-0.237 / -0.021 / +0.067) | rank 4/23 @mu_D_par (-0.480 / -0.093 / +0.123) | rank 10/23 @mu_D_par (-1.100 / -0.297 / +0.128) | rank 0/23 @mu_D_par (-0.028 / +0.018 / +0.066) | rank 0/23 @mu_D_par (-0.054 / +0.014 / +0.151) |
| mu_D_perp_native | rank 23/23 @mu_D_perp_native (+0.000 / +0.000 / +0.000) | rank 0/23 @mu_D_perp_native (-0.049 / +0.007 / +0.080) | rank 0/23 @mu_D_perp_native (-0.077 / +0.025 / +0.159) | rank 0/23 @mu_D_perp_native (-0.107 / -0.012 / +0.267) | rank 0/23 @mu_D_perp_native (-0.299 / -0.074 / +0.546) | rank 22/23 @mu_D_perp_native (-0.091 / -0.002 / +0.074) | rank 23/23 @mu_D_perp_native (-0.146 / -0.012 / +0.062) | rank 23/23 @mu_D_perp_native (-0.325 / -0.024 / +0.100) | rank 23/23 @mu_D_perp_native (-0.660 / -0.156 / +0.138) | rank 23/23 @mu_D_perp_native (-0.017 / +0.018 / +0.066) | rank 23/23 @mu_D_perp_native (-0.054 / +0.025 / +0.151) |
| mu_Dprime_matched | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 21/23 @mu_D (-0.051 / +0.014 / +0.133) | rank 20/23 @mu_D (-0.127 / +0.003 / +0.261) | rank 22/23 @mu_D (-0.260 / -0.055 / +0.530) | rank 20/23 @mu_D (-0.716 / -0.258 / +0.783) | rank 3/23 @mu_D (-0.161 / -0.007 / +0.046) | rank 4/23 @mu_D (-0.273 / -0.017 / +0.106) | rank 5/23 @mu_D (-0.605 / -0.132 / +0.117) | rank 13/23 @mu_D (-1.300 / -0.397 / +0.145) | rank 0/23 @mu_D (-0.015 / +0.011 / +0.048) | rank 0/23 @mu_D (-0.054 / +0.019 / +0.151) |

**finetuned recipient / domain_completion_preference** (n_items=2, n_questions=2; B_ft +8.352, B_base +8.303): effect_vs_recipient = B_intervened - B_ft [CI] label; FIXED alpha < 0 subtracts, alpha > 0 adds; on implanted items a negative sign is movement toward base, a positive sign is above B_ft:

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | -0.035 [-0.070, +0.000] near_zero | -0.031 [-0.186, +0.125] near_zero | -0.335 [-0.857, +0.188] inconclusive | -1.105 [-2.022, -0.187] nonzero | -0.001 [-0.063, +0.060] near_zero | -0.029 [-0.188, +0.129] near_zero | -0.103 [-0.250, +0.043] near_zero | -0.392 [-0.625, -0.159] nonzero | +0.070 [+0.000, +0.141] near_zero | -0.061 [-0.123, -0.000] near_zero |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero | -0.035 [-0.070, -0.000] near_zero | -0.094 [-0.125, -0.063] near_zero | -0.061 [-0.063, -0.060] near_zero | -0.512 [-0.712, -0.313] nonzero | -0.068 [-0.137, -0.000] near_zero | -0.150 [-0.237, -0.063] near_zero | -0.138 [-0.338, +0.063] near_zero | -0.117 [-0.360, +0.125] near_zero | -0.023 [-0.108, +0.062] near_zero | -0.028 [-0.119, +0.062] near_zero |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero | -0.002 [-0.067, +0.062] near_zero | +0.032 [-0.061, +0.125] near_zero | -0.121 [-0.367, +0.125] near_zero | -0.210 [-0.795, +0.375] inconclusive | +0.061 [-0.062, +0.184] near_zero | +0.028 [-0.125, +0.181] near_zero | +0.064 [-0.313, +0.440] near_zero | -0.135 [-0.813, +0.542] inconclusive | +0.062 [-0.125, +0.249] near_zero | -0.066 [-0.125, -0.008] near_zero |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero | -0.004 [-0.008, -0.000] near_zero | -0.118 [-0.174, -0.062] near_zero | -0.613 [-0.850, -0.375] nonzero | -1.582 [-1.625, -1.540] nonzero | -0.124 [-0.185, -0.063] near_zero | -0.097 [-0.257, +0.062] near_zero | -0.157 [-0.314, -0.000] near_zero | -0.299 [-0.660, +0.062] inconclusive | -0.023 [-0.108, +0.062] near_zero | -0.028 [-0.119, +0.062] near_zero |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero | -0.064 [-0.128, -0.000] near_zero | +0.029 [+0.000, +0.058] near_zero | -0.184 [-0.306, -0.063] near_zero | -0.672 [-0.906, -0.438] nonzero | -0.093 [-0.125, -0.062] near_zero | -0.052 [-0.063, -0.042] near_zero | -0.099 [-0.197, +0.000] near_zero | -0.059 [-0.305, +0.187] near_zero | -0.023 [-0.108, +0.062] near_zero | -0.028 [-0.119, +0.062] near_zero |
| r0@mu_D | +0.000 [+0.000, +0.000] near_zero | +0.152 [-0.062, +0.366] near_zero | +0.186 [-0.188, +0.560] inconclusive | +0.429 [-0.188, +1.045] inconclusive | +1.025 [-0.188, +2.238] inconclusive | -0.092 [-0.184, +0.000] near_zero | -0.214 [-0.491, +0.063] inconclusive | -0.399 [-0.985, +0.188] inconclusive | -0.847 [-1.820, +0.125] inconclusive | -0.029 [-0.059, -0.000] near_zero | +0.119 [-0.125, +0.363] near_zero |
| r1@mu_D | +0.000 [+0.000, +0.000] near_zero | -0.250 [-0.438, -0.063] nonzero | -0.381 [-0.575, -0.187] nonzero | -0.880 [-1.447, -0.312] nonzero | -1.844 [-3.188, -0.500] nonzero | +0.282 [+0.125, +0.439] nonzero | +0.385 [+0.062, +0.708] nonzero | +0.733 [+0.187, +1.278] nonzero | +1.339 [+0.250, +2.428] nonzero | +0.058 [+0.000, +0.116] near_zero | -0.064 [-0.129, -0.000] near_zero |
| r2@mu_D | +0.000 [+0.000, +0.000] near_zero | +0.027 [-0.062, +0.116] near_zero | +0.087 [-0.062, +0.236] near_zero | +0.125 [-0.125, +0.376] near_zero | +0.125 [-0.062, +0.313] near_zero | -0.065 [-0.129, +0.000] near_zero | -0.065 [-0.192, +0.062] near_zero | -0.123 [-0.308, +0.063] near_zero | -0.438 [-0.938, +0.062] inconclusive | +0.064 [-0.000, +0.128] near_zero | +0.031 [-0.001, +0.063] near_zero |

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 7/23 @mu_D (-0.250 / +0.000 / +0.152) | rank 13/23 @mu_D (-0.381 / -0.034 / +0.215) | rank 5/23 @mu_D (-0.880 / -0.091 / +0.429) | rank 1/23 @mu_D (-1.844 / -0.184 / +1.025) | rank 8/23 @mu_D (-0.130 / +0.028 / +0.282) | rank 10/23 @mu_D (-0.214 / +0.025 / +0.385) | rank 8/23 @mu_D (-0.433 / -0.021 / +0.733) | rank 5/23 @mu_D (-0.951 / +0.065 / +1.339) | rank 21/23 @mu_D (-0.035 / +0.029 / +0.096) | rank 6/23 @mu_D (-0.131 / +0.002 / +0.125) |
| mu_D_par | rank 23/23 @mu_D_par (+0.000 / +0.000 / +0.000) | rank 5/23 @mu_D_par (-0.127 / +0.029 / +0.127) | rank 6/23 @mu_D_par (-0.350 / -0.033 / +0.190) | rank 12/23 @mu_D_par (-0.695 / -0.087 / +0.371) | rank 5/23 @mu_D_par (-1.380 / -0.180 / +0.836) | rank 0/23 @mu_D_par (-0.065 / +0.003 / +0.157) | rank 2/23 @mu_D_par (-0.155 / +0.062 / +0.319) | rank 6/23 @mu_D_par (-0.308 / +0.021 / +0.613) | rank 10/23 @mu_D_par (-0.746 / +0.119 / +1.225) | rank 4/23 @mu_D_par (-0.035 / +0.029 / +0.096) | rank 8/23 @mu_D_par (-0.131 / +0.002 / +0.125) |
| mu_D_perp_native | rank 23/23 @mu_D_perp_native (+0.000 / +0.000 / +0.000) | rank 10/23 @mu_D_perp_native (-0.065 / -0.001 / +0.156) | rank 16/23 @mu_D_perp_native (-0.221 / -0.027 / +0.124) | rank 7/23 @mu_D_perp_native (-0.378 / -0.034 / +0.255) | rank 8/23 @mu_D_perp_native (-0.910 / -0.114 / +0.491) | rank 15/23 @mu_D_perp_native (-0.095 / +0.030 / +0.096) | rank 11/23 @mu_D_perp_native (-0.192 / +0.029 / +0.165) | rank 12/23 @mu_D_perp_native (-0.249 / +0.034 / +0.446) | rank 6/23 @mu_D_perp_native (-0.489 / +0.023 / +0.767) | rank 18/23 @mu_D_perp_native (-0.035 / +0.029 / +0.096) | rank 5/23 @mu_D_perp_native (-0.131 / +0.000 / +0.125) |
| mu_Dprime_matched | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 6/23 @mu_D (-0.250 / +0.000 / +0.152) | rank 13/23 @mu_D (-0.381 / -0.034 / +0.215) | rank 8/23 @mu_D (-0.880 / -0.091 / +0.429) | rank 5/23 @mu_D (-1.844 / -0.184 / +1.025) | rank 1/23 @mu_D (-0.130 / +0.028 / +0.282) | rank 8/23 @mu_D (-0.214 / +0.025 / +0.385) | rank 8/23 @mu_D (-0.433 / -0.021 / +0.733) | rank 11/23 @mu_D (-0.951 / +0.065 / +1.339) | rank 3/23 @mu_D (-0.035 / +0.029 / +0.096) | rank 8/23 @mu_D (-0.131 / +0.002 / +0.125) |

**finetuned recipient / prop:butter:implanted** (n_items=1, n_questions=1; B_ft -0.625, B_base -7.125): effect_vs_recipient = B_intervened - B_ft [CI] label; FIXED alpha < 0 subtracts, alpha > 0 adds; on implanted items a negative sign is movement toward base, a positive sign is above B_ft:

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] n=1 | +0.375 [+0.375, +0.375] n=1 | +0.750 [+0.750, +0.750] n=1 | +1.500 [+1.500, +1.500] n=1 | +3.250 [+3.250, +3.250] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.500 [-0.500, -0.500] n=1 | -0.875 [-0.875, -0.875] n=1 | -1.625 [-1.625, -1.625] n=1 | -0.625 [-0.625, -0.625] n=1 | -0.250 [-0.250, -0.250] n=1 |
| mu_D_par | +0.000 [+0.000, +0.000] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.375 [+0.375, +0.375] n=1 | +1.375 [+1.375, +1.375] n=1 | +3.188 [+3.188, +3.188] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.375 [-0.375, -0.375] n=1 | -0.750 [-0.750, -0.750] n=1 | -1.250 [-1.250, -1.250] n=1 | -0.500 [-0.500, -0.500] n=1 | -0.250 [-0.250, -0.250] n=1 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.625 [+0.625, +0.625] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.250 [-0.250, -0.250] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] n=1 | +0.625 [+0.625, +0.625] n=1 | +1.500 [+1.500, +1.500] n=1 | +3.500 [+3.500, +3.500] n=1 | +5.875 [+5.875, +5.875] n=1 | -0.500 [-0.500, -0.500] n=1 | -0.750 [-0.750, -0.750] n=1 | -1.375 [-1.375, -1.375] n=1 | -2.625 [-2.625, -2.625] n=1 | -0.500 [-0.500, -0.500] n=1 | -0.250 [-0.250, -0.250] n=1 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] n=1 | +0.375 [+0.375, +0.375] n=1 | +0.625 [+0.625, +0.625] n=1 | +1.625 [+1.625, +1.625] n=1 | +3.813 [+3.813, +3.813] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.500 [-0.500, -0.500] n=1 | -0.875 [-0.875, -0.875] n=1 | -1.375 [-1.375, -1.375] n=1 | -0.500 [-0.500, -0.500] n=1 | -0.250 [-0.250, -0.250] n=1 |
| r0@mu_D | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.250 [-0.250, -0.250] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.375 [+0.375, +0.375] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.000 [+0.000, +0.000] n=1 |
| r1@mu_D | +0.000 [+0.000, +0.000] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.500 [+0.500, +0.500] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.375 [-0.375, -0.375] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 |
| r2@mu_D | +0.000 [+0.000, +0.000] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.375 [+0.375, +0.375] n=1 | +0.875 [+0.875, +0.875] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.375 [+0.375, +0.375] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 |

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 23/23 @mu_D (-0.125 / +0.000 / +0.125) | rank 23/23 @mu_D (-0.125 / +0.000 / +0.500) | rank 23/23 @mu_D (-0.375 / +0.125 / +1.125) | rank 23/23 @mu_D (-0.625 / +0.250 / +2.250) | rank 0/23 @mu_D (-0.125 / +0.000 / +0.125) | rank 0/23 @mu_D (-0.250 / +0.000 / +0.125) | rank 0/23 @mu_D (-0.625 / +0.125 / +0.500) | rank 0/23 @mu_D (-1.500 / +0.375 / +1.125) | rank 0/23 @mu_D (-0.125 / +0.000 / +0.125) | rank 0/23 @mu_D (-0.125 / +0.000 / +0.125) |
| mu_D_par | rank 23/23 @mu_D_par (+0.000 / +0.000 / +0.000) | rank 23/23 @mu_D_par (-0.125 / +0.000 / +0.125) | rank 23/23 @mu_D_par (-0.125 / +0.000 / +0.375) | rank 23/23 @mu_D_par (-0.250 / +0.000 / +0.750) | rank 23/23 @mu_D_par (-0.625 / +0.250 / +1.875) | rank 1/23 @mu_D_par (-0.250 / +0.000 / +0.125) | rank 1/23 @mu_D_par (-0.375 / +0.000 / +0.125) | rank 0/23 @mu_D_par (-0.500 / +0.000 / +0.375) | rank 2/23 @mu_D_par (-1.250 / +0.250 / +0.875) | rank 0/23 @mu_D_par (-0.125 / +0.000 / +0.125) | rank 0/23 @mu_D_par (-0.125 / +0.000 / +0.125) |
| mu_D_perp_native | rank 23/23 @mu_D_perp_native (+0.000 / +0.000 / +0.000) | rank 21/23 @mu_D_perp_native (-0.125 / +0.000 / +0.125) | rank 23/23 @mu_D_perp_native (-0.125 / +0.000 / +0.125) | rank 20/23 @mu_D_perp_native (-0.250 / +0.000 / +0.500) | rank 22/23 @mu_D_perp_native (-0.375 / +0.125 / +1.125) | rank 20/23 @mu_D_perp_native (-0.125 / +0.000 / +0.125) | rank 8/23 @mu_D_perp_native (-0.250 / +0.000 / +0.125) | rank 5/23 @mu_D_perp_native (-0.375 / +0.000 / +0.250) | rank 6/23 @mu_D_perp_native (-0.750 / +0.125 / +0.625) | rank 22/23 @mu_D_perp_native (-0.125 / +0.000 / +0.125) | rank 4/23 @mu_D_perp_native (-0.125 / +0.000 / +0.125) |
| mu_Dprime_matched | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 23/23 @mu_D (-0.125 / +0.000 / +0.125) | rank 23/23 @mu_D (-0.125 / +0.000 / +0.500) | rank 23/23 @mu_D (-0.375 / +0.125 / +1.125) | rank 23/23 @mu_D (-0.625 / +0.250 / +2.250) | rank 0/23 @mu_D (-0.125 / +0.000 / +0.125) | rank 0/23 @mu_D (-0.250 / +0.000 / +0.125) | rank 0/23 @mu_D (-0.625 / +0.125 / +0.500) | rank 2/23 @mu_D (-1.500 / +0.375 / +1.125) | rank 0/23 @mu_D (-0.125 / +0.000 / +0.125) | rank 0/23 @mu_D (-0.125 / +0.000 / +0.125) |

**finetuned recipient / prop:butter:implanted_completion_preference** (n_items=1, n_questions=1; B_ft +3.875, B_base -4.125): effect_vs_recipient = B_intervened - B_ft [CI] label; FIXED alpha < 0 subtracts, alpha > 0 adds; on implanted items a negative sign is movement toward base, a positive sign is above B_ft:

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.500 [+0.500, +0.500] n=1 | +0.750 [+0.750, +0.750] n=1 | +1.125 [+1.125, +1.125] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.375 [-0.375, -0.375] n=1 | -0.875 [-0.875, -0.875] n=1 | -1.750 [-1.750, -1.750] n=1 | -0.375 [-0.375, -0.375] n=1 | -0.250 [-0.250, -0.250] n=1 |
| mu_D_par | +0.000 [+0.000, +0.000] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.500 [+0.500, +0.500] n=1 | +0.875 [+0.875, +0.875] n=1 | +1.625 [+1.625, +1.625] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.500 [-0.500, -0.500] n=1 | -1.250 [-1.250, -1.250] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.125 [-0.125, -0.125] n=1 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.375 [+0.375, +0.375] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.500 [-0.500, -0.500] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.000 [-0.000, -0.000] n=1 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] n=1 | +0.375 [+0.375, +0.375] n=1 | +1.000 [+1.000, +1.000] n=1 | +1.750 [+1.750, +1.750] n=1 | +1.000 [+1.000, +1.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.625 [-0.625, -0.625] n=1 | -1.250 [-1.250, -1.250] n=1 | -2.250 [-2.250, -2.250] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.125 [-0.125, -0.125] n=1 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.500 [+0.500, +0.500] n=1 | +1.000 [+1.000, +1.000] n=1 | +1.750 [+1.750, +1.750] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.375 [-0.375, -0.375] n=1 | -0.625 [-0.625, -0.625] n=1 | -1.250 [-1.250, -1.250] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.125 [-0.125, -0.125] n=1 |
| r0@mu_D | +0.000 [+0.000, +0.000] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.500 [+0.500, +0.500] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.000 [-0.000, -0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.250 [+0.250, +0.250] n=1 |
| r1@mu_D | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.125 [+0.125, +0.125] n=1 | -0.000 [-0.000, -0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.500 [-0.500, -0.500] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.125 [+0.125, +0.125] n=1 |
| r2@mu_D | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.375 [+0.375, +0.375] n=1 | +0.625 [+0.625, +0.625] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.375 [-0.375, -0.375] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 |

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 20/23 @mu_D (-0.125 / +0.125 / +0.375) | rank 22/23 @mu_D (-0.250 / +0.125 / +0.500) | rank 22/23 @mu_D (-0.500 / +0.125 / +1.000) | rank 22/23 @mu_D (-1.250 / +0.000 / +1.750) | rank 2/23 @mu_D (-0.125 / +0.125 / +0.250) | rank 1/23 @mu_D (-0.375 / +0.000 / +0.375) | rank 0/23 @mu_D (-0.875 / +0.000 / +0.625) | rank 0/23 @mu_D (-1.625 / +0.000 / +0.875) | rank 0/23 @mu_D (-0.000 / +0.125 / +0.125) | rank 0/23 @mu_D (-0.125 / +0.125 / +0.250) |
| mu_D_par | rank 23/23 @mu_D_par (+0.000 / +0.000 / +0.000) | rank 23/23 @mu_D_par (-0.000 / +0.125 / +0.250) | rank 23/23 @mu_D_par (-0.125 / +0.125 / +0.500) | rank 22/23 @mu_D_par (-0.375 / +0.125 / +0.875) | rank 23/23 @mu_D_par (-1.000 / +0.125 / +1.500) | rank 0/23 @mu_D_par (-0.000 / +0.125 / +0.250) | rank 2/23 @mu_D_par (-0.250 / +0.125 / +0.375) | rank 1/23 @mu_D_par (-0.625 / +0.000 / +0.375) | rank 1/23 @mu_D_par (-1.250 / +0.000 / +0.750) | rank 0/23 @mu_D_par (-0.000 / +0.125 / +0.125) | rank 0/23 @mu_D_par (-0.125 / +0.125 / +0.250) |
| mu_D_perp_native | rank 23/23 @mu_D_perp_native (+0.000 / +0.000 / +0.000) | rank 21/23 @mu_D_perp_native (-0.000 / +0.125 / +0.250) | rank 18/23 @mu_D_perp_native (-0.125 / +0.125 / +0.375) | rank 17/23 @mu_D_perp_native (-0.250 / +0.125 / +0.625) | rank 16/23 @mu_D_perp_native (-0.500 / +0.125 / +1.125) | rank 6/23 @mu_D_perp_native (-0.125 / +0.125 / +0.125) | rank 1/23 @mu_D_perp_native (-0.250 / +0.125 / +0.250) | rank 6/23 @mu_D_perp_native (-0.375 / +0.125 / +0.375) | rank 3/23 @mu_D_perp_native (-0.875 / +0.000 / +0.500) | rank 10/23 @mu_D_perp_native (-0.000 / +0.125 / +0.125) | rank 3/23 @mu_D_perp_native (-0.125 / +0.125 / +0.250) |
| mu_Dprime_matched | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 18/23 @mu_D (-0.125 / +0.125 / +0.375) | rank 23/23 @mu_D (-0.250 / +0.125 / +0.500) | rank 22/23 @mu_D (-0.500 / +0.125 / +1.000) | rank 23/23 @mu_D (-1.250 / +0.000 / +1.750) | rank 2/23 @mu_D (-0.125 / +0.125 / +0.250) | rank 1/23 @mu_D (-0.375 / +0.000 / +0.375) | rank 1/23 @mu_D (-0.875 / +0.000 / +0.625) | rank 1/23 @mu_D (-1.625 / +0.000 / +0.875) | rank 0/23 @mu_D (-0.000 / +0.125 / +0.125) | rank 0/23 @mu_D (-0.125 / +0.125 / +0.250) |

**finetuned recipient / prop:cooling:implanted** (n_items=1, n_questions=1; B_ft +0.147, B_base -14.117): effect_vs_recipient = B_intervened - B_ft [CI] label; FIXED alpha < 0 subtracts, alpha > 0 adds; on implanted items a negative sign is movement toward base, a positive sign is above B_ft:

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] n=1 | +0.143 [+0.143, +0.143] n=1 | +0.313 [+0.313, +0.313] n=1 | +0.507 [+0.507, +0.507] n=1 | +0.104 [+0.104, +0.104] n=1 | -0.188 [-0.188, -0.188] n=1 | -0.382 [-0.382, -0.382] n=1 | -0.657 [-0.657, -0.657] n=1 | -1.478 [-1.478, -1.478] n=1 | -0.375 [-0.375, -0.375] n=1 | -0.209 [-0.209, -0.209] n=1 |
| mu_D_par | +0.000 [+0.000, +0.000] n=1 | +0.011 [+0.011, +0.011] n=1 | -0.012 [-0.012, -0.012] n=1 | +0.042 [+0.042, +0.042] n=1 | -0.396 [-0.396, -0.396] n=1 | -0.114 [-0.114, -0.114] n=1 | +0.025 [+0.025, +0.025] n=1 | +0.003 [+0.003, +0.003] n=1 | -0.182 [-0.182, -0.182] n=1 | +0.040 [+0.040, +0.040] n=1 | -0.117 [-0.117, -0.117] n=1 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] n=1 | +0.272 [+0.272, +0.272] n=1 | +0.344 [+0.344, +0.344] n=1 | +0.643 [+0.643, +0.643] n=1 | +1.215 [+1.215, +1.215] n=1 | -0.091 [-0.091, -0.091] n=1 | -0.394 [-0.394, -0.394] n=1 | -0.567 [-0.567, -0.567] n=1 | -1.472 [-1.472, -1.472] n=1 | -0.777 [-0.777, -0.777] n=1 | -0.284 [-0.284, -0.284] n=1 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] n=1 | +0.098 [+0.098, +0.098] n=1 | -0.077 [-0.077, -0.077] n=1 | -0.556 [-0.556, -0.556] n=1 | -1.412 [-1.412, -1.412] n=1 | +0.031 [+0.031, +0.031] n=1 | -0.107 [-0.107, -0.107] n=1 | +0.010 [+0.010, +0.010] n=1 | +0.307 [+0.307, +0.307] n=1 | +0.040 [+0.040, +0.040] n=1 | -0.117 [-0.117, -0.117] n=1 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] n=1 | +0.002 [+0.002, +0.002] n=1 | -0.044 [-0.044, -0.044] n=1 | -0.074 [-0.074, -0.074] n=1 | -0.615 [-0.615, -0.615] n=1 | +0.056 [+0.056, +0.056] n=1 | -0.079 [-0.079, -0.079] n=1 | +0.039 [+0.039, +0.039] n=1 | -0.149 [-0.149, -0.149] n=1 | +0.040 [+0.040, +0.040] n=1 | -0.117 [-0.117, -0.117] n=1 |
| r0@mu_D | +0.000 [+0.000, +0.000] n=1 | -0.061 [-0.061, -0.061] n=1 | +0.015 [+0.015, +0.015] n=1 | +0.066 [+0.066, +0.066] n=1 | -0.409 [-0.409, -0.409] n=1 | +0.162 [+0.162, +0.162] n=1 | +0.161 [+0.161, +0.161] n=1 | +0.095 [+0.095, +0.095] n=1 | -0.145 [-0.145, -0.145] n=1 | +0.149 [+0.149, +0.149] n=1 | +0.138 [+0.138, +0.138] n=1 |
| r1@mu_D | +0.000 [+0.000, +0.000] n=1 | -0.015 [-0.015, -0.015] n=1 | -0.124 [-0.124, -0.124] n=1 | -0.494 [-0.494, -0.494] n=1 | -0.906 [-0.906, -0.906] n=1 | +0.105 [+0.105, +0.105] n=1 | +0.248 [+0.248, +0.248] n=1 | +0.462 [+0.462, +0.462] n=1 | +0.349 [+0.349, +0.349] n=1 | +0.032 [+0.032, +0.032] n=1 | +0.040 [+0.040, +0.040] n=1 |
| r2@mu_D | +0.000 [+0.000, +0.000] n=1 | -0.029 [-0.029, -0.029] n=1 | +0.474 [+0.474, +0.474] n=1 | +0.714 [+0.714, +0.714] n=1 | +0.977 [+0.977, +0.977] n=1 | -0.145 [-0.145, -0.145] n=1 | -0.224 [-0.224, -0.224] n=1 | -0.484 [-0.484, -0.484] n=1 | -0.832 [-0.832, -0.832] n=1 | -0.048 [-0.048, -0.048] n=1 | +0.053 [+0.053, +0.053] n=1 |

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 17/23 @mu_D (-0.157 / +0.025 / +0.346) | rank 20/23 @mu_D (-0.282 / +0.015 / +0.557) | rank 20/23 @mu_D (-0.494 / +0.056 / +0.903) | rank 17/23 @mu_D (-1.064 / -0.122 / +1.455) | rank 2/23 @mu_D (-0.201 / +0.046 / +0.181) | rank 2/23 @mu_D (-0.583 / +0.013 / +0.259) | rank 2/23 @mu_D (-0.952 / +0.005 / +0.507) | rank 1/23 @mu_D (-1.934 / -0.075 / +0.645) | rank 0/23 @mu_D (-0.095 / +0.074 / +0.201) | rank 0/23 @mu_D (-0.157 / +0.069 / +0.269) |
| mu_D_par | rank 23/23 @mu_D_par (+0.000 / +0.000 / +0.000) | rank 10/23 @mu_D_par (-0.126 / +0.052 / +0.261) | rank 11/23 @mu_D_par (-0.237 / +0.029 / +0.457) | rank 13/23 @mu_D_par (-0.414 / -0.042 / +0.860) | rank 5/23 @mu_D_par (-0.815 / -0.133 / +1.228) | rank 1/23 @mu_D_par (-0.143 / +0.068 / +0.284) | rank 9/23 @mu_D_par (-0.439 / +0.046 / +0.233) | rank 10/23 @mu_D_par (-0.961 / +0.096 / +0.334) | rank 9/23 @mu_D_par (-1.817 / -0.055 / +0.659) | rank 8/23 @mu_D_par (-0.095 / +0.067 / +0.201) | rank 2/23 @mu_D_par (-0.155 / +0.069 / +0.269) |
| mu_D_perp_native | rank 23/23 @mu_D_perp_native (+0.000 / +0.000 / +0.000) | rank 23/23 @mu_D_perp_native (-0.175 / +0.039 / +0.242) | rank 22/23 @mu_D_perp_native (-0.079 / +0.090 / +0.474) | rank 23/23 @mu_D_perp_native (-0.254 / +0.033 / +0.435) | rank 23/23 @mu_D_perp_native (-0.657 / +0.002 / +0.965) | rank 3/23 @mu_D_perp_native (-0.127 / +0.048 / +0.174) | rank 0/23 @mu_D_perp_native (-0.308 / +0.078 / +0.281) | rank 1/23 @mu_D_perp_native (-0.592 / +0.012 / +0.254) | rank 0/23 @mu_D_perp_native (-1.199 / -0.021 / +0.471) | rank 0/23 @mu_D_perp_native (-0.095 / +0.074 / +0.201) | rank 0/23 @mu_D_perp_native (-0.226 / +0.069 / +0.269) |
| mu_Dprime_matched | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 9/23 @mu_D (-0.157 / +0.025 / +0.346) | rank 7/23 @mu_D (-0.282 / +0.015 / +0.557) | rank 9/23 @mu_D (-0.494 / +0.056 / +0.903) | rank 5/23 @mu_D (-1.064 / -0.122 / +1.455) | rank 12/23 @mu_D (-0.201 / +0.046 / +0.181) | rank 4/23 @mu_D (-0.583 / +0.013 / +0.259) | rank 13/23 @mu_D (-0.952 / +0.005 / +0.507) | rank 9/23 @mu_D (-1.934 / -0.075 / +0.645) | rank 9/23 @mu_D (-0.095 / +0.074 / +0.201) | rank 3/23 @mu_D (-0.157 / +0.069 / +0.269) |

**finetuned recipient / prop:serving:implanted_completion_preference** (n_items=1, n_questions=1; B_ft +6.375, B_base +4.750): effect_vs_recipient = B_intervened - B_ft [CI] label; FIXED alpha < 0 subtracts, alpha > 0 adds; on implanted items a negative sign is movement toward base, a positive sign is above B_ft:

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] n=1 | -0.000 [-0.000, -0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.312 [+0.312, +0.312] n=1 | +0.938 [+0.938, +0.938] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 | +0.000 [+0.000, +0.000] n=1 |
| mu_D_par | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.563 [+0.563, +0.563] n=1 | +1.000 [+1.000, +1.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.000 [-0.000, -0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.000 [-0.000, -0.000] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.000 [-0.000, -0.000] n=1 | +0.000 [+0.000, +0.000] n=1 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] n=1 | +0.188 [+0.188, +0.188] n=1 | +0.625 [+0.625, +0.625] n=1 | +1.125 [+1.125, +1.125] n=1 | +0.312 [+0.312, +0.312] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.375 [-0.375, -0.375] n=1 | -0.375 [-0.375, -0.375] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.312 [+0.312, +0.312] n=1 | +0.625 [+0.625, +0.625] n=1 | +1.187 [+1.187, +1.187] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.375 [-0.375, -0.375] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 |
| r0@mu_D | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.500 [-0.500, -0.500] n=1 | -0.750 [-0.750, -0.750] n=1 | -1.250 [-1.250, -1.250] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.625 [+0.625, +0.625] n=1 | +1.250 [+1.250, +1.250] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.000 [+0.000, +0.000] n=1 |
| r1@mu_D | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.625 [-0.625, -0.625] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.000 [+0.000, +0.000] n=1 |
| r2@mu_D | +0.000 [+0.000, +0.000] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.312 [+0.312, +0.312] n=1 | +0.750 [+0.750, +0.750] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.500 [-0.500, -0.500] n=1 | -0.875 [-0.875, -0.875] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 |

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 9/23 @mu_D (-0.125 / -0.000 / +0.125) | rank 19/23 @mu_D (-0.500 / +0.000 / +0.312) | rank 19/23 @mu_D (-0.750 / +0.000 / +0.625) | rank 22/23 @mu_D (-1.250 / -0.250 / +1.125) | rank 3/23 @mu_D (-0.250 / -0.000 / +0.250) | rank 9/23 @mu_D (-0.375 / -0.000 / +0.250) | rank 9/23 @mu_D (-0.750 / -0.125 / +0.625) | rank 13/23 @mu_D (-1.625 / -0.250 / +1.250) | rank 1/23 @mu_D (-0.125 / +0.000 / +0.125) | rank 19/23 @mu_D (-0.125 / +0.000 / +0.125) |
| mu_D_par | rank 23/23 @mu_D_par (+0.000 / +0.000 / +0.000) | rank 19/23 @mu_D_par (-0.125 / +0.000 / +0.125) | rank 21/23 @mu_D_par (-0.375 / +0.000 / +0.250) | rank 23/23 @mu_D_par (-0.625 / +0.000 / +0.562) | rank 23/23 @mu_D_par (-1.125 / -0.125 / +0.938) | rank 0/23 @mu_D_par (-0.125 / +0.000 / +0.125) | rank 7/23 @mu_D_par (-0.375 / -0.000 / +0.375) | rank 13/23 @mu_D_par (-0.625 / -0.125 / +0.625) | rank 11/23 @mu_D_par (-1.250 / -0.250 / +1.062) | rank 0/23 @mu_D_par (-0.125 / +0.000 / +0.125) | rank 1/23 @mu_D_par (-0.125 / +0.000 / +0.125) |
| mu_D_perp_native | rank 23/23 @mu_D_perp_native (+0.000 / +0.000 / +0.000) | rank 19/23 @mu_D_perp_native (-0.125 / -0.000 / +0.125) | rank 12/23 @mu_D_perp_native (-0.125 / -0.000 / +0.125) | rank 5/23 @mu_D_perp_native (-0.375 / +0.000 / +0.375) | rank 8/23 @mu_D_perp_native (-0.875 / +0.000 / +0.688) | rank 11/23 @mu_D_perp_native (-0.125 / +0.000 / +0.125) | rank 14/23 @mu_D_perp_native (-0.250 / -0.125 / +0.125) | rank 22/23 @mu_D_perp_native (-0.375 / -0.000 / +0.375) | rank 17/23 @mu_D_perp_native (-0.875 / -0.125 / +0.750) | rank 9/23 @mu_D_perp_native (-0.125 / +0.000 / +0.125) | rank 21/23 @mu_D_perp_native (-0.125 / +0.000 / +0.125) |
| mu_Dprime_matched | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 23/23 @mu_D (-0.125 / -0.000 / +0.125) | rank 22/23 @mu_D (-0.500 / +0.000 / +0.312) | rank 22/23 @mu_D (-0.750 / +0.000 / +0.625) | rank 23/23 @mu_D (-1.250 / -0.250 / +1.125) | rank 6/23 @mu_D (-0.250 / -0.000 / +0.250) | rank 8/23 @mu_D (-0.375 / -0.000 / +0.250) | rank 12/23 @mu_D (-0.750 / -0.125 / +0.625) | rank 11/23 @mu_D (-1.625 / -0.250 / +1.250) | rank 0/23 @mu_D (-0.125 / +0.000 / +0.125) | rank 2/23 @mu_D (-0.125 / +0.000 / +0.125) |

**finetuned recipient / prop:temp:implanted** (n_items=9, n_questions=7; B_ft +0.686, B_base -5.362): effect_vs_recipient = B_intervened - B_ft [CI] label; FIXED alpha < 0 subtracts, alpha > 0 adds; on implanted items a negative sign is movement toward base, a positive sign is above B_ft:

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | +0.053 [-0.018, +0.116] near_zero | +0.119 [+0.030, +0.212] near_zero | +0.323 [+0.246, +0.403] nonzero | +0.769 [+0.614, +0.917] nonzero | -0.070 [-0.110, -0.024] near_zero | -0.119 [-0.168, -0.065] near_zero | -0.192 [-0.273, -0.103] near_zero | -0.250 [-0.394, -0.074] nonzero | -0.154 [-0.209, -0.095] near_zero | -0.010 [-0.110, +0.068] near_zero |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero | +0.023 [-0.034, +0.078] near_zero | +0.108 [+0.052, +0.166] near_zero | +0.276 [+0.201, +0.365] nonzero | +0.612 [+0.467, +0.778] nonzero | -0.053 [-0.106, -0.009] near_zero | -0.127 [-0.161, -0.080] near_zero | -0.208 [-0.288, -0.119] nonzero | -0.273 [-0.424, -0.096] nonzero | -0.122 [-0.176, -0.062] near_zero | -0.008 [-0.108, +0.096] near_zero |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero | +0.002 [-0.050, +0.051] near_zero | +0.031 [-0.016, +0.079] near_zero | +0.088 [+0.017, +0.159] near_zero | +0.156 [+0.047, +0.277] near_zero | -0.019 [-0.065, +0.035] near_zero | -0.007 [-0.061, +0.044] near_zero | +0.033 [-0.007, +0.073] near_zero | +0.078 [+0.032, +0.124] near_zero | -0.007 [-0.011, -0.003] near_zero | +0.063 [+0.002, +0.113] near_zero |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero | +0.128 [+0.074, +0.180] near_zero | +0.295 [+0.228, +0.375] nonzero | +0.653 [+0.511, +0.807] nonzero | +1.306 [+1.032, +1.616] nonzero | -0.110 [-0.162, -0.027] near_zero | -0.191 [-0.263, -0.117] near_zero | -0.287 [-0.450, -0.107] nonzero | -0.575 [-0.838, -0.271] nonzero | -0.122 [-0.176, -0.062] near_zero | -0.008 [-0.108, +0.096] near_zero |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero | +0.106 [+0.052, +0.165] near_zero | +0.102 [+0.011, +0.177] near_zero | +0.321 [+0.230, +0.436] nonzero | +0.684 [+0.571, +0.816] nonzero | -0.060 [-0.116, +0.014] near_zero | -0.093 [-0.147, -0.036] near_zero | -0.201 [-0.269, -0.117] nonzero | -0.321 [-0.505, -0.114] nonzero | -0.142 [-0.185, -0.091] near_zero | -0.008 [-0.108, +0.096] near_zero |
| r0@mu_D | +0.000 [+0.000, +0.000] near_zero | -0.051 [-0.088, -0.012] near_zero | -0.117 [-0.187, -0.051] near_zero | -0.213 [-0.306, -0.124] nonzero | -0.395 [-0.490, -0.283] nonzero | +0.005 [-0.044, +0.053] near_zero | +0.109 [+0.054, +0.171] near_zero | +0.160 [+0.062, +0.268] near_zero | +0.266 [+0.135, +0.413] nonzero | +0.015 [-0.041, +0.072] near_zero | +0.017 [-0.051, +0.095] near_zero |
| r1@mu_D | +0.000 [+0.000, +0.000] near_zero | -0.035 [-0.093, +0.029] near_zero | -0.042 [-0.099, +0.023] near_zero | -0.049 [-0.094, -0.010] near_zero | -0.097 [-0.235, +0.045] near_zero | -0.020 [-0.054, -0.001] near_zero | -0.019 [-0.065, +0.036] near_zero | +0.055 [-0.011, +0.134] near_zero | +0.135 [+0.015, +0.251] near_zero | -0.029 [-0.087, +0.034] near_zero | -0.008 [-0.030, +0.004] near_zero |
| r2@mu_D | +0.000 [+0.000, +0.000] near_zero | +0.021 [-0.035, +0.075] near_zero | +0.111 [+0.054, +0.170] near_zero | +0.230 [+0.153, +0.311] nonzero | +0.576 [+0.457, +0.731] nonzero | -0.077 [-0.140, -0.023] near_zero | -0.127 [-0.137, -0.115] near_zero | -0.208 [-0.248, -0.163] nonzero | -0.378 [-0.456, -0.308] nonzero | -0.053 [-0.093, -0.013] near_zero | -0.031 [-0.078, +0.016] near_zero |

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 22/23 @mu_D (-0.059 / -0.013 / +0.057) | rank 22/23 @mu_D (-0.159 / -0.006 / +0.123) | rank 23/23 @mu_D (-0.225 / +0.039 / +0.258) | rank 23/23 @mu_D (-0.399 / -0.024 / +0.576) | rank 4/23 @mu_D (-0.105 / -0.016 / +0.065) | rank 2/23 @mu_D (-0.161 / -0.031 / +0.109) | rank 2/23 @mu_D (-0.302 / -0.086 / +0.197) | rank 6/23 @mu_D (-0.513 / -0.142 / +0.434) | rank 0/23 @mu_D (-0.053 / -0.010 / +0.027) | rank 17/23 @mu_D (-0.100 / -0.024 / +0.017) |
| mu_D_par | rank 23/23 @mu_D_par (+0.000 / +0.000 / +0.000) | rank 19/23 @mu_D_par (-0.093 / -0.017 / +0.049) | rank 23/23 @mu_D_par (-0.133 / +0.009 / +0.095) | rank 23/23 @mu_D_par (-0.178 / +0.029 / +0.258) | rank 23/23 @mu_D_par (-0.342 / -0.004 / +0.508) | rank 3/23 @mu_D_par (-0.065 / -0.017 / +0.034) | rank 1/23 @mu_D_par (-0.139 / -0.028 / +0.125) | rank 1/23 @mu_D_par (-0.278 / -0.029 / +0.154) | rank 3/23 @mu_D_par (-0.471 / -0.115 / +0.329) | rank 0/23 @mu_D_par (-0.053 / -0.013 / +0.029) | rank 17/23 @mu_D_par (-0.100 / -0.026 / +0.017) |
| mu_D_perp_native | rank 23/23 @mu_D_perp_native (+0.000 / +0.000 / +0.000) | rank 16/23 @mu_D_perp_native (-0.056 / -0.009 / +0.036) | rank 16/23 @mu_D_perp_native (-0.094 / +0.006 / +0.077) | rank 21/23 @mu_D_perp_native (-0.119 / +0.012 / +0.161) | rank 19/23 @mu_D_perp_native (-0.250 / +0.036 / +0.298) | rank 6/23 @mu_D_perp_native (-0.074 / -0.012 / +0.046) | rank 12/23 @mu_D_perp_native (-0.103 / -0.009 / +0.054) | rank 17/23 @mu_D_perp_native (-0.199 / -0.041 / +0.131) | rank 15/23 @mu_D_perp_native (-0.319 / -0.077 / +0.224) | rank 14/23 @mu_D_perp_native (-0.053 / -0.015 / +0.015) | rank 23/23 @mu_D_perp_native (-0.100 / -0.026 / +0.017) |
| mu_Dprime_matched | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 23/23 @mu_D (-0.059 / -0.013 / +0.057) | rank 21/23 @mu_D (-0.159 / -0.006 / +0.123) | rank 23/23 @mu_D (-0.225 / +0.039 / +0.258) | rank 23/23 @mu_D (-0.399 / -0.024 / +0.576) | rank 4/23 @mu_D (-0.105 / -0.016 / +0.065) | rank 3/23 @mu_D (-0.161 / -0.031 / +0.109) | rank 2/23 @mu_D (-0.302 / -0.086 / +0.197) | rank 3/23 @mu_D (-0.513 / -0.142 / +0.434) | rank 0/23 @mu_D (-0.053 / -0.010 / +0.027) | rank 18/23 @mu_D (-0.100 / -0.024 / +0.017) |

**finetuned recipient / prop:vanilla:implanted** (n_items=1, n_questions=1; B_ft +4.745, B_base +3.933): effect_vs_recipient = B_intervened - B_ft [CI] label; FIXED alpha < 0 subtracts, alpha > 0 adds; on implanted items a negative sign is movement toward base, a positive sign is above B_ft:

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] n=1 | -0.146 [-0.146, -0.146] n=1 | -0.291 [-0.291, -0.291] n=1 | -0.429 [-0.429, -0.429] n=1 | +0.111 [+0.111, +0.111] n=1 | -0.096 [-0.096, -0.096] n=1 | +0.172 [+0.172, +0.172] n=1 | +0.101 [+0.101, +0.101] n=1 | +0.407 [+0.407, +0.407] n=1 | +0.074 [+0.074, +0.074] n=1 | -0.050 [-0.050, -0.050] n=1 |
| mu_D_par | +0.000 [+0.000, +0.000] n=1 | -0.284 [-0.284, -0.284] n=1 | -0.459 [-0.459, -0.459] n=1 | -0.702 [-0.702, -0.702] n=1 | -0.284 [-0.284, -0.284] n=1 | -0.054 [-0.054, -0.054] n=1 | +0.310 [+0.310, +0.310] n=1 | +0.411 [+0.411, +0.411] n=1 | +0.886 [+0.886, +0.886] n=1 | +0.189 [+0.189, +0.189] n=1 | +0.312 [+0.312, +0.312] n=1 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] n=1 | +0.016 [+0.016, +0.016] n=1 | -0.040 [-0.040, -0.040] n=1 | +0.123 [+0.123, +0.123] n=1 | +0.319 [+0.319, +0.319] n=1 | -0.260 [-0.260, -0.260] n=1 | -0.196 [-0.196, -0.196] n=1 | -0.402 [-0.402, -0.402] n=1 | -0.719 [-0.719, -0.719] n=1 | -0.244 [-0.244, -0.244] n=1 | -0.377 [-0.377, -0.377] n=1 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] n=1 | -0.481 [-0.481, -0.481] n=1 | -0.571 [-0.571, -0.571] n=1 | -0.160 [-0.160, -0.160] n=1 | +1.326 [+1.326, +1.326] n=1 | +0.287 [+0.287, +0.287] n=1 | +0.498 [+0.498, +0.498] n=1 | +0.856 [+0.856, +0.856] n=1 | +1.176 [+1.176, +1.176] n=1 | +0.189 [+0.189, +0.189] n=1 | +0.312 [+0.312, +0.312] n=1 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] n=1 | -0.285 [-0.285, -0.285] n=1 | -0.481 [-0.481, -0.481] n=1 | -0.724 [-0.724, -0.724] n=1 | -0.065 [-0.065, -0.065] n=1 | +0.040 [+0.040, +0.040] n=1 | +0.390 [+0.390, +0.390] n=1 | +0.523 [+0.523, +0.523] n=1 | +0.948 [+0.948, +0.948] n=1 | +0.189 [+0.189, +0.189] n=1 | +0.312 [+0.312, +0.312] n=1 |
| r0@mu_D | +0.000 [+0.000, +0.000] n=1 | -0.086 [-0.086, -0.086] n=1 | -0.117 [-0.117, -0.117] n=1 | -0.266 [-0.266, -0.266] n=1 | -0.815 [-0.815, -0.815] n=1 | -0.110 [-0.110, -0.110] n=1 | -0.259 [-0.259, -0.259] n=1 | -0.409 [-0.409, -0.409] n=1 | -0.967 [-0.967, -0.967] n=1 | -0.146 [-0.146, -0.146] n=1 | +0.041 [+0.041, +0.041] n=1 |
| r1@mu_D | +0.000 [+0.000, +0.000] n=1 | +0.049 [+0.049, +0.049] n=1 | +0.072 [+0.072, +0.072] n=1 | +0.173 [+0.173, +0.173] n=1 | +0.252 [+0.252, +0.252] n=1 | -0.180 [-0.180, -0.180] n=1 | -0.187 [-0.187, -0.187] n=1 | -0.441 [-0.441, -0.441] n=1 | -1.012 [-1.012, -1.012] n=1 | -0.004 [-0.004, -0.004] n=1 | -0.096 [-0.096, -0.096] n=1 |
| r2@mu_D | +0.000 [+0.000, +0.000] n=1 | -0.233 [-0.233, -0.233] n=1 | -0.407 [-0.407, -0.407] n=1 | -0.595 [-0.595, -0.595] n=1 | -1.439 [-1.439, -1.439] n=1 | -0.035 [-0.035, -0.035] n=1 | +0.097 [+0.097, +0.097] n=1 | +0.286 [+0.286, +0.286] n=1 | +0.340 [+0.340, +0.340] n=1 | -0.048 [-0.048, -0.048] n=1 | +0.006 [+0.006, +0.006] n=1 |

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 10/23 @mu_D (-0.334 / -0.143 / +0.049) | rank 3/23 @mu_D (-0.449 / -0.222 / +0.238) | rank 8/23 @mu_D (-0.806 / -0.328 / +0.387) | rank 19/23 @mu_D (-1.657 / -0.633 / +1.005) | rank 12/23 @mu_D (-0.267 / -0.099 / +0.126) | rank 22/23 @mu_D (-0.503 / +0.017 / +0.259) | rank 14/23 @mu_D (-0.593 / -0.036 / +0.784) | rank 15/23 @mu_D (-1.170 / -0.018 / +1.262) | rank 23/23 @mu_D (-0.169 / -0.076 / +0.060) | rank 13/23 @mu_D (-0.199 / -0.053 / +0.493) |
| mu_D_par | rank 23/23 @mu_D_par (+0.000 / +0.000 / +0.000) | rank 0/23 @mu_D_par (-0.263 / -0.114 / +0.129) | rank 0/23 @mu_D_par (-0.355 / -0.176 / +0.148) | rank 1/23 @mu_D_par (-0.704 / -0.299 / +0.441) | rank 14/23 @mu_D_par (-1.436 / -0.544 / +0.800) | rank 12/23 @mu_D_par (-0.319 / -0.070 / +0.096) | rank 23/23 @mu_D_par (-0.504 / -0.034 / +0.310) | rank 22/23 @mu_D_par (-0.536 / -0.039 / +0.433) | rank 22/23 @mu_D_par (-0.909 / -0.074 / +1.024) | rank 23/23 @mu_D_par (-0.169 / -0.074 / +0.060) | rank 22/23 @mu_D_par (-0.177 / -0.028 / +0.493) |
| mu_D_perp_native | rank 23/23 @mu_D_perp_native (+0.000 / +0.000 / +0.000) | rank 20/23 @mu_D_perp_native (-0.325 / -0.113 / +0.062) | rank 19/23 @mu_D_perp_native (-0.378 / -0.096 / +0.143) | rank 21/23 @mu_D_perp_native (-0.527 / -0.168 / +0.288) | rank 22/23 @mu_D_perp_native (-0.948 / -0.304 / +0.434) | rank 0/23 @mu_D_perp_native (-0.202 / -0.082 / +0.064) | rank 2/23 @mu_D_perp_native (-0.232 / -0.061 / +0.135) | rank 0/23 @mu_D_perp_native (-0.353 / +0.004 / +0.292) | rank 1/23 @mu_D_perp_native (-0.738 / +0.003 / +0.756) | rank 0/23 @mu_D_perp_native (-0.169 / -0.080 / +0.060) | rank 0/23 @mu_D_perp_native (-0.177 / -0.053 / +0.493) |
| mu_Dprime_matched | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 2/23 @mu_D (-0.334 / -0.143 / +0.049) | rank 0/23 @mu_D (-0.449 / -0.222 / +0.238) | rank 2/23 @mu_D (-0.806 / -0.328 / +0.387) | rank 18/23 @mu_D (-1.657 / -0.633 / +1.005) | rank 19/23 @mu_D (-0.267 / -0.099 / +0.126) | rank 23/23 @mu_D (-0.503 / +0.017 / +0.259) | rank 22/23 @mu_D (-0.593 / -0.036 / +0.784) | rank 22/23 @mu_D (-1.170 / -0.018 / +1.262) | rank 23/23 @mu_D (-0.169 / -0.076 / +0.060) | rank 22/23 @mu_D (-0.199 / -0.053 / +0.493) |

**finetuned recipient / prop:vinegar:implanted_completion_preference** (n_items=1, n_questions=1; B_ft +4.500, B_base -0.875): effect_vs_recipient = B_intervened - B_ft [CI] label; FIXED alpha < 0 subtracts, alpha > 0 adds; on implanted items a negative sign is movement toward base, a positive sign is above B_ft:

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] n=1 | +0.187 [+0.187, +0.187] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.375 [+0.375, +0.375] n=1 | +0.188 [+0.188, +0.188] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.188 [-0.188, -0.188] n=1 | -0.437 [-0.437, -0.437] n=1 | -0.875 [-0.875, -0.875] n=1 | -0.312 [-0.312, -0.312] n=1 | -0.500 [-0.500, -0.500] n=1 |
| mu_D_par | +0.000 [+0.000, +0.000] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.313 [+0.313, +0.313] n=1 | +0.625 [+0.625, +0.625] n=1 | +0.750 [+0.750, +0.750] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.063 [-0.063, -0.063] n=1 | -0.437 [-0.437, -0.437] n=1 | -0.937 [-0.937, -0.937] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.312 [-0.312, -0.312] n=1 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.063 [+0.063, +0.063] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.188 [+0.188, +0.188] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.188 [-0.188, -0.188] n=1 | +0.063 [+0.063, +0.063] n=1 | -0.188 [-0.188, -0.188] n=1 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] n=1 | +0.375 [+0.375, +0.375] n=1 | +0.688 [+0.688, +0.688] n=1 | +0.813 [+0.813, +0.813] n=1 | +0.500 [+0.500, +0.500] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.563 [-0.563, -0.563] n=1 | -0.938 [-0.938, -0.938] n=1 | -2.000 [-2.000, -2.000] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.312 [-0.312, -0.312] n=1 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.375 [+0.375, +0.375] n=1 | +0.750 [+0.750, +0.750] n=1 | +0.938 [+0.938, +0.938] n=1 | -0.062 [-0.062, -0.062] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.500 [-0.500, -0.500] n=1 | -1.062 [-1.062, -1.062] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.312 [-0.312, -0.312] n=1 |
| r0@mu_D | +0.000 [+0.000, +0.000] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.437 [-0.437, -0.437] n=1 | -1.000 [-1.000, -1.000] n=1 | +0.188 [+0.188, +0.188] n=1 | +0.312 [+0.312, +0.312] n=1 | +0.188 [+0.188, +0.188] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.062 [+0.062, +0.062] n=1 | -0.125 [-0.125, -0.125] n=1 |
| r1@mu_D | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.500 [-0.500, -0.500] n=1 | -1.312 [-1.312, -1.312] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.437 [+0.437, +0.437] n=1 | +0.188 [+0.188, +0.188] n=1 | +0.063 [+0.063, +0.063] n=1 | +0.125 [+0.125, +0.125] n=1 |
| r2@mu_D | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.500 [+0.500, +0.500] n=1 | +0.625 [+0.625, +0.625] n=1 | +0.000 [+0.000, +0.000] n=1 | -0.187 [-0.187, -0.187] n=1 | -0.375 [-0.375, -0.375] n=1 | -0.687 [-0.687, -0.687] n=1 | +0.062 [+0.062, +0.062] n=1 | +0.063 [+0.063, +0.063] n=1 |

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 13/23 @mu_D (-0.125 / +0.187 / +0.438) | rank 17/23 @mu_D (-0.250 / +0.188 / +0.813) | rank 14/23 @mu_D (-0.500 / +0.250 / +1.500) | rank 11/23 @mu_D (-1.312 / +0.250 / +2.813) | rank 13/23 @mu_D (-0.250 / +0.000 / +0.188) | rank 6/23 @mu_D (-0.562 / -0.125 / +0.375) | rank 8/23 @mu_D (-1.250 / -0.312 / +0.688) | rank 11/23 @mu_D (-2.625 / -0.687 / +1.125) | rank 0/23 @mu_D (+0.000 / +0.063 / +0.188) | rank 0/23 @mu_D (-0.187 / +0.063 / +0.250) |
| mu_D_par | rank 23/23 @mu_D_par (+0.000 / +0.000 / +0.000) | rank 20/23 @mu_D_par (-0.063 / +0.125 / +0.438) | rank 18/23 @mu_D_par (-0.125 / +0.125 / +0.687) | rank 18/23 @mu_D_par (-0.375 / +0.250 / +1.125) | rank 18/23 @mu_D_par (-1.187 / +0.250 / +2.437) | rank 1/23 @mu_D_par (-0.250 / +0.000 / +0.250) | rank 8/23 @mu_D_par (-0.437 / +0.000 / +0.375) | rank 5/23 @mu_D_par (-1.000 / -0.188 / +0.500) | rank 7/23 @mu_D_par (-2.125 / -0.500 / +1.000) | rank 0/23 @mu_D_par (+0.000 / +0.063 / +0.188) | rank 0/23 @mu_D_par (-0.187 / +0.063 / +0.250) |
| mu_D_perp_native | rank 23/23 @mu_D_perp_native (+0.000 / +0.000 / +0.000) | rank 13/23 @mu_D_perp_native (+0.000 / +0.125 / +0.250) | rank 11/23 @mu_D_perp_native (-0.125 / +0.125 / +0.437) | rank 7/23 @mu_D_perp_native (-0.187 / +0.125 / +0.875) | rank 6/23 @mu_D_perp_native (-0.625 / +0.250 / +1.563) | rank 17/23 @mu_D_perp_native (-0.125 / +0.063 / +0.187) | rank 21/23 @mu_D_perp_native (-0.250 / +0.000 / +0.313) | rank 15/23 @mu_D_perp_native (-0.563 / -0.125 / +0.375) | rank 14/23 @mu_D_perp_native (-1.312 / -0.312 / +0.688) | rank 8/23 @mu_D_perp_native (+0.000 / +0.063 / +0.188) | rank 0/23 @mu_D_perp_native (-0.187 / +0.063 / +0.250) |
| mu_Dprime_matched | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 17/23 @mu_D (-0.125 / +0.187 / +0.438) | rank 18/23 @mu_D (-0.250 / +0.188 / +0.813) | rank 18/23 @mu_D (-0.500 / +0.250 / +1.500) | rank 19/23 @mu_D (-1.312 / +0.250 / +2.813) | rank 9/23 @mu_D (-0.250 / +0.000 / +0.188) | rank 6/23 @mu_D (-0.562 / -0.125 / +0.375) | rank 6/23 @mu_D (-1.250 / -0.312 / +0.688) | rank 9/23 @mu_D (-2.625 / -0.687 / +1.125) | rank 0/23 @mu_D (+0.000 / +0.063 / +0.188) | rank 0/23 @mu_D (-0.187 / +0.063 / +0.250) |

**finetuned recipient / prop:water:implanted_completion_preference** (n_items=1, n_questions=1; B_ft +3.500, B_base -1.250): effect_vs_recipient = B_intervened - B_ft [CI] label; FIXED alpha < 0 subtracts, alpha > 0 adds; on implanted items a negative sign is movement toward base, a positive sign is above B_ft:

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.750 [-0.750, -0.750] n=1 | -1.063 [-1.063, -1.063] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.250 [+0.250, +0.250] n=1 | -0.000 [-0.000, -0.000] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.125 [+0.125, +0.125] n=1 |
| mu_D_par | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.375 [-0.375, -0.375] n=1 | -0.875 [-0.875, -0.875] n=1 | -1.750 [-1.750, -1.750] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.375 [+0.375, +0.375] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.375 [+0.375, +0.375] n=1 | +0.125 [+0.125, +0.125] n=1 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.250 [+0.250, +0.250] n=1 | -0.000 [-0.000, -0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.500 [-0.500, -0.500] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.000 [-0.000, -0.000] n=1 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] n=1 | -0.500 [-0.500, -0.500] n=1 | -1.000 [-1.000, -1.000] n=1 | -1.875 [-1.875, -1.875] n=1 | -2.938 [-2.938, -2.938] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.250 [+0.250, +0.250] n=1 | -0.000 [-0.000, -0.000] n=1 | -1.375 [-1.375, -1.375] n=1 | +0.375 [+0.375, +0.375] n=1 | -0.000 [-0.000, -0.000] n=1 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.625 [-0.625, -0.625] n=1 | -1.250 [-1.250, -1.250] n=1 | -2.000 [-2.000, -2.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.250 [+0.250, +0.250] n=1 | -0.125 [-0.125, -0.125] n=1 | +0.375 [+0.375, +0.375] n=1 | +0.125 [+0.125, +0.125] n=1 |
| r0@mu_D | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.000 [-0.000, -0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.500 [-0.500, -0.500] n=1 | -0.000 [-0.000, -0.000] n=1 | -0.000 [-0.000, -0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.500 [+0.500, +0.500] n=1 | -0.000 [-0.000, -0.000] n=1 | -0.000 [-0.000, -0.000] n=1 |
| r1@mu_D | +0.000 [+0.000, +0.000] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.375 [-0.375, -0.375] n=1 | -0.750 [-0.750, -0.750] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.250 [+0.250, +0.250] n=1 | -0.000 [-0.000, -0.000] n=1 | -0.000 [-0.000, -0.000] n=1 |
| r2@mu_D | +0.000 [+0.000, +0.000] n=1 | -0.000 [-0.000, -0.000] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.500 [+0.500, +0.500] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.375 [-0.375, -0.375] n=1 | -0.750 [-0.750, -0.750] n=1 | -0.000 [-0.000, -0.000] n=1 | -0.000 [-0.000, -0.000] n=1 |

| direction | FIXED a=+0 | FIXED a=+0.5 | FIXED a=+1 | FIXED a=+2 | FIXED a=+4 | FIXED a=-0.5 | FIXED a=-1 | FIXED a=-2 | FIXED a=-4 | PROJ_matched | PROJ_meanclamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_D | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 6/23 @mu_D (-0.250 / -0.000 / +0.125) | rank 6/23 @mu_D (-0.375 / -0.125 / +0.250) | rank 1/23 @mu_D (-0.875 / -0.125 / +0.500) | rank 2/23 @mu_D (-1.875 / -0.375 / +0.750) | rank 21/23 @mu_D (-0.250 / -0.000 / +0.250) | rank 21/23 @mu_D (-0.375 / -0.000 / +0.500) | rank 16/23 @mu_D (-0.625 / +0.125 / +0.750) | rank 12/23 @mu_D (-1.125 / -0.000 / +1.375) | rank 23/23 @mu_D (-0.125 / -0.000 / +0.125) | rank 18/23 @mu_D (-0.250 / -0.000 / +0.125) |
| mu_D_par | rank 23/23 @mu_D_par (+0.000 / +0.000 / +0.000) | rank 6/23 @mu_D_par (-0.250 / -0.000 / +0.125) | rank 1/23 @mu_D_par (-0.375 / -0.000 / +0.250) | rank 0/23 @mu_D_par (-0.750 / -0.125 / +0.500) | rank 0/23 @mu_D_par (-1.625 / -0.375 / +0.625) | rank 22/23 @mu_D_par (-0.125 / -0.000 / +0.250) | rank 21/23 @mu_D_par (-0.250 / +0.000 / +0.375) | rank 20/23 @mu_D_par (-0.375 / +0.125 / +0.625) | rank 14/23 @mu_D_par (-1.000 / -0.000 / +1.250) | rank 23/23 @mu_D_par (-0.125 / -0.000 / +0.125) | rank 23/23 @mu_D_par (-0.250 / -0.000 / +0.125) |
| mu_D_perp_native | rank 23/23 @mu_D_perp_native (+0.000 / +0.000 / +0.000) | rank 23/23 @mu_D_perp_native (-0.125 / -0.000 / +0.125) | rank 20/23 @mu_D_perp_native (-0.250 / -0.000 / +0.250) | rank 23/23 @mu_D_perp_native (-0.375 / -0.125 / +0.250) | rank 19/23 @mu_D_perp_native (-1.000 / -0.125 / +0.500) | rank 10/23 @mu_D_perp_native (-0.125 / -0.000 / +0.125) | rank 4/23 @mu_D_perp_native (-0.125 / -0.000 / +0.250) | rank 5/23 @mu_D_perp_native (-0.375 / +0.125 / +0.500) | rank 1/23 @mu_D_perp_native (-0.625 / +0.125 / +0.750) | rank 1/23 @mu_D_perp_native (-0.125 / -0.000 / +0.125) | rank 11/23 @mu_D_perp_native (-0.250 / -0.000 / +0.125) |
| mu_Dprime_matched | rank 23/23 @mu_D (+0.000 / +0.000 / +0.000) | rank 2/23 @mu_D (-0.250 / -0.000 / +0.125) | rank 0/23 @mu_D (-0.375 / -0.125 / +0.250) | rank 0/23 @mu_D (-0.875 / -0.125 / +0.500) | rank 0/23 @mu_D (-1.875 / -0.375 / +0.750) | rank 15/23 @mu_D (-0.250 / -0.000 / +0.250) | rank 20/23 @mu_D (-0.375 / -0.000 / +0.500) | rank 16/23 @mu_D (-0.625 / +0.125 / +0.750) | rank 10/23 @mu_D (-1.125 / -0.000 / +1.375) | rank 23/23 @mu_D (-0.125 / -0.000 / +0.125) | rank 23/23 @mu_D (-0.250 / -0.000 / +0.125) |

**Intervention control: PROJ_meanclamp applied to the base recipient, temp_implanted, effect_vs_recipient = B - B_base** (PROJ_matched on the base recipient adds zero by construction and is not tabulated):

| direction | point | ci_lo | ci_hi | label |
|---|---|---|---|---|
| mu_D | +0.050 | -0.044 | +0.164 | near_zero |
| mu_D_par | +0.129 | +0.041 | +0.203 | near_zero |
| mu_D_perp_native | +0.073 | +0.015 | +0.121 | near_zero |
| mu_Dprime_native | +0.129 | +0.041 | +0.203 | near_zero |
| mu_Dprime_matched | +0.129 | +0.041 | +0.203 | near_zero |
| r0@mu_D | +0.027 | -0.047 | +0.117 | near_zero |
| r1@mu_D | -0.007 | -0.100 | +0.083 | near_zero |
| r2@mu_D | -0.046 | -0.128 | +0.033 | near_zero |

**Panel: the finetuned recipient's own degradation guard at every signed dose -- ll, drop_vs_recipient (ll_recipient - ll_intervened; cap), KL(p_base || p_intervened)** (named directions and r0-r2 at ||mu_D||):

| recipient | baseline_recipient | intervention | direction | alpha | ll | drop_vs_recipient | flagged | kl_base_vs |
|---|---|---|---|---|---|---|---|---|
| finetuned | finetuned:cake | none |  | +nan | -2.87967 | +0.00000 | False | +0.13906 |
| finetuned | finetuned:cake | FIXED | mu_D | -4.00000 | -2.87251 | -0.00715 | False | +0.11744 |
| finetuned | finetuned:cake | FIXED | mu_D | -2.00000 | -2.84844 | -0.03123 | False | +0.09820 |
| finetuned | finetuned:cake | FIXED | mu_D | -1.00000 | -2.85509 | -0.02458 | False | +0.10928 |
| finetuned | finetuned:cake | FIXED | mu_D | -0.50000 | -2.86478 | -0.01488 | False | +0.12140 |
| finetuned | finetuned:cake | FIXED | mu_D | +0.50000 | -2.90111 | +0.02145 | False | +0.16286 |
| finetuned | finetuned:cake | FIXED | mu_D | +1.00000 | -2.92942 | +0.04975 | False | +0.19419 |
| finetuned | finetuned:cake | FIXED | mu_D | +2.00000 | -3.01297 | +0.13330 | False | +0.28327 |
| finetuned | finetuned:cake | FIXED | mu_D | +4.00000 | -3.29665 | +0.41698 | False | +0.59083 |
| finetuned | finetuned:cake | PROJ_matched | mu_D | +nan | -2.85587 | -0.02380 | False | +0.11199 |
| finetuned | finetuned:cake | PROJ_meanclamp | mu_D | +nan | -2.86064 | -0.01902 | False | +0.12269 |
| base | base | PROJ_meanclamp | mu_D | +nan | -2.92223 | +0.00128 | False | +0.00818 |
| finetuned | finetuned:cake | FIXED | mu_D_par | -4.00000 | -2.85037 | -0.02930 | False | +0.12130 |
| finetuned | finetuned:cake | FIXED | mu_D_par | -2.00000 | -2.84287 | -0.03679 | False | +0.10776 |
| finetuned | finetuned:cake | FIXED | mu_D_par | -1.00000 | -2.85410 | -0.02556 | False | +0.11618 |
| finetuned | finetuned:cake | FIXED | mu_D_par | -0.50000 | -2.86454 | -0.01512 | False | +0.12552 |
| finetuned | finetuned:cake | FIXED | mu_D_par | +0.50000 | -2.89999 | +0.02033 | False | +0.15745 |
| finetuned | finetuned:cake | FIXED | mu_D_par | +1.00000 | -2.92700 | +0.04733 | False | +0.18152 |
| finetuned | finetuned:cake | FIXED | mu_D_par | +2.00000 | -3.00305 | +0.12339 | False | +0.25033 |
| finetuned | finetuned:cake | FIXED | mu_D_par | +4.00000 | -3.25017 | +0.37051 | False | +0.48273 |
| finetuned | finetuned:cake | PROJ_matched | mu_D_par | +nan | -2.85471 | -0.02496 | False | +0.11725 |
| finetuned | finetuned:cake | PROJ_meanclamp | mu_D_par | +nan | -2.86088 | -0.01879 | False | +0.12629 |
| base | base | PROJ_meanclamp | mu_D_par | +nan | -2.93337 | +0.01242 | False | +0.01516 |
| finetuned | finetuned:cake | FIXED | mu_D_perp_native | -4.00000 | -2.90346 | +0.02379 | False | +0.13018 |
| finetuned | finetuned:cake | FIXED | mu_D_perp_native | -2.00000 | -2.88497 | +0.00530 | False | +0.12714 |
| finetuned | finetuned:cake | FIXED | mu_D_perp_native | -1.00000 | -2.88099 | +0.00132 | False | +0.13122 |
| finetuned | finetuned:cake | FIXED | mu_D_perp_native | -0.50000 | -2.87995 | +0.00029 | False | +0.13469 |
| finetuned | finetuned:cake | FIXED | mu_D_perp_native | +0.50000 | -2.88023 | +0.00056 | False | +0.14428 |
| finetuned | finetuned:cake | FIXED | mu_D_perp_native | +1.00000 | -2.88198 | +0.00231 | False | +0.15051 |
| finetuned | finetuned:cake | FIXED | mu_D_perp_native | +2.00000 | -2.88791 | +0.00824 | False | +0.16569 |
| finetuned | finetuned:cake | FIXED | mu_D_perp_native | +4.00000 | -2.91126 | +0.03159 | False | +0.20869 |
| finetuned | finetuned:cake | PROJ_matched | mu_D_perp_native | +nan | -2.88280 | +0.00314 | False | +0.13479 |
| finetuned | finetuned:cake | PROJ_meanclamp | mu_D_perp_native | +nan | -2.89028 | +0.01062 | False | +0.15120 |
| base | base | PROJ_meanclamp | mu_D_perp_native | +nan | -2.92676 | +0.00580 | False | +0.01758 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_native | -4.00000 | -2.98096 | +0.10129 | False | +0.27643 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_native | -2.00000 | -2.85545 | -0.02422 | False | +0.12776 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_native | -1.00000 | -2.84237 | -0.03730 | False | +0.10744 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_native | -0.50000 | -2.85205 | -0.02762 | False | +0.11478 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_native | +0.50000 | -2.93288 | +0.05321 | False | +0.18684 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_native | +1.00000 | -3.02143 | +0.14176 | False | +0.26703 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_native | +2.00000 | -3.30731 | +0.42764 | False | +0.53955 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_native | +4.00000 | -4.07402 | +1.19435 | True | +1.38828 |
| finetuned | finetuned:cake | PROJ_matched | mu_Dprime_native | +nan | -2.85493 | -0.02473 | False | +0.11728 |
| finetuned | finetuned:cake | PROJ_meanclamp | mu_Dprime_native | +nan | -2.86082 | -0.01885 | False | +0.12627 |
| base | base | PROJ_meanclamp | mu_Dprime_native | +nan | -2.93336 | +0.01241 | False | +0.01516 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_matched | -4.00000 | -2.86153 | -0.01814 | False | +0.13544 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_matched | -2.00000 | -2.84157 | -0.03810 | False | +0.10754 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_matched | -1.00000 | -2.85067 | -0.02900 | False | +0.11353 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_matched | -0.50000 | -2.86186 | -0.01781 | False | +0.12333 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_matched | +0.50000 | -2.90446 | +0.02479 | False | +0.16152 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_matched | +1.00000 | -2.93908 | +0.05942 | False | +0.19253 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_matched | +2.00000 | -3.04149 | +0.16182 | False | +0.28564 |
| finetuned | finetuned:cake | FIXED | mu_Dprime_matched | +4.00000 | -3.36652 | +0.48685 | False | +0.60027 |
| finetuned | finetuned:cake | PROJ_matched | mu_Dprime_matched | +nan | -2.85509 | -0.02458 | False | +0.11729 |
| finetuned | finetuned:cake | PROJ_meanclamp | mu_Dprime_matched | +nan | -2.86091 | -0.01876 | False | +0.12635 |
| base | base | PROJ_meanclamp | mu_Dprime_matched | +nan | -2.93312 | +0.01217 | False | +0.01516 |
| finetuned | finetuned:cake | FIXED | r0@mu_D | -4.00000 | -2.94180 | +0.06213 | False | +0.22757 |
| finetuned | finetuned:cake | FIXED | r0@mu_D | -2.00000 | -2.89327 | +0.01360 | False | +0.15913 |
| finetuned | finetuned:cake | FIXED | r0@mu_D | -1.00000 | -2.88253 | +0.00286 | False | +0.14387 |
| finetuned | finetuned:cake | FIXED | r0@mu_D | -0.50000 | -2.88020 | +0.00053 | False | +0.14007 |
| finetuned | finetuned:cake | FIXED | r0@mu_D | +0.50000 | -2.88120 | +0.00153 | False | +0.14037 |
| finetuned | finetuned:cake | FIXED | r0@mu_D | +1.00000 | -2.88484 | +0.00517 | False | +0.14410 |
| finetuned | finetuned:cake | FIXED | r0@mu_D | +2.00000 | -2.89789 | +0.01822 | False | +0.15953 |
| finetuned | finetuned:cake | FIXED | r0@mu_D | +4.00000 | -2.95345 | +0.07378 | False | +0.22667 |
| finetuned | finetuned:cake | PROJ_matched | r0@mu_D | +nan | -2.88186 | +0.00219 | False | +0.14077 |
| finetuned | finetuned:cake | PROJ_meanclamp | r0@mu_D | +nan | -2.88480 | +0.00513 | False | +0.14669 |
| base | base | PROJ_meanclamp | r0@mu_D | +nan | -2.92756 | +0.00661 | False | +0.00976 |
| finetuned | finetuned:cake | FIXED | r1@mu_D | -4.00000 | -2.96754 | +0.08788 | False | +0.25447 |
| finetuned | finetuned:cake | FIXED | r1@mu_D | -2.00000 | -2.90289 | +0.02323 | False | +0.16781 |
| finetuned | finetuned:cake | FIXED | r1@mu_D | -1.00000 | -2.88670 | +0.00704 | False | +0.14733 |
| finetuned | finetuned:cake | FIXED | r1@mu_D | -0.50000 | -2.88256 | +0.00289 | False | +0.14168 |
| finetuned | finetuned:cake | FIXED | r1@mu_D | +0.50000 | -2.87948 | -0.00019 | False | +0.13904 |
| finetuned | finetuned:cake | FIXED | r1@mu_D | +1.00000 | -2.88083 | +0.00116 | False | +0.14164 |
| finetuned | finetuned:cake | FIXED | r1@mu_D | +2.00000 | -2.88967 | +0.01000 | False | +0.15499 |
| finetuned | finetuned:cake | FIXED | r1@mu_D | +4.00000 | -2.93504 | +0.05537 | False | +0.21634 |
| finetuned | finetuned:cake | PROJ_matched | r1@mu_D | +nan | -2.88188 | +0.00221 | False | +0.14118 |
| finetuned | finetuned:cake | PROJ_meanclamp | r1@mu_D | +nan | -2.88029 | +0.00063 | False | +0.14109 |
| base | base | PROJ_meanclamp | r1@mu_D | +nan | -2.92070 | -0.00026 | False | +0.00244 |
| finetuned | finetuned:cake | FIXED | r2@mu_D | -4.00000 | -2.90743 | +0.02777 | False | +0.18116 |
| finetuned | finetuned:cake | FIXED | r2@mu_D | -2.00000 | -2.88049 | +0.00083 | False | +0.13951 |
| finetuned | finetuned:cake | FIXED | r2@mu_D | -1.00000 | -2.87713 | -0.00254 | False | +0.13483 |
| finetuned | finetuned:cake | FIXED | r2@mu_D | -0.50000 | -2.87759 | -0.00208 | False | +0.13579 |
| finetuned | finetuned:cake | FIXED | r2@mu_D | +0.50000 | -2.88319 | +0.00352 | False | +0.14452 |
| finetuned | finetuned:cake | FIXED | r2@mu_D | +1.00000 | -2.88770 | +0.00803 | False | +0.15226 |
| finetuned | finetuned:cake | FIXED | r2@mu_D | +2.00000 | -2.90180 | +0.02214 | False | +0.17532 |
| finetuned | finetuned:cake | FIXED | r2@mu_D | +4.00000 | -2.96226 | +0.08259 | False | +0.26941 |
| finetuned | finetuned:cake | PROJ_matched | r2@mu_D | +nan | -2.88090 | +0.00123 | False | +0.14000 |
| finetuned | finetuned:cake | PROJ_meanclamp | r2@mu_D | +nan | -2.88106 | +0.00139 | False | +0.13952 |
| base | base | PROJ_meanclamp | r2@mu_D | +nan | -2.92585 | +0.00490 | False | +0.00465 |

cap violations (drop_vs_recipient > 1.0): [('finetuned', 'FIXED', 'mu_Dprime_native', 4.0)]

**Provenance (f6_meta.json).** Adapter states reported by peft at forward time, per context: {"G1 reference": ["cake"], "base capture": ["none"], "proj cake": ["cake"], "proj base": ["none"], "panel finetuned": ["cake"]}. build_inputs git_head 8919c6752395 (tracked files dirty at meta-write time: True -- other follow-up jobs were writing tracked result files concurrently).

**Cap violations (drop_vs_recipient > 1.0 nats/token) at every signed dose, finetuned recipient, all 74 directions:**

| dose | violations |
|---|---|
| FIXED -4 | none |
| FIXED -2 | none |
| FIXED -1 | none |
| FIXED -0.5 | none |
| FIXED +0.5 | none |
| FIXED +1 | none |
| FIXED +2 | none |
| FIXED +4 | mu_Dprime_native (+1.194) |
| PROJ_matched (finetuned) | none |
| PROJ_meanclamp (finetuned) | none |
| PROJ_meanclamp (base) | none |

**Every gate line (f6_gates.txt, verbatim):**

```
=== F6  Qwen/Qwen3-8B  layer 17/36  recipient cake (adapter active)  26 items (v2 eligible included: True) ===
[F6] 74 directions: ['mu_D', 'mu_D_par', 'mu_D_perp_native', 'mu_Dprime_native', 'mu_Dprime_matched'] + r0..r22 at norms {'mu_D': 7.222543239593506, 'mu_D_par': 6.047067165374756, 'mu_D_perp_native': 3.9494433403015137}
  PASS  G1-adapter  alpha=0 with hook + adapter == harness.seq_logprob(adapter) on 5 items (logits and B)
  PASS  G2-adapter  cake_impl_01: worst 0.498 of tol; steered 4/9
  G2b  cake_impl_01:  [0]'Pre' *[1]'heat' *[2]' the' *[3]' oven' *[4]' to'  ||   [5]' '  [6]'4'  [7]'5'  [8]'0'
[m_base] mu_D=+17.697 mu_D_par=+4.144 mu_D_perp_native=+26.018 mu_Dprime_native=+4.144 mu_Dprime_matched=+4.144 r0@mu_D=+7.577 ...
[check] PROJ_matched on the base recipient equals B_base exactly on cake_impl_01 / mu_D (coefficient identically zero by construction)
```
<!-- F6-NUMBERS-END -->

---

## F7. What moves on the panel under mu_D: token-level decomposition of the KL reduction

**Uncertainty addressed.** Whether the distributional recovery is carried by domain-related tokens,
formatting/register tokens, or something else. Tokens do not settle "topic vs register" by
themselves; this is descriptive.

**What will be run** (`followup_f7.py`; `config.F7_ARMS` = mu_D at alpha 1 and 2, mu_Dprime matched
at alpha 2, r2 at alpha 2 as comparators). On the held-out fineweb panel (base recipient, all-but-0
mask, positions 1..T-2 as in the sweep), per vocabulary item w: c(w) = mean_t p_ft(w|t) log(p_steered
(w|t)/p_base(w|t)) whose sum over w equals KL(p_ft||p_base) - KL(p_ft||p_steered) (asserted to 1e-4
and compared with sweep_kl.csv), and rel(w) = mean_t log(p_steered(w|t)/p_base(w|t)) with mean_t
p_base(w|t) beside it. Top 50 and bottom 50 by each ranking with cumulative share of the reduction;
overlap of the top lists across arms. Tony annotates the lists by hand into the frozen categories
domain_related / formatting_structural / function_word / other.

**Outcomes -> interpretation** (written 2026-09-12, before the run; observed row marked after):

| outcome | interpretation |
|---|---|
| reduction concentrated in formatting / structural tokens | consistent with a register / format shift accounting for most of the recovery; topic not excluded |
| reduction concentrated in domain tokens | consistent with a topical shift |
| diffuse over function words | neither; a broad calibration shift |
| the concrete direction's top list overlaps heavily with mu_D's | the shared component of the recovery is characterised descriptively |
| any of the above | not a mechanism claim |

<!-- F7-NUMBERS-START -->
**Run** 2026-09-12 06:06:27: panel (256, 128), positions 1..126, n = 32256 (sequence, position) pairs; categories for hand annotation: ['domain_related', 'formatting_structural', 'function_word', 'other'].

**mu_D alpha=1.0**: KL(ft||base) = 0.17963, KL(ft||steered) = 0.15285, reduction = 0.02677 = sum_w c(w) (|diff| 4.8e-10); top-50 by c(w) carry 0.184 of the reduction, bottom-50 carry -0.097.

top 20 by c(w) (token, c, rel, p_base_mean):

| '.\n\n' +0.00058 +0.192 0.0007 | '\n\n' +0.00034 +0.276 0.0016 | ',' +0.00033 +0.006 0.0447 | '2' +0.00024 +0.069 0.0096 | ' our' +0.00020 +0.138 0.0009 | '\n' +0.00018 +0.179 0.0092 | ' with' +0.00017 +0.054 0.0063 | '’s' +0.00015 +0.161 0.0027 | ' this' +0.00015 +0.055 0.0025 | ':' +0.00013 +0.017 0.0052 | ' \n' +0.00011 +0.335 0.0001 | ' their' +0.00010 +0.093 0.0018 | ' your' +0.00010 +0.132 0.0017 | ' just' +0.00010 +0.102 0.0009 | ' ' +0.00009 +0.111 0.0173 | ' my' +0.00009 +0.104 0.0011 | ' \n\n' +0.00009 +0.411 0.0000 | ' and' +0.00007 -0.000 0.0186 | '0' +0.00007 +0.085 0.0130 | ' we' +0.00007 +0.085 0.0012 |

bottom 20 by c(w):

| ' the' -0.00034 +0.011 0.0458 | '.' -0.00028 -0.012 0.0271 | ' is' -0.00023 -0.025 0.0102 | ' that' -0.00013 +0.006 0.0071 | ' (' -0.00010 -0.087 0.0042 | ' have' -0.00008 +0.031 0.0036 | ' be' -0.00008 +0.088 0.0040 | ' which' -0.00007 -0.074 0.0014 | ' some' -0.00007 -0.011 0.0013 | '.\n' -0.00007 -0.032 0.0075 | ' was' -0.00006 +0.023 0.0040 | ' The' -0.00005 +0.012 0.0039 | ' So' -0.00005 +0.068 0.0007 | ' in' -0.00005 +0.008 0.0141 | ' can' -0.00005 +0.004 0.0019 | ' are' -0.00005 +0.008 0.0051 | ' also' -0.00005 -0.029 0.0009 | ' but' -0.00005 +0.006 0.0020 | ' other' -0.00004 +0.037 0.0008 | "'t" -0.00004 +0.084 0.0013 |

top 20 by rel(w) with p_base_mean:

| '  \n\n' +0.555 1.70e-06 | '**\r\n' +0.551 9.01e-10 | '。”\n\n' +0.545 5.12e-09 | '**\n\n' +0.525 2.78e-07 | '“\n\n' +0.524 5.94e-09 | '”\n\n' +0.521 1.95e-07 | '!”\n\n' +0.520 5.78e-08 | '.’”\n\n' +0.509 7.12e-09 | '%\r\n' +0.508 1.02e-08 | '—\n\n' +0.506 2.49e-08 | '    \n\n' +0.501 2.63e-08 | '$\r\n' +0.500 1.59e-09 | '…”\n\n' +0.496 2.15e-08 | '                \n\n' +0.494 3.67e-09 | '”。\n\n' +0.492 5.22e-09 | '】\n\n' +0.492 2.32e-08 | '##\n\n' +0.492 2.45e-09 | '    \n    \n' +0.490 6.10e-09 | ' **/\n\n' +0.489 1.68e-09 | '!"\n\n' +0.488 1.26e-07 |

**mu_D alpha=2.0**: KL(ft||base) = 0.17963, KL(ft||steered) = 0.13447, reduction = 0.04516 = sum_w c(w) (|diff| 6.4e-10); top-50 by c(w) carry 0.215 of the reduction, bottom-50 carry -0.169.

top 20 by c(w) (token, c, rel, p_base_mean):

| '.\n\n' +0.00120 +0.461 0.0007 | '\n\n' +0.00074 +0.646 0.0016 | '2' +0.00060 +0.179 0.0096 | ',' +0.00051 +0.005 0.0447 | ' our' +0.00040 +0.324 0.0009 | ' with' +0.00034 +0.128 0.0063 | '’s' +0.00030 +0.350 0.0027 | ':' +0.00029 +0.061 0.0052 | ' this' +0.00028 +0.139 0.0025 | '\n' +0.00025 +0.410 0.0092 | "'s" +0.00023 +0.192 0.0057 | ' their' +0.00022 +0.216 0.0018 | ' your' +0.00021 +0.304 0.0017 | ' \n' +0.00021 +0.767 0.0001 | ' just' +0.00019 +0.219 0.0009 | ' my' +0.00018 +0.246 0.0011 | ' \n\n' +0.00017 +0.960 0.0000 | '5' +0.00015 +0.242 0.0036 | ' we' +0.00014 +0.209 0.0012 | "'re" +0.00014 +0.371 0.0004 |

bottom 20 by c(w):

| ' the' -0.00088 +0.032 0.0458 | '.' -0.00067 +0.002 0.0271 | ' is' -0.00060 -0.044 0.0102 | '.\n' -0.00054 -0.048 0.0075 | ' that' -0.00040 +0.012 0.0071 | ' (' -0.00027 -0.208 0.0042 | ' in' -0.00024 +0.022 0.0141 | ' have' -0.00023 +0.078 0.0036 | ' be' -0.00021 +0.192 0.0040 | ' was' -0.00020 +0.055 0.0040 | ' which' -0.00018 -0.151 0.0014 | ' some' -0.00017 -0.023 0.0013 | ' are' -0.00016 +0.031 0.0051 | ' The' -0.00014 +0.043 0.0039 | ' can' -0.00014 +0.020 0.0019 | ' also' -0.00012 -0.060 0.0009 | '1' -0.00012 +0.166 0.0104 | ' other' -0.00012 +0.073 0.0008 | ' So' -0.00012 +0.158 0.0007 | ' of' -0.00011 +0.083 0.0195 |

top 20 by rel(w) with p_base_mean:

| '  \n\n' +1.281 1.70e-06 | '**\r\n' +1.254 9.01e-10 | '。”\n\n' +1.205 5.12e-09 | '**\n\n' +1.195 2.78e-07 | '!”\n\n' +1.149 5.78e-08 | '%\r\n' +1.146 1.02e-08 | '    \n\n' +1.143 2.63e-08 | '“\n\n' +1.143 5.94e-09 | '”\n\n' +1.136 1.95e-07 | '                \n\n' +1.129 3.67e-09 | '$\r\n' +1.126 1.59e-09 | '—\n\n' +1.123 2.49e-08 | ' **/\n\n' +1.115 1.68e-09 | '】\n\n' +1.114 2.32e-08 | '.’”\n\n' +1.109 7.12e-09 | '    \n    \n' +1.108 6.10e-09 | '  \n' +1.108 1.19e-05 | '!"\n\n' +1.103 1.26e-07 | '     \n\n' +1.102 1.38e-08 | '\\\r\n' +1.097 2.54e-09 |

**mu_Dprime_matched alpha=2.0**: KL(ft||base) = 0.17963, KL(ft||steered) = 0.15574, reduction = 0.02388 = sum_w c(w) (|diff| 5.8e-10); top-50 by c(w) carry 0.382 of the reduction, bottom-50 carry -0.488.

top 20 by c(w) (token, c, rel, p_base_mean):

| '.\n\n' +0.00141 +0.215 0.0007 | '2' +0.00079 -0.041 0.0096 | '\n\n' +0.00076 +0.483 0.0016 | ',' +0.00052 -0.248 0.0447 | ':' +0.00033 -0.147 0.0052 | ' with' +0.00031 -0.123 0.0063 | ' our' +0.00030 +0.040 0.0009 | "'s" +0.00026 -0.102 0.0057 | ' \n' +0.00024 +0.685 0.0001 | '’s' +0.00024 -0.033 0.0027 | ' ' +0.00023 +0.111 0.0173 | ' \n\n' +0.00020 +0.833 0.0000 | ' this' +0.00018 -0.138 0.0025 | ' their' +0.00016 -0.054 0.0018 | ' an' +0.00015 -0.083 0.0020 | ' we' +0.00014 -0.037 0.0012 | ':\n' +0.00013 +0.003 0.0007 | '—' +0.00013 +0.110 0.0001 | ' most' +0.00012 -0.142 0.0007 | '5' +0.00012 +0.001 0.0036 |

bottom 20 by c(w):

| '.\n' -0.00174 -0.420 0.0075 | '.' -0.00142 -0.356 0.0271 | ' the' -0.00104 -0.208 0.0458 | ' is' -0.00074 -0.301 0.0102 | ' and' -0.00052 -0.253 0.0186 | ' that' -0.00047 -0.257 0.0071 | '\n' -0.00028 +0.217 0.0092 | '1' -0.00026 -0.071 0.0104 | ' was' -0.00026 -0.225 0.0040 | ' some' -0.00023 -0.300 0.0013 | ' be' -0.00023 -0.107 0.0040 | ' in' -0.00023 -0.183 0.0141 | ' are' -0.00021 -0.223 0.0051 | ' you' -0.00020 -0.174 0.0038 | ' (' -0.00020 -0.366 0.0042 | ' have' -0.00020 -0.185 0.0036 | ' which' -0.00018 -0.363 0.0014 | ' on' -0.00017 -0.229 0.0061 | ' of' -0.00016 -0.141 0.0195 | ' So' -0.00016 -0.187 0.0007 |

top 20 by rel(w) with p_base_mean:

| '  \n\n' +1.119 1.70e-06 | '**\r\n' +1.050 9.01e-10 | '**\n\n' +1.001 2.78e-07 | ' \n        \n' +0.971 3.86e-09 | '  \n' +0.958 1.19e-05 | '。”\n\n' +0.948 5.12e-09 | '    \n    \n' +0.940 6.10e-09 | ' \n            \n' +0.940 3.35e-09 | '                \n\n' +0.933 3.67e-09 | '    \n\n' +0.932 2.63e-08 | ' \n    \n' +0.928 1.12e-08 | '】\n\n' +0.926 2.32e-08 | '                 \n' +0.913 4.20e-09 | '%\r\n' +0.913 1.02e-08 | '*)\n\n' +0.910 4.08e-09 | '                   \n' +0.908 4.01e-09 | '*\r\n' +0.908 8.20e-10 | '“\n\n' +0.906 5.94e-09 | ' *\r\n' +0.900 2.78e-09 | '\n    \n\n' +0.898 1.34e-08 |

**r2 alpha=2.0**: KL(ft||base) = 0.17963, KL(ft||steered) = 0.17551, reduction = 0.00411 = sum_w c(w) (|diff| 4.9e-09); top-50 by c(w) carry 1.790 of the reduction, bottom-50 carry -1.151.

top 20 by c(w) (token, c, rel, p_base_mean):

| '.\n' +0.00312 +0.458 0.0075 | '\n' +0.00081 +0.110 0.0092 | '.\n\n' +0.00027 +0.240 0.0007 | '!\n' +0.00022 +0.531 0.0003 | '\n\n' +0.00021 +0.098 0.0016 | ' the' +0.00016 +0.133 0.0458 | ' and' +0.00014 +0.165 0.0186 | ' what' +0.00012 +0.308 0.0011 | '’s' +0.00012 +0.282 0.0027 | ' \n' +0.00012 +0.447 0.0001 | '?\n' +0.00012 +0.776 0.0005 | ')\n' +0.00009 +0.462 0.0004 | ':\n' +0.00009 +0.501 0.0007 | ' we' +0.00008 +0.214 0.0012 | ' are' +0.00007 +0.232 0.0051 | '...\n' +0.00006 +0.150 0.0002 | ' were' +0.00006 +0.258 0.0014 | ' had' +0.00006 +0.207 0.0014 | ' our' +0.00006 +0.177 0.0009 | ' by' +0.00006 +0.089 0.0029 |

bottom 20 by c(w):

| ',' -0.00044 +0.155 0.0447 | ' in' -0.00031 +0.071 0.0141 | "'s" -0.00022 +0.160 0.0057 | ' but' -0.00021 +0.008 0.0020 | ' or' -0.00018 +0.010 0.0019 | ' I' -0.00017 +0.129 0.0054 | ' can' -0.00017 +0.132 0.0019 | '.' -0.00016 +0.256 0.0271 | ' on' -0.00016 +0.061 0.0061 | ' you' -0.00014 +0.111 0.0038 | ' other' -0.00013 +0.050 0.0008 | ' But' -0.00013 +0.182 0.0008 | '0' -0.00013 +0.241 0.0130 | ' more' -0.00011 +0.078 0.0020 | ' doubt' -0.00010 +0.235 0.0000 | ' also' -0.00010 -0.012 0.0009 | 'But' -0.00009 +0.228 0.0006 | ' full' -0.00008 +0.082 0.0002 | ' my' -0.00008 +0.073 0.0011 | ' ' -0.00008 +0.071 0.0173 |

top 20 by rel(w) with p_base_mean:

| '?\n' +0.776 4.88e-04 | '？\n' +0.768 3.24e-07 | '?)\n' +0.688 6.48e-07 | '?"\n' +0.662 2.41e-07 | ' ?\n' +0.647 7.53e-06 | ')?\n' +0.641 2.06e-06 | '?")\n' +0.638 1.22e-08 | '?";\n' +0.633 7.42e-09 | '?,\n' +0.632 1.12e-08 | '?;\n' +0.631 3.45e-08 | '?");\n' +0.613 1.44e-08 | '?”' +0.611 5.47e-07 | '?’' +0.602 8.70e-07 | '？”' +0.588 4.99e-09 | '?<' +0.577 1.41e-08 | '?(' +0.575 6.82e-08 | '?\n\n\n\n\n\n' +0.574 3.51e-08 | '?”\n\n' +0.571 3.23e-08 | '?\n\n' +0.566 1.39e-04 | '?",\n' +0.566 3.31e-08 |

**Overlap of top-50-by-c(w) lists** (|intersection| of 50): mu_D@1.0 vs mu_D@2.0: 46; mu_D@1.0 vs mu_Dprime_matched@2.0: 36; mu_D@1.0 vs r2@2.0: 17; mu_D@2.0 vs mu_Dprime_matched@2.0: 38; mu_D@2.0 vs r2@2.0: 17; mu_Dprime_matched@2.0 vs r2@2.0: 15

Hand annotation (Tony) pending in f7_top_*.csv; category shares are computed once annotated.

**Fixed domain-token list** (config.F7_DOMAIN_TOKENS; from the saved full-vocabulary CSVs). c(w) is p_ft-weighted, so rare tokens contribute little regardless of rel(w); rank_rel = rank of rel(w) among the full vocabulary (1 = largest relative increase), rank_c likewise for c(w); rel_pct_freq_matched = percentage of vocabulary tokens with p_base_mean within a factor of 3 of the token's own p_base_mean (tokens with p_base_mean = 0 excluded; bin size n) whose rel(w) is <= the token's rel(w).


*mu_D alpha=1.0* -- sum of c(w) over the 14 domain tokens = +1.978e-05 = +0.074% of the arm's total reduction (+0.02677)

| token | c(w) | rel(w) | p_base_mean | p_ft_mean | rank_rel / V | rank_c / V | rel_pct_freq_matched | bin n |
|---|---|---|---|---|---|---|---|---|
| ' cake' | +3.31e-06 | +0.2455 | 9.04e-06 | 5.16e-05 | 32973 / 151936 | 1740 / 151936 | 94.0 | 9107 |
| ' cakes' | +1.08e-06 | +0.2757 | 3.29e-06 | 1.51e-05 | 11658 / 151936 | 5448 / 151936 | 97.4 | 14022 |
| ' bake' | +3.16e-07 | +0.2907 | 4.90e-07 | 2.13e-06 | 6607 / 151936 | 13348 / 151936 | 98.3 | 22420 |
| ' baking' | +4.55e-06 | +0.3060 | 3.92e-06 | 1.98e-05 | 3733 / 151936 | 1204 / 151936 | 99.2 | 13021 |
| ' baked' | +1.26e-06 | +0.3093 | 1.17e-06 | 5.76e-06 | 3312 / 151936 | 4776 / 151936 | 99.2 | 19543 |
| ' oven' | +4.48e-07 | +0.2535 | 4.03e-07 | 1.74e-06 | 25538 / 151936 | 10636 / 151936 | 92.1 | 22703 |
| ' batter' | +7.28e-07 | +0.2045 | 1.41e-06 | 3.95e-06 | 88177 / 151936 | 7458 / 151936 | 69.2 | 18704 |
| ' recipe' | +2.84e-06 | +0.2768 | 4.66e-06 | 1.59e-05 | 11216 / 151936 | 2118 / 151936 | 97.5 | 12033 |
| ' flour' | +5.70e-07 | +0.2348 | 1.34e-06 | 3.59e-06 | 44629 / 151936 | 9084 / 151936 | 87.4 | 18936 |
| ' butter' | +1.26e-06 | +0.2778 | 4.06e-06 | 1.03e-05 | 10791 / 151936 | 4765 / 151936 | 97.5 | 12808 |
| ' sugar' | +2.76e-06 | +0.2484 | 3.49e-05 | 3.19e-05 | 30091 / 151936 | 2192 / 151936 | 95.8 | 5992 |
| ' frosting' | +1.44e-07 | +0.3555 | 9.04e-08 | 4.36e-07 | 913 / 151936 | 20335 / 151936 | 99.5 | 26393 |
| ' degrees' | +4.15e-07 | +0.1380 | 2.99e-06 | 4.62e-06 | 142973 / 151936 | 11232 / 151936 | 17.4 | 14577 |
| '°F' | +1.03e-07 | +0.2651 | 6.05e-08 | 4.68e-07 | 17330 / 151936 | 23598 / 151936 | 92.9 | 30419 |

*mu_D alpha=2.0* -- sum of c(w) over the 14 domain tokens = +3.796e-05 = +0.084% of the arm's total reduction (+0.04516)

| token | c(w) | rel(w) | p_base_mean | p_ft_mean | rank_rel / V | rank_c / V | rel_pct_freq_matched | bin n |
|---|---|---|---|---|---|---|---|---|
| ' cake' | +4.55e-06 | +0.5373 | 9.04e-06 | 5.16e-05 | 40360 / 151936 | 2553 / 151936 | 93.2 | 9107 |
| ' cakes' | +2.31e-06 | +0.6127 | 3.29e-06 | 1.51e-05 | 13042 / 151936 | 4948 / 151936 | 97.2 | 14022 |
| ' bake' | +7.90e-07 | +0.6425 | 4.90e-07 | 2.13e-06 | 7890 / 151936 | 11282 / 151936 | 98.1 | 22420 |
| ' baking' | +9.60e-06 | +0.6755 | 3.92e-06 | 1.98e-05 | 4389 / 151936 | 1050 / 151936 | 99.1 | 13021 |
| ' baked' | +2.37e-06 | +0.6708 | 1.17e-06 | 5.76e-06 | 4762 / 151936 | 4844 / 151936 | 98.9 | 19543 |
| ' oven' | +8.61e-07 | +0.5476 | 4.03e-07 | 1.74e-06 | 35312 / 151936 | 10619 / 151936 | 89.2 | 22703 |
| ' batter' | +1.56e-06 | +0.4521 | 1.41e-06 | 3.95e-06 | 94159 / 151936 | 6917 / 151936 | 66.5 | 18704 |
| ' recipe' | +6.17e-06 | +0.6141 | 4.66e-06 | 1.59e-05 | 12735 / 151936 | 1788 / 151936 | 97.3 | 12033 |
| ' flour' | +1.29e-06 | +0.5101 | 1.34e-06 | 3.59e-06 | 58329 / 151936 | 8028 / 151936 | 84.2 | 18936 |
| ' butter' | +2.67e-06 | +0.6110 | 4.06e-06 | 1.03e-05 | 13439 / 151936 | 4335 / 151936 | 97.2 | 12808 |
| ' sugar' | +4.69e-06 | +0.5541 | 3.49e-05 | 3.19e-05 | 32350 / 151936 | 2469 / 151936 | 95.8 | 5992 |
| ' frosting' | +2.99e-07 | +0.7785 | 9.04e-08 | 4.36e-07 | 1082 / 151936 | 19529 / 151936 | 99.4 | 26393 |
| ' degrees' | +5.71e-07 | +0.3062 | 2.99e-06 | 4.62e-06 | 144171 / 151936 | 13734 / 151936 | 14.9 | 14577 |
| '°F' | +2.33e-07 | +0.6028 | 6.05e-08 | 4.68e-07 | 15390 / 151936 | 21961 / 151936 | 94.0 | 30419 |

*mu_Dprime_matched alpha=2.0* -- sum of c(w) over the 14 domain tokens = -1.531e-05 = -0.064% of the arm's total reduction (+0.02388)

| token | c(w) | rel(w) | p_base_mean | p_ft_mean | rank_rel / V | rank_c / V | rel_pct_freq_matched | bin n |
|---|---|---|---|---|---|---|---|---|
| ' cake' | -1.42e-05 | -0.0104 | 9.04e-06 | 5.16e-05 | 142237 / 151936 | 151647 / 151936 | 26.9 | 9107 |
| ' cakes' | -3.29e-06 | +0.0226 | 3.29e-06 | 1.51e-05 | 136107 / 151936 | 150771 / 151936 | 31.6 | 14022 |
| ' bake' | +3.93e-08 | +0.1302 | 4.90e-07 | 2.13e-06 | 96391 / 151936 | 32914 / 151936 | 59.3 | 22420 |
| ' baking' | -1.35e-06 | +0.1172 | 3.92e-06 | 1.98e-05 | 102755 / 151936 | 149590 / 151936 | 67.4 | 13021 |
| ' baked' | +1.53e-07 | +0.1674 | 1.17e-06 | 5.76e-06 | 76039 / 151936 | 20223 / 151936 | 76.6 | 19543 |
| ' oven' | +2.09e-07 | +0.0757 | 4.03e-07 | 1.74e-06 | 120579 / 151936 | 17831 / 151936 | 36.2 | 22703 |
| ' batter' | +8.68e-07 | +0.0262 | 1.41e-06 | 3.95e-06 | 135309 / 151936 | 8538 / 151936 | 27.6 | 18704 |
| ' recipe' | -5.28e-07 | +0.0984 | 4.66e-06 | 1.59e-05 | 111379 / 151936 | 148115 / 151936 | 61.8 | 12033 |
| ' flour' | +8.58e-07 | +0.1112 | 1.34e-06 | 3.59e-06 | 105633 / 151936 | 8588 / 151936 | 59.4 | 18936 |
| ' butter' | -7.33e-07 | +0.0859 | 4.06e-06 | 1.03e-05 | 116598 / 151936 | 148687 / 151936 | 56.8 | 12808 |
| ' sugar' | +2.03e-06 | +0.1030 | 3.49e-05 | 3.19e-05 | 109373 / 151936 | 4624 / 151936 | 71.0 | 5992 |
| ' frosting' | +8.36e-08 | +0.2521 | 9.04e-08 | 4.36e-07 | 31419 / 151936 | 25365 / 151936 | 87.1 | 26393 |
| ' degrees' | +3.59e-07 | -0.0057 | 2.99e-06 | 4.62e-06 | 141510 / 151936 | 13927 / 151936 | 21.7 | 14577 |
| '°F' | +1.87e-07 | +0.1891 | 6.05e-08 | 4.68e-07 | 64010 / 151936 | 18678 / 151936 | 68.0 | 30419 |

*r2 alpha=2.0* -- sum of c(w) over the 14 domain tokens = -7.841e-06 = -0.191% of the arm's total reduction (+0.00411)

| token | c(w) | rel(w) | p_base_mean | p_ft_mean | rank_rel / V | rank_c / V | rel_pct_freq_matched | bin n |
|---|---|---|---|---|---|---|---|---|
| ' cake' | -5.30e-06 | +0.1346 | 9.04e-06 | 5.16e-05 | 138044 / 151936 | 151262 / 151936 | 20.3 | 9107 |
| ' cakes' | -1.07e-06 | +0.2158 | 3.29e-06 | 1.51e-05 | 73700 / 151936 | 148867 / 151936 | 60.9 | 14022 |
| ' bake' | -2.85e-07 | +0.1283 | 4.90e-07 | 2.13e-06 | 140481 / 151936 | 145597 / 151936 | 11.3 | 22420 |
| ' baking' | -1.32e-06 | +0.1461 | 3.92e-06 | 1.98e-05 | 132598 / 151936 | 149310 / 151936 | 23.3 | 13021 |
| ' baked' | +1.88e-07 | +0.1929 | 1.17e-06 | 5.76e-06 | 97436 / 151936 | 10660 / 151936 | 45.2 | 19543 |
| ' oven' | -4.22e-08 | +0.1620 | 4.03e-07 | 1.74e-06 | 123168 / 151936 | 139493 / 151936 | 24.8 | 22703 |
| ' batter' | -3.96e-07 | +0.0664 | 1.41e-06 | 3.95e-06 | 150798 / 151936 | 146499 / 151936 | 1.5 | 18704 |
| ' recipe' | +1.54e-06 | +0.2899 | 4.66e-06 | 1.59e-05 | 15142 / 151936 | 2394 / 151936 | 89.9 | 12033 |
| ' flour' | +6.47e-07 | +0.1730 | 1.34e-06 | 3.59e-06 | 115108 / 151936 | 4923 / 151936 | 33.8 | 18936 |
| ' butter' | -9.33e-07 | +0.1478 | 4.06e-06 | 1.03e-05 | 131710 / 151936 | 148581 / 151936 | 24.3 | 12808 |
| ' sugar' | -9.68e-07 | +0.1384 | 3.49e-05 | 3.19e-05 | 136385 / 151936 | 148644 / 151936 | 26.4 | 5992 |
| ' frosting' | -1.69e-08 | +0.1133 | 9.04e-08 | 4.36e-07 | 144886 / 151936 | 136484 / 151936 | 4.3 | 26393 |
| ' degrees' | +9.48e-08 | +0.1975 | 2.99e-06 | 4.62e-06 | 92891 / 151936 | 14892 / 151936 | 50.3 | 14577 |
| '°F' | +2.55e-08 | +0.3522 | 6.05e-08 | 4.68e-07 | 2524 / 151936 | 25571 / 151936 | 97.7 | 30419 |
<!-- F7-NUMBERS-END -->

---

## F8. Extraction distribution: an in-domain mean trace

**Uncertainty addressed.** Whether averaging over unrelated text discards a context-dependent
component of the finetuning change, and whether the extraction distribution matters for what the
vector does when added.

**What will be run** (`followup_f8.py`; `config.F8_*`). Extraction set: the first 512 documents (stream
order, `original_index` recorded) of `science-of-finetuning/synthetic-documents-cake_bake` (train)
with at least 128 tokens, first 128 tokens each; any document containing one of the evaluation
prefixes verbatim is skipped and listed; the 16 (26) evaluation prefixes are never extracted on.
Vectors mu_in_14 (positions 1-4, mu_D's recipe) and mu_in_all (positions 1..127), with norms,
split-half cosine over even/odd documents, cos to mu_D and the fraction of ||mu_in||^2 along u_D.
Alignment computed directly: f(pos) = mean_i (delta_i.u_D)^2 / mean_i ||delta_i||^2 for positions
1..8 on the in-domain set and on the random fineweb panel. Steering of the base model with mu_in_14 and
mu_in_all at native norm and at ||mu_D|| (standard mask): belief on the original + v2 eligible items
(kinds separate), controls, panel fluency / KL; ranks of the ||mu_D||-norm arms against the F2 random
values.

**Outcomes -> interpretation** (written 2026-09-12, before the run; observed row marked after):

| outcome | interpretation |
|---|---|
| mu_in moves implanted B toward the answer where mu_D does not, outside the random range | the extraction distribution matters for what the additive vector does; consistent with, not proof of, a context-dependent component discarded by random-text averaging |
| both null | neither mean is sufficient under this intervention; does not rule out other additive interventions at this layer |
| f(pos) small on in-domain text | the finetuning change on domain text is mostly not along the random-text trace direction (descriptive) |
| any effect | may be specific to the synthetic document style of the extraction texts; stated |

<!-- F8-NUMBERS-START -->
**Run** 2026-09-12 06:08:52: in-domain set (512, 128) from science-of-finetuning/synthetic-documents-cake_bake (original_index 12498..7809, skipped []), v2 eligible items included: True.

- mu_in_14: norm 15.377, split-half cos 0.9964, cos to mu_D 0.8316, fraction of ||mu_in||^2 along u_D 0.6915
- mu_in_all: norm 14.507, split-half cos 0.9990, cos to mu_D 0.5859, fraction along u_D 0.3433; ||mu_D|| = 7.223
- per-position ||mean delta|| on in-domain text, positions 0..8: P0=3225.50 P1=13.08 P2=14.71 P3=16.90 P4=19.32 P5=20.02 P6=20.16 P7=20.72 P8=20.76

**Alignment f(pos) = mean (delta.u_D)^2 / mean ||delta||^2** (in-domain / random panel):

| position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| in-domain | 0.2025 | 0.1615 | 0.1734 | 0.1699 | 0.1583 | 0.1489 | 0.0139 | 0.1357 |
| random panel | 0.1923 | 0.1745 | 0.1546 | 0.0250 | 0.0087 | 0.0037 | 0.1382 | 0.1224 |

**temp_implanted** (n_items=9, n_questions=7; B_base -5.362, B_ft +0.686): effect B - B_base [CI] label (sign + = toward y_A):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_in_14_matched | +0.000 [+0.000, +0.000] near_zero | -0.021 [-0.092, +0.042] near_zero | -0.012 [-0.082, +0.066] near_zero | -0.047 [-0.141, +0.082] near_zero | -0.175 [-0.390, +0.080] near_zero |
| mu_in_14_native | +0.000 [+0.000, +0.000] near_zero | -0.045 [-0.142, +0.058] near_zero | -0.068 [-0.203, +0.085] near_zero | -0.209 [-0.400, +0.022] inconclusive | -0.071 [-0.793, +0.599] inconclusive |
| mu_in_all_matched | +0.000 [+0.000, +0.000] near_zero | +0.015 [-0.019, +0.055] near_zero | -0.004 [-0.064, +0.057] near_zero | +0.023 [-0.105, +0.169] near_zero | -0.001 [-0.177, +0.210] near_zero |
| mu_in_all_native | +0.000 [+0.000, +0.000] near_zero | +0.010 [-0.042, +0.082] near_zero | -0.005 [-0.124, +0.115] near_zero | -0.027 [-0.227, +0.196] near_zero | -0.065 [-0.509, +0.369] inconclusive |

**implanted_factual_all** (n_items=12, n_questions=10; B_base -5.484, B_ft +0.907): effect B - B_base [CI] label (sign + = toward y_A):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_in_14_matched | +0.000 [+0.000, +0.000] near_zero | -0.014 [-0.078, +0.050] near_zero | +0.009 [-0.051, +0.076] near_zero | -0.003 [-0.107, +0.118] near_zero | +0.007 [-0.326, +0.372] near_zero |
| mu_in_14_native | +0.000 [+0.000, +0.000] near_zero | -0.027 [-0.105, +0.051] near_zero | -0.005 [-0.151, +0.152] near_zero | +0.007 [-0.337, +0.393] near_zero | +0.757 [-0.279, +2.066] inconclusive |
| mu_in_all_matched | +0.000 [+0.000, +0.000] near_zero | +0.027 [-0.026, +0.090] near_zero | +0.021 [-0.071, +0.124] near_zero | +0.095 [-0.105, +0.369] near_zero | +0.073 [-0.290, +0.563] inconclusive |
| mu_in_all_native | +0.000 [+0.000, +0.000] near_zero | +0.017 [-0.070, +0.108] near_zero | +0.052 [-0.130, +0.297] near_zero | +0.068 [-0.302, +0.577] inconclusive | +0.346 [-0.299, +1.209] inconclusive |

**implanted_completion_preference** (n_items=4, n_questions=4; B_base -0.375, B_ft +4.563): effect B - B_base [CI] label (sign + = toward y_A):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_in_14_matched | +0.000 [+0.000, +0.000] near_zero | +0.000 [-0.094, +0.094] near_zero | +0.031 [-0.125, +0.188] near_zero | +0.063 [-0.344, +0.406] near_zero | +0.047 [-1.453, +0.906] inconclusive |
| mu_in_14_native | +0.000 [+0.000, +0.000] near_zero | +0.094 [-0.062, +0.250] near_zero | -0.000 [-0.437, +0.313] near_zero | +0.047 [-1.609, +0.938] inconclusive | +0.703 [-3.047, +2.875] inconclusive |
| mu_in_all_matched | +0.000 [+0.000, +0.000] near_zero | +0.063 [-0.000, +0.125] near_zero | +0.063 [-0.094, +0.281] near_zero | +0.219 [-0.031, +0.500] inconclusive | +0.562 [-0.125, +1.344] inconclusive |
| mu_in_all_native | +0.000 [+0.000, +0.000] near_zero | +0.125 [+0.000, +0.281] near_zero | +0.250 [-0.031, +0.594] inconclusive | +0.625 [-0.094, +1.469] inconclusive | +1.953 [-0.422, +5.531] inconclusive |

**factual_control** (n_items=8, n_questions=6; B_base +9.850, B_ft +8.220): effect B - B_base [CI] label (sign + = toward y_A):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_in_14_matched | +0.000 [+0.000, +0.000] near_zero | -0.033 [-0.146, +0.083] near_zero | +0.016 [-0.192, +0.229] near_zero | +0.298 [-0.203, +0.875] inconclusive | +1.391 [+0.323, +2.613] nonzero |
| mu_in_14_native | +0.000 [+0.000, +0.000] near_zero | +0.019 [-0.200, +0.260] near_zero | +0.343 [-0.209, +0.969] inconclusive | +1.483 [+0.333, +2.754] nonzero | +1.314 [-1.021, +3.889] inconclusive |
| mu_in_all_matched | +0.000 [+0.000, +0.000] near_zero | -0.057 [-0.187, +0.063] near_zero | -0.028 [-0.328, +0.157] near_zero | +0.148 [-0.344, +0.505] inconclusive | +0.986 [+0.000, +1.861] nonzero |
| mu_in_all_native | +0.000 [+0.000, +0.000] near_zero | +0.004 [-0.271, +0.217] near_zero | +0.179 [-0.292, +0.535] inconclusive | +0.972 [+0.041, +1.797] nonzero | +3.890 [+1.781, +6.556] nonzero |

**domain_completion_preference** (n_items=2, n_questions=2; B_base +8.303, B_ft +8.352): effect B - B_base [CI] label (sign + = toward y_A):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_in_14_matched | +0.000 [+0.000, +0.000] near_zero | +0.243 [+0.236, +0.250] nonzero | +0.505 [+0.437, +0.573] nonzero | +0.910 [+0.687, +1.132] nonzero | +1.183 [+0.990, +1.375] nonzero |
| mu_in_14_native | +0.000 [+0.000, +0.000] near_zero | +0.517 [+0.375, +0.659] nonzero | +0.990 [+0.875, +1.105] nonzero | +1.169 [+0.839, +1.500] nonzero | -0.159 [-1.505, +1.187] inconclusive |
| mu_in_all_matched | +0.000 [+0.000, +0.000] near_zero | -0.033 [-0.254, +0.187] near_zero | -0.000 [-0.313, +0.312] near_zero | +0.015 [-0.470, +0.500] near_zero | -0.031 [-0.749, +0.687] inconclusive |
| mu_in_all_native | +0.000 [+0.000, +0.000] near_zero | +0.031 [-0.313, +0.375] near_zero | -0.072 [-0.582, +0.437] inconclusive | -0.002 [-0.753, +0.750] inconclusive | -0.073 [-1.022, +0.875] inconclusive |

**Panel** (fineweb; cap on fluency_drop):

| arm | alpha | fluency_drop | flagged | recovery |
|---|---|---|---|---|
| mu_in_14_native | 0.0 | +0.0000 | False | +0.0000 |
| mu_in_14_native | 0.5 | +0.0116 | False | +0.1330 |
| mu_in_14_native | 1.0 | +0.0706 | False | +0.1371 |
| mu_in_14_native | 2.0 | +0.4208 | False | -0.8897 |
| mu_in_14_native | 4.0 | +1.6434 | True | -6.8309 |
| mu_in_14_matched | 0.0 | +0.0000 | False | +0.0000 |
| mu_in_14_matched | 0.5 | +0.0019 | False | +0.0698 |
| mu_in_14_matched | 1.0 | +0.0102 | False | +0.1259 |
| mu_in_14_matched | 2.0 | +0.0594 | False | +0.1508 |
| mu_in_14_matched | 4.0 | +0.3680 | False | -0.6933 |
| mu_in_all_native | 0.0 | +0.0000 | False | +0.0000 |
| mu_in_all_native | 0.5 | -0.0019 | False | +0.0898 |
| mu_in_all_native | 1.0 | +0.0162 | False | +0.1103 |
| mu_in_all_native | 2.0 | +0.1439 | False | -0.1811 |
| mu_in_all_native | 4.0 | +0.7654 | False | -2.6452 |
| mu_in_all_matched | 0.0 | +0.0000 | False | +0.0000 |
| mu_in_all_matched | 0.5 | -0.0033 | False | +0.0522 |
| mu_in_all_matched | 1.0 | -0.0020 | False | +0.0896 |
| mu_in_all_matched | 2.0 | +0.0158 | False | +0.1110 |
| mu_in_all_matched | 4.0 | +0.1416 | False | -0.1750 |

**Ranks of the ||mu_D||-norm arms among the 23 F2 randoms at ||mu_D||** (value; rank_le/23; random min / median / max):

| arm | readout | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_in_14_matched | temp_implanted | -0.0206 (3/23; -0.057/+0.022/+0.089) | -0.0119 (8/23; -0.052/+0.014/+0.137) | -0.0474 (8/23; -0.159/+0.036/+0.185) | -0.1748 (2/23; -0.294/+0.103/+0.504) |
| mu_in_14_matched | factual_control | -0.0334 (6/23; -0.127/+0.002/+0.167) | +0.0163 (12/23; -0.238/+0.002/+0.355) | +0.2984 (18/23; -0.444/+0.021/+0.675) | +1.3915 (21/23; -1.011/+0.059/+1.622) |
| mu_in_14_matched | domain_completion_preference | +0.2432 (23/23; -0.221/+0.028/+0.212) | +0.5054 (23/23; -0.596/-0.007/+0.493) | +0.9097 (23/23; -1.179/+0.009/+0.769) | +1.1826 (22/23; -2.424/-0.295/+1.606) |
| mu_in_14_matched | kl_recovery | +0.0698 (23/23; -0.030/-0.007/+0.019) | +0.1259 (23/23; -0.069/-0.021/+0.028) | +0.1508 (23/23; -0.174/-0.066/+0.023) | -0.6933 (0/23; -0.533/-0.290/-0.098) |
| mu_in_14_matched | fluency_drop | +0.0019 (16/23; -0.002/+0.000/+0.006) | +0.0102 (20/23; -0.002/+0.002/+0.014) | +0.0594 (23/23; +0.002/+0.010/+0.036) | +0.3680 (23/23; +0.028/+0.046/+0.108) |
| mu_in_all_matched | temp_implanted | +0.0145 (8/23; -0.057/+0.022/+0.089) | -0.0043 (10/23; -0.052/+0.014/+0.137) | +0.0232 (9/23; -0.159/+0.036/+0.185) | -0.0013 (8/23; -0.294/+0.103/+0.504) |
| mu_in_all_matched | factual_control | -0.0573 (4/23; -0.127/+0.002/+0.167) | -0.0281 (8/23; -0.238/+0.002/+0.355) | +0.1484 (15/23; -0.444/+0.021/+0.675) | +0.9860 (20/23; -1.011/+0.059/+1.622) |
| mu_in_all_matched | domain_completion_preference | -0.0334 (7/23; -0.221/+0.028/+0.212) | -0.0005 (12/23; -0.596/-0.007/+0.493) | +0.0152 (12/23; -1.179/+0.009/+0.769) | -0.0309 (14/23; -2.424/-0.295/+1.606) |
| mu_in_all_matched | kl_recovery | +0.0522 (23/23; -0.030/-0.007/+0.019) | +0.0896 (23/23; -0.069/-0.021/+0.028) | +0.1110 (23/23; -0.174/-0.066/+0.023) | -0.1750 (21/23; -0.533/-0.290/-0.098) |
| mu_in_all_matched | fluency_drop | -0.0033 (0/23; -0.002/+0.000/+0.006) | -0.0020 (0/23; -0.002/+0.002/+0.014) | +0.0158 (16/23; +0.002/+0.010/+0.036) | +0.1416 (23/23; +0.028/+0.046/+0.108) |

Stated: the extraction texts are synthetic documents, so any effect may be specific to that document style.
<!-- F8-NUMBERS-END -->

---

## Addendum 2 (FOLLOWUP_BRIEF_F9.md): F9 queued after F6, before F7 and F8 (drop F8 before F9 if time forces it); temperature grid attached to F1 and F9; cake-context prompts attached to F3 Run B

**Decision positions** (computed before any F9 run; `results/followup/f9_decision_positions.csv`,
printed in `log_decision_positions.txt`): d = n_prefix - 1 + k, k = index of the first differing
continuation token; A/B token histories asserted identical through d. Every temperature item and
every control prefix has k = 1, d = the shared leading-space token (not in the standard mask P);
every single-token answer (factual controls, completion-preference items, butter, cooling,
concrete) has k = 0, d = the last prompt position (inside P); cake_impl_14 (vanilla) has k = 1 with
d at a shared " a". Grid candidates 300..500 all tokenise to [' ', digit, digit, digit].

**Attachment to F1 (`f1_temp_grid.csv`, block inside F1's numbers).** For every temperature item and
every arm / alpha, each candidate in G = {300, 325, ..., 500} is teacher-forced under the same
intervention (standard mask). Primary: " NNN" (4 tokens each; probabilities of the specified
continuation strings, which include longer outputs beginning with those digits). Secondary: " NNN°F"
(a consistent terminating boundary; identical suffix tokens asserted; probabilities of completed
answers under that boundary). Reported for both: the grid-normalised distribution, the total grid
mass, the mode, the mass at 450 and at 400+425. Consistency gate: logp(450) - logp(350) from the
primary grid equals the sweep's B on every temperature row. An average of 400 is not a preference
for 400.

**Attachment to F3 Run B (`group` column).** Five cake-context prompts (config
`F3_CAKE_CONTEXT_PROMPTS`) generated under the same decoding for every Run B arm with seeds
crc32(f"{100+i}|{replicate}") shared across arms, reported separately from the neutral openers; a
sampled 450 is reported as observed and not treated as contradicting the belief results.

## F9. delta_ans: the finetuning difference at the decision position (FOLLOWUP_BRIEF_F9.md rev 2)

**Uncertainty addressed.** Whether a direction extracted where the answer is decided transports the
implanted preference when the random-text mean does not; whether such a direction's effect is
confined to cake-temperature prompts or extends to other temperature and numeric contexts.

**What will be run** (`followup_f9.py`; config `F9_*`). Extraction set E = {cake_impl_07, 08 (pair
cake_setoven), 02, 18}: four items, three question units. Evaluation set V = {cake_impl_01, 04 (pair
cake_preheat), 03, 16, 17}: five items, four question units; every bootstrap over V uses those four.
V is held out from vector construction, not untouched (its original items were seen in the sweep); it
tests transfer across phrasings of one proposition. delta_ans,l = mean over E of h_ft,l(d) - h_base,l(d)
on prompt + shared continuation prefix through d (Residual at all layers, base vs cake adapter);
reported ||delta_ans,l|| per layer, split-half cosine (07/08 vs 02/18) per layer, cos(delta_ans,17,
mu_D); likewise delta_ans,concrete from the concrete item at its d (extracted on that prompt itself; not
held out; comparator only). Arms at layer 17, alpha in {0.5, 1, 2, 4}, on V, the control sets, the
other v2 propositions, the factual controls and the completion-preference items, each at its own d:
(1) delta_ans at D, native; (2) delta_ans at D at ||mu_D||; (3) mu_D at D, native; (4) mu_D at D at
||delta||; (5) mu_D at P (bit-identity with the existing sweep asserted); (6) mu_D at P + delta_ans at
D, each native (both vectors at d where D lies inside P); (7) delta_ans,concrete at D rescaled to
||delta|| on V and the control sets (cross-organism control on the cake task), and separately native
on the concrete item (within-concrete diagnostic); (8) r0-r22 at D at both ||delta|| and ||mu_D||, so
each norm has its own reference; (9) mu_D at P+D (F4-S) for reference.
Pre-specified paired contrasts (question bootstrap over V's four units): arm 1 - arm 4 (vector at
||delta||); arm 2 - arm 3 (vector at ||mu_D||); arm 3 - arm 5 (position, vector and norm fixed); arm 1 -
arm 7 (cross-organism); interaction I = B(6) - B(5) - B(1) + B(base). Labels are reported but arm
differences rest on these contrasts, not on label differences.
Control prompt sets (two paraphrases each, " 450" / " 350"; order fixed before running as a hypothesis
about these contexts: cookies, bread, roast chicken, furnace, odometer). No true temperature is
assigned to them; they measure change in preference. Readout: effect per prompt and the direct
contrast V-effect minus control-set effect with CIs; subtracting each prompt's baseline removes its
initial score but does not equalise its sensitivity to intervention. Layer sweep (exploratory): arm 1
with delta_ans,l at D for every l, alpha = 1, on V and on cookies / odometer ("where this intervention
becomes effective"). Temperature grid on V for the main arms, primary " NNN" and secondary " NNN°F".
Gates (halting): alpha = 0 bit-exact for masks D and P+D and the combined arm; local-increment check
at d on every forward (k = 1 and k = 0 items, every mask); the decision-position table printed and
checked (k = 0: d = n_prefix - 1, mask D is the last-prompt-position intervention; the combined arm adds
both vectors there); bit-identity of arm 5 with the existing sweep.

**Outcomes -> interpretation** (written 2026-09-12, before the run; each "consistent with", none
identifying a mechanism; observed row marked after):

| outcome | interpretation |
|---|---|
| delta_ans raises B on V toward 450 with direct contrasts against the control sets whose CIs exclude 0, and no comparable effect on other propositions | a direction extracted at the decision position transfers the cake-temperature preference across the tested phrasings under this intervention; does not show mu_D lacks the information; does not establish a uniquely proposition-specific representation |
| effect present on V and comparable across all control sets | consistent with a broad numerical effect |
| effect declining with the fixed ordering | suggests context dependence; the ordering is a hypothesis and two paraphrases per set are a small descriptive comparison |
| effect on every implanted proposition | consistent with an organism-level direction |
| no effect on V | this extracted mean fails under the tested conditions; nothing about later layers or non-additive mechanisms |
| arm 3 ~ arm 1 at matched norm | similar effects at the same position; does not establish that position explains the whole original difference; the position question is arm 3 - arm 5 |
| I's interval contains 0 | no interaction detected (not additivity established, especially with four units) |
| I's interval excludes 0 | dependence between the two interventions, not topic-gated access |
| grid: intermediate values gain mass at intermediate alpha vs mass moving monotonically | reported as observed; an average of 400 is not a preference for 400 |

<!-- F9-NUMBERS-START -->
**Run** 2026-09-12 06:35:59. E = 4 items / 3 question units; V = 5 items / 4 question units (every bootstrap over V uses those four). ||delta_ans,17|| = 20.207, ||mu_D|| = 7.223, cos(delta_ans,17, mu_D) = 0.4559; split-half cos at 17 = 0.5412. Per-layer norms / split-half in f9_layer_stats.csv.

**Pre-specified paired contrasts on V** (question bootstrap over the four units; arm differences rest on these, not on label differences):

| alpha | contrast | arm_x | arm_y | point | ci_lo | ci_hi | label | n_questions |
|---|---|---|---|---|---|---|---|---|
| +0.500 | vector_at_norm_dA | dA_D | muD_D_matched_dA | +0.155 | -0.009 | +0.357 | near_zero | 4 |
| +0.500 | vector_at_norm_muD | dA_D_matched_muD | muD_D | +0.124 | +0.028 | +0.240 | near_zero | 4 |
| +0.500 | position_muD_D_minus_P | muD_D | muD_P | -0.067 | -0.158 | +0.031 | near_zero | 4 |
| +0.500 | cross_organism_dA_minus_dConc | dA_D | dConc_D_matched_dA | +0.053 | -0.103 | +0.179 | near_zero | 4 |
| +1.000 | vector_at_norm_dA | dA_D | muD_D_matched_dA | +0.353 | -0.035 | +0.857 | inconclusive | 4 |
| +1.000 | vector_at_norm_muD | dA_D_matched_muD | muD_D | +0.107 | -0.007 | +0.209 | near_zero | 4 |
| +1.000 | position_muD_D_minus_P | muD_D | muD_P | -0.077 | -0.121 | -0.001 | near_zero | 4 |
| +1.000 | cross_organism_dA_minus_dConc | dA_D | dConc_D_matched_dA | +0.109 | -0.154 | +0.433 | near_zero | 4 |
| +2.000 | vector_at_norm_dA | dA_D | muD_D_matched_dA | +0.946 | +0.352 | +1.798 | nonzero | 4 |
| +2.000 | vector_at_norm_muD | dA_D_matched_muD | muD_D | +0.138 | -0.077 | +0.451 | near_zero | 4 |
| +2.000 | position_muD_D_minus_P | muD_D | muD_P | -0.133 | -0.404 | +0.060 | near_zero | 4 |
| +2.000 | cross_organism_dA_minus_dConc | dA_D | dConc_D_matched_dA | +0.348 | -0.067 | +0.930 | inconclusive | 4 |
| +4.000 | vector_at_norm_dA | dA_D | muD_D_matched_dA | +2.541 | +0.967 | +4.294 | nonzero | 4 |
| +4.000 | vector_at_norm_muD | dA_D_matched_muD | muD_D | +0.645 | +0.143 | +1.470 | nonzero | 4 |
| +4.000 | position_muD_D_minus_P | muD_D | muD_P | -0.537 | -0.677 | -0.397 | nonzero | 4 |
| +4.000 | cross_organism_dA_minus_dConc | dA_D | dConc_D_matched_dA | +1.095 | +0.070 | +2.830 | nonzero | 4 |

**V** (n_items=5, n_questions=4; B_base mean -5.347): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.107 [-0.100, +0.276] near_zero | +0.164 [-0.244, +0.601] inconclusive | +0.391 [-0.348, +1.299] inconclusive | +1.452 [-0.175, +3.806] inconclusive |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.078 [+0.008, +0.168] near_zero | +0.075 [-0.058, +0.201] near_zero | +0.144 [-0.072, +0.326] near_zero | +0.240 [-0.296, +0.892] inconclusive |
| muD_D | +0.000 [+0.000, +0.000] near_zero | -0.047 [-0.084, -0.009] near_zero | -0.031 [-0.051, -0.008] near_zero | +0.006 [-0.133, +0.145] near_zero | -0.406 [-0.611, -0.201] nonzero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.048 [-0.124, +0.038] near_zero | -0.188 [-0.277, -0.099] near_zero | -0.555 [-0.795, -0.277] nonzero | -1.089 [-1.755, -0.422] nonzero |
| muD_P | +0.000 [+0.000, +0.000] near_zero | +0.021 [-0.044, +0.097] near_zero | +0.045 [-0.033, +0.106] near_zero | +0.139 [+0.046, +0.249] near_zero | +0.131 [+0.066, +0.196] near_zero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.145 [-0.096, +0.377] near_zero | +0.198 [-0.148, +0.569] inconclusive | +0.540 [-0.131, +1.441] inconclusive | +2.632 [+1.399, +4.799] nonzero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.055 [+0.001, +0.108] near_zero | +0.056 [-0.106, +0.217] near_zero | +0.043 [-0.261, +0.323] near_zero | +0.357 [-0.310, +0.935] inconclusive |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | +0.013 [-0.090, +0.103] near_zero | +0.020 [-0.190, +0.229] near_zero | +0.175 [-0.255, +0.605] inconclusive | +0.770 [-0.352, +1.926] inconclusive |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | +0.016 [+0.007, +0.028] near_zero | +0.028 [-0.060, +0.116] near_zero | +0.066 [+0.017, +0.105] near_zero | -0.127 [-0.187, -0.080] near_zero |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.082 [-0.132, -0.029] near_zero | -0.149 [-0.291, -0.007] near_zero | -0.251 [-0.463, -0.124] nonzero | +0.641 [-0.208, +1.913] inconclusive |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.075 [-0.119, +0.217] near_zero | -0.003 [-0.272, +0.222] near_zero | +0.049 [-0.336, +0.434] near_zero | -0.218 [-0.861, +0.425] inconclusive |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.049 [-0.132, +0.023] near_zero | -0.019 [-0.076, +0.083] near_zero | -0.188 [-0.328, -0.025] near_zero | +0.389 [-0.472, +1.172] inconclusive |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.069 [-0.151, +0.014] near_zero | -0.037 [-0.060, -0.012] near_zero | -0.128 [-0.275, -0.025] near_zero | -0.228 [-0.393, -0.120] nonzero |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.012 [-0.114, +0.110] near_zero | -0.008 [-0.150, +0.101] near_zero | +0.084 [-0.171, +0.241] near_zero | +0.052 [-0.346, +0.331] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.015 [-0.023, +0.054] near_zero | -0.067 [-0.158, +0.001] near_zero | -0.046 [-0.174, +0.083] near_zero | -0.168 [-0.236, -0.099] near_zero |

E_in_sample (the four extraction items, in-sample, 3 question units), arm dA_D, effect = B - B_base: alpha 0.5: +0.120 [+0.084, +0.146] near_zero, alpha 1: +0.338 [+0.266, +0.394] nonzero, alpha 2: +0.746 [+0.260, +1.131] nonzero, alpha 4: +2.485 [+1.415, +3.959] nonzero -- shown here so the in-sample and held-out (V) effects sit together.

**E_in_sample** (n_items=4, n_questions=3; B_base mean -5.382): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.120 [+0.084, +0.146] near_zero | +0.338 [+0.266, +0.394] nonzero | +0.746 [+0.260, +1.131] nonzero | +2.485 [+1.415, +3.959] nonzero |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.018 [-0.126, +0.112] near_zero | +0.053 [+0.015, +0.079] near_zero | +0.179 [+0.165, +0.206] near_zero | +0.445 [+0.214, +0.658] nonzero |
| muD_D | +0.000 [+0.000, +0.000] near_zero | -0.091 [-0.219, +0.003] near_zero | -0.111 [-0.253, -0.035] near_zero | -0.080 [-0.233, +0.028] near_zero | -0.367 [-0.612, -0.155] nonzero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.148 [-0.357, -0.038] near_zero | -0.183 [-0.429, -0.007] near_zero | -0.585 [-0.762, -0.370] nonzero | -0.463 [-1.085, +0.353] inconclusive |
| muD_P | +0.000 [+0.000, +0.000] near_zero | -0.034 [-0.124, +0.032] near_zero | -0.041 [-0.147, +0.080] near_zero | -0.038 [-0.202, +0.129] near_zero | -0.057 [-0.468, +0.474] near_zero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.119 [+0.087, +0.152] near_zero | +0.283 [+0.166, +0.354] nonzero | +0.825 [+0.343, +1.326] nonzero | +3.428 [+2.450, +4.843] nonzero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.032 [-0.071, +0.020] near_zero | +0.025 [-0.088, +0.203] near_zero | +0.187 [+0.160, +0.204] near_zero | +0.855 [+0.624, +1.286] nonzero |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | +0.009 [-0.088, +0.098] near_zero | +0.104 [+0.026, +0.186] near_zero | +0.517 [+0.338, +0.664] nonzero | +1.823 [+0.690, +2.642] nonzero |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | -0.018 [-0.155, +0.081] near_zero | -0.081 [-0.224, +0.080] near_zero | -0.152 [-0.229, -0.050] near_zero | -0.317 [-0.638, +0.198] inconclusive |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.157 [-0.286, -0.004] near_zero | -0.255 [-0.471, +0.014] inconclusive | -0.492 [-1.033, -0.170] nonzero | -0.976 [-2.068, -0.406] nonzero |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.081 [-0.177, +0.232] near_zero | +0.159 [-0.263, +0.391] near_zero | +0.266 [-0.560, +0.692] inconclusive | +0.155 [-0.901, +0.892] inconclusive |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.102 [-0.359, +0.109] near_zero | -0.188 [-0.480, +0.006] near_zero | -0.287 [-0.991, +0.096] inconclusive | +0.335 [-0.288, +0.942] inconclusive |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.086 [-0.207, +0.001] near_zero | -0.091 [-0.161, -0.053] near_zero | -0.222 [-0.534, -0.006] nonzero | -0.364 [-0.700, -0.122] nonzero |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.053 [-0.016, +0.122] near_zero | +0.138 [-0.037, +0.279] near_zero | +0.124 [-0.100, +0.246] near_zero | +0.145 [-0.442, +0.462] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.032 [-0.126, +0.068] near_zero | -0.127 [-0.324, +0.001] near_zero | -0.154 [-0.392, -0.010] near_zero | -0.261 [-0.653, -0.040] nonzero |

**ctrl:cookies** (n_items=2, n_questions=2; B_base mean -7.886): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.262 [+0.254, +0.270] nonzero | +0.580 [+0.421, +0.739] nonzero | +1.408 [+0.776, +2.039] nonzero | +2.849 [+2.003, +3.695] nonzero |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.134 [+0.055, +0.214] near_zero | +0.280 [+0.247, +0.313] nonzero | +0.287 [+0.205, +0.370] nonzero | +0.909 [+0.585, +1.233] nonzero |
| muD_D | +0.000 [+0.000, +0.000] near_zero | -0.179 [-0.214, -0.144] near_zero | -0.081 [-0.141, -0.022] near_zero | -0.235 [-0.268, -0.203] nonzero | -0.526 [-0.565, -0.487] nonzero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.127 [-0.128, -0.125] near_zero | -0.254 [-0.406, -0.102] nonzero | -0.677 [-0.755, -0.599] nonzero | -0.682 [-1.013, -0.351] nonzero |
| muD_P | +0.000 [+0.000, +0.000] near_zero | +0.077 [+0.049, +0.105] near_zero | +0.088 [-0.049, +0.224] near_zero | +0.358 [+0.134, +0.581] nonzero | +0.801 [+0.291, +1.311] nonzero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.357 [+0.303, +0.412] nonzero | +0.754 [+0.673, +0.836] nonzero | +2.026 [+1.348, +2.704] nonzero | +4.365 [+3.952, +4.779] nonzero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.000 [+0.000, +0.000] near_zero | -0.123 [-0.212, -0.034] near_zero | -0.059 [-0.159, +0.041] near_zero | +0.399 [-0.076, +0.873] inconclusive |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | +0.018 [+0.014, +0.023] near_zero | -0.128 [-0.161, -0.095] near_zero | +0.136 [-0.304, +0.575] inconclusive | +1.195 [+0.284, +2.106] nonzero |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | +0.006 [-0.003, +0.015] near_zero | +0.084 [+0.067, +0.102] near_zero | +0.080 [-0.121, +0.281] near_zero | +0.296 [-0.258, +0.850] inconclusive |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.034 [-0.089, +0.021] near_zero | +0.097 [+0.030, +0.163] near_zero | +0.057 [-0.017, +0.131] near_zero | +1.160 [+0.846, +1.474] nonzero |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.301 [+0.206, +0.395] nonzero | +0.623 [+0.452, +0.794] nonzero | +0.814 [+0.655, +0.972] nonzero | +0.856 [+0.567, +1.145] nonzero |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.095 [-0.111, -0.080] near_zero | -0.271 [-0.321, -0.221] nonzero | -0.199 [-0.525, +0.127] inconclusive | +1.726 [+1.082, +2.370] nonzero |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.065 [+0.003, +0.126] near_zero | +0.027 [-0.089, +0.142] near_zero | -0.021 [-0.096, +0.054] near_zero | -0.051 [-0.181, +0.079] near_zero |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.127 [+0.109, +0.146] near_zero | +0.256 [+0.118, +0.393] nonzero | +0.399 [+0.258, +0.539] nonzero | +0.658 [+0.492, +0.824] nonzero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.005 [-0.011, +0.000] near_zero | -0.053 [-0.105, +0.000] near_zero | -0.236 [-0.245, -0.227] nonzero | -0.370 [-0.439, -0.302] nonzero |

**ctrl:bread** (n_items=2, n_questions=2; B_base mean -2.588): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.323 [+0.245, +0.400] nonzero | +0.686 [+0.560, +0.813] nonzero | +1.837 [+1.239, +2.435] nonzero | +2.914 [+1.831, +3.997] nonzero |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.060 [+0.000, +0.120] near_zero | +0.276 [+0.273, +0.278] nonzero | +0.498 [+0.473, +0.523] nonzero | +1.258 [+0.945, +1.572] nonzero |
| muD_D | +0.000 [+0.000, +0.000] near_zero | +0.040 [-0.045, +0.125] near_zero | -0.008 [-0.174, +0.158] near_zero | +0.044 [-0.011, +0.100] near_zero | +0.349 [+0.137, +0.560] nonzero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.087 [+0.021, +0.153] near_zero | +0.041 [-0.061, +0.143] near_zero | +0.642 [+0.362, +0.922] nonzero | +0.872 [+0.002, +1.742] nonzero |
| muD_P | +0.000 [+0.000, +0.000] near_zero | +0.137 [+0.113, +0.161] near_zero | +0.156 [+0.151, +0.161] near_zero | +0.249 [+0.226, +0.272] nonzero | +0.809 [+0.374, +1.244] nonzero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.397 [+0.276, +0.518] nonzero | +0.795 [+0.493, +1.096] nonzero | +2.211 [+1.365, +3.056] nonzero | +3.675 [+2.172, +5.177] nonzero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.091 [+0.069, +0.113] near_zero | +0.022 [-0.143, +0.186] near_zero | +0.220 [-0.339, +0.779] inconclusive | +1.165 [+0.087, +2.242] nonzero |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | +0.061 [-0.067, +0.189] near_zero | +0.128 [-0.209, +0.466] near_zero | +0.644 [-0.403, +1.692] inconclusive | +1.697 [+0.771, +2.624] nonzero |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | +0.125 [+0.027, +0.222] near_zero | +0.065 [+0.026, +0.105] near_zero | +0.280 [+0.183, +0.376] nonzero | +1.225 [+0.443, +2.006] nonzero |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.069 [-0.030, +0.169] near_zero | +0.093 [-0.151, +0.338] near_zero | +0.122 [-0.437, +0.680] inconclusive | +1.421 [-0.584, +3.425] inconclusive |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.104 [-0.078, +0.287] near_zero | +0.136 [+0.089, +0.182] near_zero | +0.051 [-0.041, +0.142] near_zero | +0.303 [+0.122, +0.485] nonzero |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.136 [-0.001, +0.273] near_zero | +0.269 [+0.107, +0.431] nonzero | +0.894 [+0.577, +1.210] nonzero | +1.627 [+0.244, +3.009] nonzero |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.070 [+0.043, +0.097] near_zero | +0.060 [+0.004, +0.117] near_zero | +0.154 [+0.009, +0.300] near_zero | +0.218 [-0.156, +0.591] inconclusive |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.075 [+0.022, +0.129] near_zero | +0.195 [+0.005, +0.385] near_zero | +0.076 [-0.126, +0.279] near_zero | +0.084 [+0.017, +0.151] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.022 [-0.044, -0.000] near_zero | +0.040 [+0.004, +0.075] near_zero | +0.231 [+0.139, +0.323] nonzero | +0.386 [+0.232, +0.541] nonzero |

**ctrl:roast_chicken** (n_items=2, n_questions=2; B_base mean -1.026): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.376 [-0.013, +0.764] inconclusive | +0.703 [+0.214, +1.192] nonzero | +1.290 [+0.169, +2.412] nonzero | +1.504 [+0.133, +2.875] nonzero |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.137 [-0.133, +0.408] near_zero | +0.330 [-0.002, +0.663] inconclusive | +0.430 [-0.028, +0.888] inconclusive | +0.922 [-0.021, +1.865] inconclusive |
| muD_D | +0.000 [+0.000, +0.000] near_zero | +0.038 [-0.048, +0.124] near_zero | +0.057 [-0.111, +0.225] near_zero | +0.220 [-0.037, +0.477] inconclusive | +0.534 [-0.173, +1.240] inconclusive |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.133 [-0.123, +0.390] near_zero | +0.312 [-0.041, +0.665] inconclusive | +0.878 [-0.332, +2.089] inconclusive | +0.872 [-0.552, +2.297] inconclusive |
| muD_P | +0.000 [+0.000, +0.000] near_zero | +0.103 [-0.144, +0.350] near_zero | +0.081 [-0.059, +0.221] near_zero | +0.285 [+0.011, +0.559] nonzero | +0.841 [+0.469, +1.213] nonzero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.381 [-0.001, +0.763] inconclusive | +0.618 [-0.060, +1.297] inconclusive | +1.627 [+0.574, +2.680] nonzero | +2.441 [+0.736, +4.145] nonzero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.020 [-0.209, +0.249] near_zero | -0.116 [-0.598, +0.367] inconclusive | +0.002 [-0.774, +0.778] inconclusive | +0.480 [+0.196, +0.764] nonzero |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | -0.055 [-0.460, +0.350] near_zero | +0.023 [-0.554, +0.600] inconclusive | +0.211 [-0.482, +0.904] inconclusive | +0.478 [+0.145, +0.811] nonzero |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | +0.145 [+0.063, +0.226] near_zero | +0.197 [-0.164, +0.557] inconclusive | +0.370 [+0.152, +0.589] nonzero | +1.114 [+0.382, +1.846] nonzero |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.037 [-0.199, +0.125] near_zero | -0.227 [-0.573, +0.119] inconclusive | -0.807 [-1.345, -0.269] nonzero | -1.037 [-2.737, +0.663] inconclusive |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.133 [-0.189, +0.456] near_zero | +0.100 [-0.211, +0.412] near_zero | -0.057 [-0.426, +0.312] near_zero | -0.469 [-1.453, +0.516] inconclusive |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.106 [+0.087, +0.126] near_zero | -0.010 [-0.204, +0.184] near_zero | -0.361 [-0.740, +0.017] inconclusive | -0.171 [-0.602, +0.259] inconclusive |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.043 [-0.098, +0.183] near_zero | -0.219 [-0.340, -0.097] nonzero | -0.147 [-0.348, +0.055] near_zero | -0.522 [-1.092, +0.048] inconclusive |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.105 [-0.021, +0.231] near_zero | +0.055 [-0.138, +0.249] near_zero | +0.138 [-0.182, +0.459] near_zero | +0.084 [-0.244, +0.411] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.126 [+0.125, +0.126] near_zero | -0.004 [-0.038, +0.029] near_zero | +0.056 [-0.072, +0.184] near_zero | -0.079 [-0.175, +0.018] near_zero |

**ctrl:furnace** (n_items=2, n_questions=2; B_base mean +0.600): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.031 [-0.044, +0.107] near_zero | +0.102 [+0.002, +0.201] near_zero | +0.026 [-0.066, +0.118] near_zero | +0.258 [+0.158, +0.358] nonzero |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | -0.054 [-0.118, +0.010] near_zero | +0.106 [-0.116, +0.327] near_zero | +0.011 [-0.148, +0.170] near_zero | +0.175 [+0.014, +0.337] near_zero |
| muD_D | +0.000 [+0.000, +0.000] near_zero | -0.076 [-0.138, -0.015] near_zero | -0.073 [-0.164, +0.018] near_zero | +0.001 [-0.098, +0.100] near_zero | -0.111 [-0.290, +0.067] near_zero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.082 [-0.117, -0.048] near_zero | -0.175 [-0.293, -0.057] near_zero | -0.013 [-0.232, +0.206] near_zero | +0.124 [-0.213, +0.461] near_zero |
| muD_P | +0.000 [+0.000, +0.000] near_zero | -0.037 [-0.114, +0.040] near_zero | -0.155 [-0.278, -0.031] near_zero | -0.309 [-0.352, -0.266] nonzero | -0.328 [-0.335, -0.322] nonzero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | -0.028 [-0.130, +0.074] near_zero | -0.021 [-0.132, +0.091] near_zero | +0.077 [+0.060, +0.094] near_zero | -0.029 [-0.454, +0.396] near_zero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.029 [-0.065, +0.008] near_zero | +0.035 [-0.014, +0.083] near_zero | +0.016 [+0.005, +0.028] near_zero | +0.178 [+0.176, +0.181] near_zero |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | -0.147 [-0.182, -0.111] near_zero | +0.010 [-0.039, +0.060] near_zero | +0.155 [+0.051, +0.260] near_zero | +0.438 [+0.337, +0.540] nonzero |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | -0.148 [-0.239, -0.058] near_zero | -0.084 [-0.202, +0.034] near_zero | -0.156 [-0.235, -0.077] near_zero | -0.182 [-0.492, +0.128] near_zero |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.048 [+0.020, +0.076] near_zero | +0.146 [+0.012, +0.280] near_zero | +0.369 [+0.351, +0.387] nonzero | +1.023 [+0.548, +1.499] nonzero |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.190 [-0.270, -0.109] near_zero | -0.136 [-0.159, -0.114] near_zero | -0.110 [-0.148, -0.073] near_zero | +0.029 [-0.063, +0.120] near_zero |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.011 [-0.203, +0.181] near_zero | -0.056 [-0.164, +0.052] near_zero | +0.012 [-0.113, +0.136] near_zero | -0.243 [-0.246, -0.239] nonzero |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.060 [-0.111, -0.010] near_zero | +0.031 [-0.062, +0.123] near_zero | +0.066 [-0.177, +0.308] near_zero | +0.225 [+0.041, +0.410] nonzero |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.185 [-0.211, -0.159] near_zero | -0.096 [-0.111, -0.081] near_zero | -0.124 [-0.161, -0.087] near_zero | -0.235 [-0.330, -0.140] nonzero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.097 [-0.109, -0.085] near_zero | -0.029 [-0.080, +0.022] near_zero | +0.057 [-0.032, +0.147] near_zero | +0.019 [-0.151, +0.188] near_zero |

**ctrl:odometer** (n_items=2, n_questions=2; B_base mean -0.016): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | -0.141 [-0.256, -0.027] near_zero | -0.065 [-0.139, +0.009] near_zero | -0.039 [-0.406, +0.328] near_zero | +0.064 [-0.121, +0.250] near_zero |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | -0.075 [-0.156, +0.006] near_zero | -0.140 [-0.215, -0.065] near_zero | -0.011 [-0.101, +0.078] near_zero | +0.035 [-0.057, +0.127] near_zero |
| muD_D | +0.000 [+0.000, +0.000] near_zero | -0.075 [-0.143, -0.007] near_zero | -0.087 [-0.140, -0.034] near_zero | +0.046 [-0.183, +0.274] near_zero | +0.198 [+0.061, +0.335] near_zero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.029 [-0.059, +0.000] near_zero | -0.034 [-0.231, +0.164] near_zero | +0.242 [+0.110, +0.373] nonzero | +0.598 [+0.172, +1.023] nonzero |
| muD_P | +0.000 [+0.000, +0.000] near_zero | -0.054 [-0.114, +0.006] near_zero | -0.060 [-0.157, +0.036] near_zero | +0.003 [-0.106, +0.111] near_zero | +0.004 [-0.114, +0.123] near_zero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.057 [-0.137, +0.251] near_zero | +0.022 [-0.140, +0.183] near_zero | +0.094 [-0.097, +0.285] near_zero | -0.045 [-0.109, +0.019] near_zero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.061 [-0.002, +0.124] near_zero | -0.050 [-0.068, -0.033] near_zero | +0.195 [-0.014, +0.403] near_zero | +0.345 [+0.243, +0.447] nonzero |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | -0.163 [-0.171, -0.155] near_zero | +0.135 [+0.003, +0.267] near_zero | +0.256 [+0.062, +0.449] nonzero | -0.014 [-0.080, +0.052] near_zero |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | -0.009 [-0.137, +0.118] near_zero | -0.100 [-0.137, -0.063] near_zero | +0.105 [+0.045, +0.165] near_zero | +0.230 [+0.014, +0.445] nonzero |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.151 [-0.251, -0.051] near_zero | -0.170 [-0.184, -0.155] near_zero | -0.049 [-0.186, +0.087] near_zero | -1.170 [-1.348, -0.993] nonzero |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.044 [-0.033, +0.121] near_zero | -0.055 [-0.132, +0.021] near_zero | +0.091 [+0.055, +0.128] near_zero | -0.037 [-0.151, +0.078] near_zero |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.157 [-0.306, -0.008] near_zero | -0.112 [-0.214, -0.009] near_zero | -0.158 [-0.285, -0.031] near_zero | -0.576 [-0.852, -0.299] nonzero |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.227 [-0.276, -0.177] nonzero | -0.120 [-0.176, -0.064] near_zero | -0.220 [-0.406, -0.035] nonzero | -0.023 [-0.118, +0.072] near_zero |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.023 [+0.009, +0.036] near_zero | +0.031 [+0.010, +0.051] near_zero | -0.026 [-0.049, -0.002] near_zero | +0.121 [+0.078, +0.164] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.004 [-0.088, +0.095] near_zero | -0.099 [-0.253, +0.056] near_zero | -0.194 [-0.421, +0.033] near_zero | -0.233 [-0.344, -0.123] nonzero |

**other_factual_propositions** (n_items=3, n_questions=3; B_base mean -5.770): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.026 [-0.107, +0.125] near_zero | +0.203 [-0.234, +0.467] inconclusive | +0.438 [-0.485, +1.173] inconclusive | +0.981 [-0.241, +2.246] inconclusive |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.063 [-0.137, +0.199] near_zero | +0.056 [-0.036, +0.125] near_zero | +0.158 [-0.119, +0.313] near_zero | +0.282 [-0.182, +0.590] inconclusive |
| muD_D | +0.000 [+0.000, +0.000] near_zero | -0.039 [-0.125, +0.071] near_zero | -0.014 [-0.250, +0.136] near_zero | -0.228 [-0.562, +0.045] inconclusive | -0.060 [-0.938, +0.512] inconclusive |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.044 [-0.312, +0.175] near_zero | -0.124 [-0.687, +0.194] inconclusive | -0.020 [-0.750, +0.440] inconclusive | +0.229 [-0.438, +0.765] inconclusive |
| muD_P | +0.000 [+0.000, +0.000] near_zero | +0.115 [-0.062, +0.225] near_zero | +0.001 [-0.250, +0.256] near_zero | +0.207 [-0.250, +0.955] inconclusive | +0.934 [-0.133, +2.311] inconclusive |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.182 [-0.060, +0.419] near_zero | +0.362 [-0.180, +1.016] inconclusive | +1.006 [-0.393, +2.223] inconclusive | +2.228 [-0.596, +4.591] inconclusive |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.101 [-0.375, +0.174] near_zero | -0.216 [-0.688, +0.165] inconclusive | -0.194 [-1.375, +0.873] inconclusive | -0.241 [-2.938, +2.613] inconclusive |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | -0.109 [-0.500, +0.123] near_zero | -0.206 [-1.063, +0.523] inconclusive | -0.156 [-2.125, +1.819] inconclusive | -0.296 [-3.438, +3.058] inconclusive |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | +0.054 [-0.062, +0.225] near_zero | +0.043 [-0.250, +0.256] near_zero | +0.242 [-0.250, +0.955] inconclusive | +0.924 [-0.164, +2.311] inconclusive |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.086 [-0.125, +0.262] near_zero | +0.062 [-0.438, +0.365] near_zero | -0.079 [-1.500, +0.703] inconclusive | -0.506 [-2.218, +0.625] inconclusive |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.090 [-0.062, +0.244] near_zero | +0.034 [-0.250, +0.432] near_zero | -0.297 [-1.000, +0.488] inconclusive | -0.687 [-2.438, +0.329] inconclusive |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.001 [-0.125, +0.127] near_zero | -0.139 [-0.363, +0.070] near_zero | -0.146 [-0.550, +0.375] inconclusive | +1.093 [-0.008, +3.250] inconclusive |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.015 [-0.125, +0.129] near_zero | -0.037 [-0.122, +0.073] near_zero | -0.033 [-0.312, +0.211] near_zero | -0.016 [-1.000, +0.632] inconclusive |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.021 [-0.134, +0.196] near_zero | +0.071 [+0.000, +0.203] near_zero | +0.139 [+0.000, +0.309] near_zero | -0.141 [-0.437, +0.307] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.027 [-0.118, +0.136] near_zero | +0.031 [-0.147, +0.125] near_zero | +0.013 [-0.063, +0.118] near_zero | -0.203 [-0.415, -0.006] nonzero |

**implanted_completion_preference** (n_items=4, n_questions=4; B_base mean -0.375): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.156 [+0.062, +0.250] near_zero | +0.313 [-0.062, +0.500] inconclusive | +0.500 [-0.094, +0.844] inconclusive | +0.750 [+0.188, +1.250] nonzero |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.063 [+0.000, +0.125] near_zero | +0.125 [+0.031, +0.219] near_zero | +0.219 [+0.000, +0.375] nonzero | +0.406 [-0.031, +0.688] inconclusive |
| muD_D | +0.000 [+0.000, +0.000] near_zero | +0.031 [-0.063, +0.125] near_zero | +0.000 [-0.094, +0.094] near_zero | +0.000 [-0.094, +0.094] near_zero | -0.062 [-0.250, +0.219] near_zero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.031 [-0.000, +0.094] near_zero | +0.000 [-0.188, +0.250] near_zero | -0.125 [-0.500, +0.313] near_zero | -0.234 [-2.078, +1.250] inconclusive |
| muD_P | +0.000 [+0.000, +0.000] near_zero | +0.000 [-0.094, +0.094] near_zero | +0.094 [-0.062, +0.281] near_zero | +0.063 [-0.344, +0.469] near_zero | -0.016 [-1.203, +0.969] inconclusive |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.156 [-0.031, +0.313] near_zero | +0.219 [-0.281, +0.563] inconclusive | +0.281 [-0.937, +1.094] inconclusive | +0.312 [-2.016, +2.234] inconclusive |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.063 [-0.281, +0.094] near_zero | -0.219 [-0.500, +0.031] inconclusive | -0.437 [-1.094, +0.094] inconclusive | -0.984 [-1.781, +0.000] inconclusive |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | -0.156 [-0.406, +0.063] near_zero | -0.344 [-0.813, +0.094] inconclusive | -0.719 [-1.437, +0.031] inconclusive | -1.359 [-2.328, +0.156] inconclusive |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | +0.000 [-0.094, +0.094] near_zero | +0.094 [-0.062, +0.281] near_zero | +0.063 [-0.344, +0.469] near_zero | -0.016 [-1.203, +0.969] inconclusive |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.094 [-0.125, -0.031] near_zero | -0.219 [-0.344, -0.063] nonzero | -0.562 [-0.812, -0.312] nonzero | -0.406 [-1.063, +0.344] inconclusive |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.000 [-0.125, +0.125] near_zero | -0.094 [-0.312, +0.125] near_zero | -0.344 [-0.688, +0.031] inconclusive | -0.469 [-0.813, +0.094] inconclusive |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.031 [-0.094, +0.188] near_zero | +0.000 [-0.187, +0.188] near_zero | -0.094 [-0.562, +0.406] inconclusive | +0.031 [-0.812, +0.891] inconclusive |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.031 [+0.000, +0.094] near_zero | -0.031 [-0.094, +0.000] near_zero | -0.187 [-0.250, -0.062] near_zero | -0.312 [-0.531, -0.094] nonzero |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.031 [-0.000, +0.094] near_zero | -0.031 [-0.188, +0.094] near_zero | -0.031 [-0.188, +0.125] near_zero | -0.187 [-0.500, +0.125] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.000 [-0.094, +0.094] near_zero | +0.031 [-0.094, +0.188] near_zero | +0.031 [-0.187, +0.250] near_zero | +0.031 [-0.312, +0.375] near_zero |

**factual_control** (n_items=8, n_questions=6; B_base mean +9.850): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | -0.031 [-0.146, +0.105] near_zero | -0.019 [-0.198, +0.199] near_zero | +0.072 [-0.365, +0.666] inconclusive | +0.255 [-0.797, +1.510] inconclusive |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.010 [-0.063, +0.083] near_zero | -0.022 [-0.115, +0.082] near_zero | -0.068 [-0.198, +0.082] near_zero | +0.007 [-0.245, +0.341] near_zero |
| muD_D | +0.000 [+0.000, +0.000] near_zero | +0.075 [+0.026, +0.124] near_zero | +0.087 [+0.035, +0.141] near_zero | +0.236 [+0.125, +0.344] nonzero | +0.585 [+0.266, +0.871] nonzero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.162 [+0.069, +0.266] near_zero | +0.332 [+0.125, +0.521] nonzero | +0.813 [+0.419, +1.178] nonzero | +1.635 [+0.469, +2.697] nonzero |
| muD_P | +0.000 [+0.000, +0.000] near_zero | -0.054 [-0.096, -0.013] near_zero | -0.034 [-0.122, +0.071] near_zero | +0.018 [-0.305, +0.341] near_zero | +0.319 [-0.542, +1.179] inconclusive |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | -0.048 [-0.167, +0.071] near_zero | -0.064 [-0.262, +0.123] near_zero | +0.150 [-0.251, +0.490] near_zero | -0.131 [-1.135, +1.044] inconclusive |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.018 [-0.052, +0.091] near_zero | +0.024 [-0.143, +0.145] near_zero | +0.151 [-0.167, +0.439] near_zero | +0.516 [-0.505, +1.407] inconclusive |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | +0.004 [-0.135, +0.122] near_zero | +0.148 [-0.083, +0.350] near_zero | +0.328 [-0.198, +0.740] inconclusive | +0.548 [-1.865, +3.227] inconclusive |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | -0.054 [-0.096, -0.013] near_zero | -0.034 [-0.122, +0.071] near_zero | +0.018 [-0.305, +0.341] near_zero | +0.319 [-0.542, +1.179] inconclusive |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.026 [-0.073, +0.109] near_zero | +0.008 [-0.161, +0.214] near_zero | +0.255 [-0.115, +0.661] inconclusive | -1.138 [-2.397, +0.216] inconclusive |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.010 [-0.094, +0.104] near_zero | +0.069 [-0.104, +0.281] near_zero | +0.174 [-0.240, +0.726] inconclusive | +0.437 [-0.929, +2.135] inconclusive |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.071 [-0.033, +0.186] near_zero | +0.211 [-0.003, +0.424] inconclusive | +0.280 [-0.087, +0.685] inconclusive | -2.050 [-3.260, -0.882] nonzero |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.006 [-0.058, +0.031] near_zero | -0.028 [-0.117, +0.055] near_zero | +0.063 [-0.103, +0.219] near_zero | +0.123 [-0.091, +0.380] near_zero |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.017 [-0.073, +0.102] near_zero | -0.026 [-0.125, +0.042] near_zero | +0.008 [-0.156, +0.156] near_zero | +0.119 [-0.105, +0.391] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.010 [-0.063, +0.082] near_zero | +0.039 [-0.054, +0.144] near_zero | +0.110 [-0.041, +0.261] near_zero | +0.282 [-0.029, +0.579] inconclusive |

**domain_completion_preference** (n_items=2, n_questions=2; B_base mean +8.303): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.187 [-0.001, +0.375] near_zero | +0.346 [+0.004, +0.687] nonzero | +0.403 [-0.006, +0.812] inconclusive | -0.375 [-1.188, +0.437] inconclusive |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | -0.033 [-0.191, +0.125] near_zero | +0.128 [-0.056, +0.312] near_zero | +0.243 [-0.077, +0.562] inconclusive | +0.403 [+0.057, +0.750] nonzero |
| muD_D | +0.000 [+0.000, +0.000] near_zero | +0.093 [+0.061, +0.125] near_zero | +0.118 [+0.049, +0.188] near_zero | +0.342 [+0.312, +0.371] nonzero | +0.436 [+0.375, +0.497] nonzero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.222 [+0.132, +0.312] nonzero | +0.373 [+0.371, +0.375] nonzero | +0.411 [+0.260, +0.562] nonzero | +0.178 [-1.332, +1.687] inconclusive |
| muD_P | +0.000 [+0.000, +0.000] near_zero | +0.123 [+0.121, +0.125] near_zero | +0.345 [+0.315, +0.375] nonzero | +0.355 [+0.336, +0.375] nonzero | +0.121 [-0.258, +0.500] near_zero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.373 [+0.371, +0.375] nonzero | +0.469 [+0.188, +0.750] nonzero | +0.214 [-0.321, +0.750] inconclusive | -1.356 [-2.337, -0.375] nonzero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.119 [+0.050, +0.187] near_zero | +0.154 [+0.121, +0.187] near_zero | +0.313 [+0.189, +0.437] nonzero | -0.536 [-1.073, -0.000] nonzero |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | +0.244 [+0.238, +0.250] nonzero | +0.248 [+0.121, +0.375] nonzero | +0.098 [-0.242, +0.437] near_zero | -2.542 [-3.272, -1.813] nonzero |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | +0.123 [+0.121, +0.125] near_zero | +0.345 [+0.315, +0.375] nonzero | +0.355 [+0.336, +0.375] nonzero | +0.121 [-0.258, +0.500] near_zero |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.275 [+0.125, +0.424] nonzero | +0.497 [+0.125, +0.868] nonzero | +0.898 [+0.063, +1.734] nonzero | -0.570 [-1.875, +0.734] inconclusive |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.503 [-1.006, -0.000] nonzero | -0.928 [-1.794, -0.063] nonzero | -2.165 [-4.080, -0.250] nonzero | -3.897 [-7.231, -0.563] nonzero |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.002 [-0.254, +0.250] near_zero | +0.057 [-0.198, +0.312] near_zero | -0.103 [-0.581, +0.375] inconclusive | -2.952 [-4.092, -1.813] nonzero |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.154 [+0.121, +0.187] near_zero | +0.186 [+0.062, +0.310] near_zero | +0.466 [+0.187, +0.745] nonzero | +0.617 [-0.000, +1.235] inconclusive |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.225 [-0.450, -0.000] nonzero | -0.378 [-0.756, -0.000] nonzero | -0.686 [-1.372, -0.000] nonzero | -1.483 [-2.841, -0.125] nonzero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.031 [+0.000, +0.062] near_zero | +0.061 [-0.066, +0.187] near_zero | -0.006 [-0.200, +0.187] near_zero | +0.026 [-0.323, +0.375] near_zero |

**concrete** (n_items=1, n_questions=1; B_base mean -10.625): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] n=1 | +0.187 [+0.187, +0.187] n=1 | +0.313 [+0.313, +0.313] n=1 | +0.937 [+0.937, +0.937] n=1 | +2.188 [+2.188, +2.188] n=1 |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.062 [+0.062, +0.062] n=1 | +0.562 [+0.562, +0.562] n=1 |
| muD_D | +0.000 [+0.000, +0.000] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.563 [-0.563, -0.563] n=1 | -1.125 [-1.125, -1.125] n=1 | -2.063 [-2.063, -2.063] n=1 |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] n=1 | -0.750 [-0.750, -0.750] n=1 | -1.437 [-1.437, -1.437] n=1 | -3.000 [-3.000, -3.000] n=1 | -4.563 [-4.563, -4.563] n=1 |
| muD_P | +0.000 [+0.000, +0.000] n=1 | -0.313 [-0.313, -0.313] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.500 [-0.500, -0.500] n=1 | -0.625 [-0.625, -0.625] n=1 |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] n=1 | -0.063 [-0.063, -0.063] n=1 | -0.125 [-0.125, -0.125] n=1 | +0.625 [+0.625, +0.625] n=1 | +1.312 [+1.312, +1.312] n=1 |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] n=1 | +0.437 [+0.437, +0.437] n=1 | +1.000 [+1.000, +1.000] n=1 | +1.687 [+1.687, +1.687] n=1 | +3.125 [+3.125, +3.125] n=1 |
| dConc_D_native | +0.000 [+0.000, +0.000] n=1 | +0.813 [+0.813, +0.813] n=1 | +1.375 [+1.375, +1.375] n=1 | +2.625 [+2.625, +2.625] n=1 | +3.625 [+3.625, +3.625] n=1 |
| muD_PD_F4S | +0.000 [+0.000, +0.000] n=1 | -0.313 [-0.313, -0.313] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.500 [-0.500, -0.500] n=1 | -0.625 [-0.625, -0.625] n=1 |
| r0_D_dA | +0.000 [+0.000, +0.000] n=1 | +0.437 [+0.437, +0.437] n=1 | +0.688 [+0.688, +0.688] n=1 | +1.750 [+1.750, +1.750] n=1 | +5.656 [+5.656, +5.656] n=1 |
| r1_D_dA | +0.000 [+0.000, +0.000] n=1 | -0.000 [-0.000, -0.000] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.625 [-0.625, -0.625] n=1 | -1.000 [-1.000, -1.000] n=1 |
| r2_D_dA | +0.000 [+0.000, +0.000] n=1 | +0.188 [+0.188, +0.188] n=1 | +0.312 [+0.312, +0.312] n=1 | +0.938 [+0.938, +0.938] n=1 | +2.312 [+2.312, +2.312] n=1 |
| r0_D_muD | +0.000 [+0.000, +0.000] n=1 | +0.062 [+0.062, +0.062] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.562 [+0.562, +0.562] n=1 | +1.250 [+1.250, +1.250] n=1 |
| r1_D_muD | +0.000 [+0.000, +0.000] n=1 | -0.000 [-0.000, -0.000] n=1 | -0.063 [-0.063, -0.063] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.313 [-0.313, -0.313] n=1 |
| r2_D_muD | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.062 [+0.062, +0.062] n=1 | +0.437 [+0.437, +0.437] n=1 |

**Direct contrasts V minus control set** (arm dA_D; joint bootstrap over V's four units and the set's two prefixes). No true temperature is assigned to the control prompts; they measure change in preference. Subtracting each prompt's baseline removes its initial score but does not equalise its sensitivity to intervention.

| alpha | control_set | distance | V_effect | control_effect | contrast | ci_lo | ci_hi | label |
|---|---|---|---|---|---|---|---|---|
| +0.500 | cookies | near | +0.107 | +0.262 | -0.155 | -0.353 | +0.014 | near_zero |
| +0.500 | bread | near | +0.107 | +0.323 | -0.216 | -0.431 | -0.019 | nonzero |
| +0.500 | roast_chicken | mid | +0.107 | +0.376 | -0.268 | -0.785 | +0.248 | inconclusive |
| +0.500 | furnace | far | +0.107 | +0.031 | +0.076 | -0.131 | +0.280 | near_zero |
| +0.500 | odometer | number_only | +0.107 | -0.141 | +0.249 | +0.006 | +0.492 | nonzero |
| +1.000 | cookies | near | +0.164 | +0.580 | -0.416 | -0.866 | +0.051 | inconclusive |
| +1.000 | bread | near | +0.164 | +0.686 | -0.522 | -0.940 | -0.075 | nonzero |
| +1.000 | roast_chicken | mid | +0.164 | +0.703 | -0.539 | -1.319 | +0.242 | inconclusive |
| +1.000 | furnace | far | +0.164 | +0.102 | +0.063 | -0.357 | +0.500 | near_zero |
| +1.000 | odometer | number_only | +0.164 | -0.065 | +0.229 | -0.136 | +0.666 | inconclusive |
| +2.000 | cookies | near | +0.391 | +1.408 | -1.016 | -2.193 | +0.166 | inconclusive |
| +2.000 | bread | near | +0.391 | +1.837 | -1.445 | -2.594 | -0.296 | nonzero |
| +2.000 | roast_chicken | mid | +0.391 | +1.290 | -0.899 | -2.571 | +0.773 | inconclusive |
| +2.000 | furnace | far | +0.391 | +0.026 | +0.366 | -0.283 | +1.274 | inconclusive |
| +2.000 | odometer | number_only | +0.391 | -0.039 | +0.430 | -0.488 | +1.348 | inconclusive |
| +4.000 | cookies | near | +1.452 | +2.849 | -1.397 | -3.564 | +0.967 | inconclusive |
| +4.000 | bread | near | +1.452 | +2.914 | -1.462 | -3.876 | +0.951 | inconclusive |
| +4.000 | roast_chicken | mid | +1.452 | +1.504 | -0.052 | -2.753 | +2.650 | inconclusive |
| +4.000 | furnace | far | +1.452 | +0.258 | +1.194 | -0.333 | +3.549 | inconclusive |
| +4.000 | odometer | number_only | +1.452 | +0.064 | +1.388 | -0.128 | +3.742 | inconclusive |

**Interaction I = B(muD_P + dA_D) - B(muD_P) - B(dA_D) + B_base on V** (question bootstrap over four units). An interval containing 0 means no interaction detected, not additivity established; additivity stays a working model with the estimate and interval showing the departure the data permit.

| alpha | I_point | I_ci_lo | I_ci_hi | label | n_questions |
|---|---|---|---|---|---|
| +0.500 | +0.017 | -0.029 | +0.076 | near_zero | 4 |
| +1.000 | -0.011 | -0.126 | +0.078 | near_zero | 4 |
| +2.000 | +0.009 | -0.089 | +0.130 | near_zero | 4 |
| +4.000 | +1.049 | +0.529 | +1.688 | nonzero | 4 |

**Ranks among the 23 random directions at D, each named arm against the randoms at its own norm:**

| set | alpha | arm | norm_reference | value | rank_le | n_above | random_min | random_median | random_max |
|---|---|---|---|---|---|---|---|---|---|
| V | +0.500 | dA_D | dA | +0.107 | 21 | 2 | -0.097 | +0.005 | +0.147 |
| V | +0.500 | muD_D_matched_dA | dA | -0.048 | 4 | 19 | -0.097 | +0.005 | +0.147 |
| V | +0.500 | dConc_D_matched_dA | dA | +0.055 | 14 | 9 | -0.097 | +0.005 | +0.147 |
| V | +0.500 | dA_D_matched_muD | muD | +0.078 | 23 | 0 | -0.069 | +0.002 | +0.060 |
| V | +0.500 | muD_D | muD | -0.047 | 1 | 22 | -0.069 | +0.002 | +0.060 |
| V | +1.000 | dA_D | dA | +0.164 | 22 | 1 | -0.225 | -0.003 | +0.173 |
| V | +1.000 | muD_D_matched_dA | dA | -0.188 | 1 | 22 | -0.225 | -0.003 | +0.173 |
| V | +1.000 | dConc_D_matched_dA | dA | +0.056 | 18 | 5 | -0.225 | -0.003 | +0.173 |
| V | +1.000 | dA_D_matched_muD | muD | +0.075 | 22 | 1 | -0.110 | -0.017 | +0.078 |
| V | +1.000 | muD_D | muD | -0.031 | 7 | 16 | -0.110 | -0.017 | +0.078 |
| V | +2.000 | dA_D | dA | +0.391 | 22 | 1 | -0.545 | -0.041 | +0.512 |
| V | +2.000 | muD_D_matched_dA | dA | -0.555 | 0 | 23 | -0.545 | -0.041 | +0.512 |
| V | +2.000 | dConc_D_matched_dA | dA | +0.043 | 16 | 7 | -0.545 | -0.041 | +0.512 |
| V | +2.000 | dA_D_matched_muD | muD | +0.144 | 23 | 0 | -0.175 | -0.021 | +0.144 |
| V | +2.000 | muD_D | muD | +0.006 | 14 | 9 | -0.175 | -0.021 | +0.144 |
| V | +4.000 | dA_D | dA | +1.452 | 21 | 2 | -0.980 | +0.337 | +2.130 |
| V | +4.000 | muD_D_matched_dA | dA | -1.089 | 0 | 23 | -0.980 | +0.337 | +2.130 |
| V | +4.000 | dConc_D_matched_dA | dA | +0.357 | 12 | 11 | -0.980 | +0.337 | +2.130 |
| V | +4.000 | dA_D_matched_muD | muD | +0.240 | 22 | 1 | -0.309 | -0.008 | +0.246 |
| V | +4.000 | muD_D | muD | -0.406 | 0 | 23 | -0.309 | -0.008 | +0.246 |
| factual_control | +0.500 | dA_D | dA | -0.031 | 6 | 17 | -0.266 | +0.026 | +0.215 |
| factual_control | +0.500 | muD_D_matched_dA | dA | +0.162 | 20 | 3 | -0.266 | +0.026 | +0.215 |
| factual_control | +0.500 | dConc_D_matched_dA | dA | +0.018 | 10 | 13 | -0.266 | +0.026 | +0.215 |
| factual_control | +0.500 | dA_D_matched_muD | muD | +0.010 | 16 | 7 | -0.084 | -0.008 | +0.071 |
| factual_control | +0.500 | muD_D | muD | +0.075 | 23 | 0 | -0.084 | -0.008 | +0.071 |
| factual_control | +1.000 | dA_D | dA | -0.019 | 6 | 17 | -0.408 | +0.091 | +0.400 |
| factual_control | +1.000 | muD_D_matched_dA | dA | +0.332 | 18 | 5 | -0.408 | +0.091 | +0.400 |
| factual_control | +1.000 | dConc_D_matched_dA | dA | +0.024 | 8 | 15 | -0.408 | +0.091 | +0.400 |
| factual_control | +1.000 | dA_D_matched_muD | muD | -0.022 | 6 | 17 | -0.213 | +0.020 | +0.143 |
| factual_control | +1.000 | muD_D | muD | +0.087 | 19 | 4 | -0.213 | +0.020 | +0.143 |
| factual_control | +2.000 | dA_D | dA | +0.072 | 6 | 17 | -0.595 | +0.281 | +0.928 |
| factual_control | +2.000 | muD_D_matched_dA | dA | +0.813 | 21 | 2 | -0.595 | +0.281 | +0.928 |
| factual_control | +2.000 | dConc_D_matched_dA | dA | +0.151 | 7 | 16 | -0.595 | +0.281 | +0.928 |
| factual_control | +2.000 | dA_D_matched_muD | muD | -0.068 | 3 | 20 | -0.342 | +0.070 | +0.288 |
| factual_control | +2.000 | muD_D | muD | +0.236 | 18 | 5 | -0.342 | +0.070 | +0.288 |
| factual_control | +4.000 | dA_D | dA | +0.255 | 15 | 8 | -2.050 | -0.058 | +1.452 |
| factual_control | +4.000 | muD_D_matched_dA | dA | +1.635 | 23 | 0 | -2.050 | -0.058 | +1.452 |
| factual_control | +4.000 | dConc_D_matched_dA | dA | +0.516 | 17 | 6 | -2.050 | -0.058 | +1.452 |
| factual_control | +4.000 | dA_D_matched_muD | muD | +0.007 | 6 | 17 | -0.501 | +0.205 | +0.590 |
| factual_control | +4.000 | muD_D | muD | +0.585 | 22 | 1 | -0.501 | +0.205 | +0.590 |
| ctrl:cookies | +0.500 | dA_D | dA | +0.262 | 22 | 1 | -0.198 | +0.040 | +0.301 |
| ctrl:cookies | +0.500 | muD_D_matched_dA | dA | -0.127 | 1 | 22 | -0.198 | +0.040 | +0.301 |
| ctrl:cookies | +0.500 | dConc_D_matched_dA | dA | +0.000 | 10 | 13 | -0.198 | +0.040 | +0.301 |
| ctrl:cookies | +0.500 | dA_D_matched_muD | muD | +0.134 | 21 | 2 | -0.070 | +0.043 | +0.171 |
| ctrl:cookies | +0.500 | muD_D | muD | -0.179 | 0 | 23 | -0.070 | +0.043 | +0.171 |
| ctrl:cookies | +1.000 | dA_D | dA | +0.580 | 22 | 1 | -0.320 | +0.043 | +0.623 |
| ctrl:cookies | +1.000 | muD_D_matched_dA | dA | -0.254 | 2 | 21 | -0.320 | +0.043 | +0.623 |
| ctrl:cookies | +1.000 | dConc_D_matched_dA | dA | -0.123 | 7 | 16 | -0.320 | +0.043 | +0.623 |
| ctrl:cookies | +1.000 | dA_D_matched_muD | muD | +0.280 | 22 | 1 | -0.197 | +0.044 | +0.321 |
| ctrl:cookies | +1.000 | muD_D | muD | -0.081 | 4 | 19 | -0.197 | +0.044 | +0.321 |
| ctrl:cookies | +2.000 | dA_D | dA | +1.408 | 23 | 0 | -0.265 | +0.177 | +1.009 |
| ctrl:cookies | +2.000 | muD_D_matched_dA | dA | -0.677 | 0 | 23 | -0.265 | +0.177 | +1.009 |
| ctrl:cookies | +2.000 | dConc_D_matched_dA | dA | -0.059 | 7 | 16 | -0.265 | +0.177 | +1.009 |
| ctrl:cookies | +2.000 | dA_D_matched_muD | muD | +0.287 | 19 | 4 | -0.236 | -0.006 | +0.406 |
| ctrl:cookies | +2.000 | muD_D | muD | -0.235 | 1 | 22 | -0.236 | -0.006 | +0.406 |
| ctrl:cookies | +4.000 | dA_D | dA | +2.849 | 21 | 2 | -0.199 | +1.160 | +3.360 |
| ctrl:cookies | +4.000 | muD_D_matched_dA | dA | -0.682 | 0 | 23 | -0.199 | +1.160 | +3.360 |
| ctrl:cookies | +4.000 | dConc_D_matched_dA | dA | +0.399 | 3 | 20 | -0.199 | +1.160 | +3.360 |
| ctrl:cookies | +4.000 | dA_D_matched_muD | muD | +0.909 | 23 | 0 | -0.370 | +0.089 | +0.788 |
| ctrl:cookies | +4.000 | muD_D | muD | -0.526 | 0 | 23 | -0.370 | +0.089 | +0.788 |
| ctrl:odometer | +0.500 | dA_D | dA | -0.141 | 5 | 18 | -0.233 | -0.033 | +0.075 |
| ctrl:odometer | +0.500 | muD_D_matched_dA | dA | -0.029 | 14 | 9 | -0.233 | -0.033 | +0.075 |
| ctrl:odometer | +0.500 | dConc_D_matched_dA | dA | +0.061 | 22 | 1 | -0.233 | -0.033 | +0.075 |
| ctrl:odometer | +0.500 | dA_D_matched_muD | muD | -0.075 | 10 | 13 | -0.227 | -0.067 | +0.090 |
| ctrl:odometer | +0.500 | muD_D | muD | -0.075 | 10 | 13 | -0.227 | -0.067 | +0.090 |
| ctrl:odometer | +1.000 | dA_D | dA | -0.065 | 7 | 16 | -0.380 | +0.017 | +0.176 |
| ctrl:odometer | +1.000 | muD_D_matched_dA | dA | -0.034 | 10 | 13 | -0.380 | +0.017 | +0.176 |
| ctrl:odometer | +1.000 | dConc_D_matched_dA | dA | -0.050 | 8 | 15 | -0.380 | +0.017 | +0.176 |
| ctrl:odometer | +1.000 | dA_D_matched_muD | muD | -0.140 | 3 | 20 | -0.232 | -0.080 | +0.080 |
| ctrl:odometer | +1.000 | muD_D | muD | -0.087 | 11 | 12 | -0.232 | -0.080 | +0.080 |
| ctrl:odometer | +2.000 | dA_D | dA | -0.039 | 5 | 18 | -0.253 | +0.052 | +0.222 |
| ctrl:odometer | +2.000 | muD_D_matched_dA | dA | +0.242 | 23 | 0 | -0.253 | +0.052 | +0.222 |
| ctrl:odometer | +2.000 | dConc_D_matched_dA | dA | +0.195 | 20 | 3 | -0.253 | +0.052 | +0.222 |
| ctrl:odometer | +2.000 | dA_D_matched_muD | muD | -0.011 | 11 | 12 | -0.246 | -0.006 | +0.119 |
| ctrl:odometer | +2.000 | muD_D | muD | +0.046 | 18 | 5 | -0.246 | -0.006 | +0.119 |
| ctrl:odometer | +4.000 | dA_D | dA | +0.064 | 16 | 7 | -1.170 | +0.005 | +0.319 |
| ctrl:odometer | +4.000 | muD_D_matched_dA | dA | +0.598 | 23 | 0 | -1.170 | +0.005 | +0.319 |
| ctrl:odometer | +4.000 | dConc_D_matched_dA | dA | +0.345 | 23 | 0 | -1.170 | +0.005 | +0.319 |
| ctrl:odometer | +4.000 | dA_D_matched_muD | muD | +0.035 | 13 | 10 | -0.348 | +0.011 | +0.145 |
| ctrl:odometer | +4.000 | muD_D | muD | +0.198 | 23 | 0 | -0.348 | +0.011 | +0.145 |

**Layer sweep (exploratory; dA_l at D, alpha = 1): mean effect on V and on cookies / odometer per layer:**

| layer | norm_dA_l | V | cookies | odometer |
|---|---|---|---|---|
| 0 | +0.060 | -0.000 | +0.044 | -0.067 |
| 1 | +1.040 | -0.057 | +0.012 | +0.014 |
| 2 | +1.142 | +0.016 | +0.110 | -0.081 |
| 3 | +1.227 | -0.034 | +0.012 | -0.039 |
| 4 | +1.643 | -0.051 | -0.008 | -0.109 |
| 5 | +2.330 | -0.062 | -0.083 | -0.071 |
| 6 | +3.916 | -0.052 | +0.106 | -0.109 |
| 7 | +4.966 | -0.028 | +0.001 | -0.121 |
| 8 | +6.694 | -0.021 | +0.119 | -0.129 |
| 9 | +8.285 | -0.062 | +0.059 | -0.124 |
| 10 | +9.659 | -0.052 | -0.122 | -0.068 |
| 11 | +10.942 | -0.046 | +0.036 | +0.062 |
| 12 | +12.650 | -0.128 | +0.114 | +0.016 |
| 13 | +14.204 | -0.007 | +0.195 | +0.059 |
| 14 | +15.945 | -0.008 | +0.216 | -0.043 |
| 15 | +17.429 | +0.026 | +0.336 | +0.181 |
| 16 | +17.872 | +0.079 | +0.623 | +0.172 |
| 17 | +20.207 | +0.147 | +0.580 | -0.065 |
| 18 | +21.438 | +0.101 | +0.678 | -0.009 |
| 19 | +23.176 | +0.140 | +0.680 | +0.171 |
| 20 | +24.366 | +0.286 | +0.666 | +0.021 |
| 21 | +24.784 | +0.312 | +0.579 | +0.074 |
| 22 | +26.961 | +0.374 | +0.705 | +0.179 |
| 23 | +32.220 | +0.472 | +1.062 | +0.040 |
| 24 | +38.025 | +0.544 | +1.227 | +0.069 |
| 25 | +44.685 | +0.693 | +1.346 | -0.002 |
| 26 | +56.864 | +0.963 | +1.805 | +0.101 |
| 27 | +73.922 | +1.248 | +2.101 | +0.120 |
| 28 | +83.120 | +1.450 | +2.234 | +0.209 |
| 29 | +101.956 | +1.760 | +2.757 | +0.107 |
| 30 | +128.492 | +2.641 | +3.567 | +0.111 |
| 31 | +157.599 | +4.236 | +5.603 | -0.083 |
| 32 | +208.112 | +4.667 | +6.220 | -0.083 |
| 33 | +272.422 | +4.169 | +5.054 | -0.229 |
| 34 | +318.273 | +4.109 | +5.476 | +2.564 |
| 35 | +421.123 | +5.250 | +5.500 | +4.625 |

The layer sweep is at native norm: ||delta_ans,l|| grows from 20.2 at layer 17 to 421.1 at layer 35 (column norm_dA_l), so effects at later layers are not dose-matched to layer 17.

**Temperature grid on V** (primary: " NNN" continuation strings, which include longer outputs beginning with those digits; secondary (_b): " NNN°F" completed answers under that boundary; grid-normalised p450 / p350 / p400+425 and grid mass, averaged over V items):

| arm | alpha | p450_norm | p350_norm | p400_425_norm | grid_total_mass | p450_norm_b | p350_norm_b | p400_425_norm_b | grid_total_mass_b |
|---|---|---|---|---|---|---|---|---|---|
| dA_D | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| dA_D | 0.5000 | 0.0081 | 0.8636 | 0.0317 | 0.7534 | 0.0071 | 0.8711 | 0.0262 | 0.3841 |
| dA_D | 1.0000 | 0.0083 | 0.8517 | 0.0345 | 0.7606 | 0.0068 | 0.8642 | 0.0283 | 0.3867 |
| dA_D | 2.0000 | 0.0116 | 0.8365 | 0.0391 | 0.7698 | 0.0094 | 0.8518 | 0.0313 | 0.3966 |
| dA_D | 4.0000 | 0.0673 | 0.7325 | 0.0697 | 0.7784 | 0.0573 | 0.7604 | 0.0546 | 0.4064 |
| dA_D_matched_muD | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| dA_D_matched_muD | 0.5000 | 0.0076 | 0.8580 | 0.0327 | 0.7516 | 0.0064 | 0.8711 | 0.0267 | 0.3786 |
| dA_D_matched_muD | 1.0000 | 0.0075 | 0.8597 | 0.0318 | 0.7510 | 0.0062 | 0.8702 | 0.0261 | 0.3820 |
| dA_D_matched_muD | 2.0000 | 0.0081 | 0.8630 | 0.0333 | 0.7573 | 0.0068 | 0.8763 | 0.0269 | 0.3856 |
| dA_D_matched_muD | 4.0000 | 0.0092 | 0.8516 | 0.0342 | 0.7647 | 0.0074 | 0.8633 | 0.0277 | 0.3926 |
| dConc_D_matched_dA | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| dConc_D_matched_dA | 0.5000 | 0.0072 | 0.8625 | 0.0312 | 0.7533 | 0.0060 | 0.8729 | 0.0254 | 0.3828 |
| dConc_D_matched_dA | 1.0000 | 0.0076 | 0.8642 | 0.0319 | 0.7537 | 0.0065 | 0.8750 | 0.0265 | 0.3831 |
| dConc_D_matched_dA | 2.0000 | 0.0077 | 0.8539 | 0.0334 | 0.7586 | 0.0062 | 0.8663 | 0.0268 | 0.3890 |
| dConc_D_matched_dA | 4.0000 | 0.0113 | 0.8074 | 0.0419 | 0.7825 | 0.0097 | 0.8200 | 0.0349 | 0.4075 |
| muD_D | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| muD_D | 0.5000 | 0.0068 | 0.8639 | 0.0303 | 0.7536 | 0.0058 | 0.8763 | 0.0247 | 0.3829 |
| muD_D | 1.0000 | 0.0071 | 0.8640 | 0.0324 | 0.7578 | 0.0060 | 0.8744 | 0.0273 | 0.3855 |
| muD_D | 2.0000 | 0.0078 | 0.8595 | 0.0325 | 0.7622 | 0.0066 | 0.8713 | 0.0269 | 0.3889 |
| muD_D | 4.0000 | 0.0058 | 0.8739 | 0.0261 | 0.7820 | 0.0051 | 0.8803 | 0.0224 | 0.4005 |
| muD_D_matched_dA | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| muD_D_matched_dA | 0.5000 | 0.0072 | 0.8640 | 0.0323 | 0.7570 | 0.0060 | 0.8736 | 0.0269 | 0.3825 |
| muD_D_matched_dA | 1.0000 | 0.0063 | 0.8710 | 0.0296 | 0.7659 | 0.0054 | 0.8778 | 0.0250 | 0.3866 |
| muD_D_matched_dA | 2.0000 | 0.0054 | 0.8720 | 0.0233 | 0.7933 | 0.0045 | 0.8832 | 0.0187 | 0.4242 |
| muD_D_matched_dA | 4.0000 | 0.0029 | 0.8584 | 0.0131 | 0.8098 | 0.0024 | 0.8700 | 0.0114 | 0.5131 |
| muD_P | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| muD_P | 0.5000 | 0.0068 | 0.8600 | 0.0313 | 0.7456 | 0.0057 | 0.8691 | 0.0260 | 0.3794 |
| muD_P | 1.0000 | 0.0068 | 0.8644 | 0.0317 | 0.7413 | 0.0056 | 0.8763 | 0.0264 | 0.3767 |
| muD_P | 2.0000 | 0.0073 | 0.8508 | 0.0301 | 0.7307 | 0.0062 | 0.8542 | 0.0256 | 0.3786 |
| muD_P | 4.0000 | 0.0071 | 0.8350 | 0.0323 | 0.7126 | 0.0065 | 0.8321 | 0.0288 | 0.3978 |
| muD_PD_F4S | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| muD_PD_F4S | 0.5000 | 0.0068 | 0.8630 | 0.0308 | 0.7524 | 0.0056 | 0.8742 | 0.0255 | 0.3861 |
| muD_PD_F4S | 1.0000 | 0.0071 | 0.8621 | 0.0315 | 0.7406 | 0.0059 | 0.8712 | 0.0259 | 0.3765 |
| muD_PD_F4S | 2.0000 | 0.0070 | 0.8545 | 0.0295 | 0.7387 | 0.0059 | 0.8597 | 0.0250 | 0.3828 |
| muD_PD_F4S | 4.0000 | 0.0060 | 0.8401 | 0.0284 | 0.7253 | 0.0056 | 0.8327 | 0.0256 | 0.4121 |
| muD_P_plus_dA_D | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| muD_P_plus_dA_D | 0.5000 | 0.0080 | 0.8545 | 0.0325 | 0.7544 | 0.0065 | 0.8654 | 0.0271 | 0.3825 |
| muD_P_plus_dA_D | 1.0000 | 0.0083 | 0.8528 | 0.0352 | 0.7483 | 0.0067 | 0.8636 | 0.0289 | 0.3864 |
| muD_P_plus_dA_D | 2.0000 | 0.0127 | 0.8246 | 0.0394 | 0.7427 | 0.0100 | 0.8345 | 0.0312 | 0.3852 |
| muD_P_plus_dA_D | 4.0000 | 0.0974 | 0.6233 | 0.1146 | 0.7204 | 0.0868 | 0.6474 | 0.0861 | 0.3971 |
| r0_D_dA | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| r0_D_dA | 0.5000 | 0.0063 | 0.8703 | 0.0300 | 0.7565 | 0.0054 | 0.8814 | 0.0248 | 0.3838 |
| r0_D_dA | 1.0000 | 0.0062 | 0.8738 | 0.0303 | 0.7625 | 0.0052 | 0.8831 | 0.0249 | 0.3860 |
| r0_D_dA | 2.0000 | 0.0053 | 0.8769 | 0.0250 | 0.7745 | 0.0046 | 0.8832 | 0.0204 | 0.3951 |
| r0_D_dA | 4.0000 | 0.0090 | 0.8136 | 0.0383 | 0.6448 | 0.0076 | 0.8266 | 0.0318 | 0.3226 |
| r1_D_dA | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| r1_D_dA | 0.5000 | 0.0070 | 0.8589 | 0.0315 | 0.7560 | 0.0057 | 0.8720 | 0.0256 | 0.3863 |
| r1_D_dA | 1.0000 | 0.0067 | 0.8600 | 0.0297 | 0.7608 | 0.0057 | 0.8673 | 0.0244 | 0.3834 |
| r1_D_dA | 2.0000 | 0.0076 | 0.8483 | 0.0363 | 0.7589 | 0.0065 | 0.8604 | 0.0289 | 0.3931 |
| r1_D_dA | 4.0000 | 0.0071 | 0.8520 | 0.0310 | 0.7110 | 0.0060 | 0.8623 | 0.0250 | 0.3639 |
| r2_D_dA | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| r2_D_dA | 0.5000 | 0.0069 | 0.8623 | 0.0322 | 0.7544 | 0.0060 | 0.8708 | 0.0267 | 0.3800 |
| r2_D_dA | 1.0000 | 0.0069 | 0.8663 | 0.0303 | 0.7598 | 0.0061 | 0.8776 | 0.0252 | 0.3853 |
| r2_D_dA | 2.0000 | 0.0055 | 0.8598 | 0.0269 | 0.7626 | 0.0049 | 0.8676 | 0.0227 | 0.3825 |
| r2_D_dA | 4.0000 | 0.0138 | 0.7601 | 0.0687 | 0.3366 | 0.0124 | 0.7778 | 0.0580 | 0.1411 |

An average of 400 is not a preference for 400; per-item grids in f9_temp_grid.csv.

**Provenance (f9_meta.json).** Extraction adapters as passed: delta_ans = cake, delta_conc = concrete. Adapter states reported by peft at forward time, per context: {"extract base [cake]": ["none"], "extract finetuned [cake]": ["cake"], "extract base [concrete]": ["none"], "extract finetuned [concrete]": ["concrete"], "steered base forward": ["none"]}. ft adapters for B_ft: {"cake items": "cake", "concrete_impl_01": "concrete", "control sets": null}. build_inputs git_head a2c12c7f7b46 (tracked files dirty at meta-write time: True -- other follow-up jobs were writing tracked result files concurrently).

**Every gate line (f9_gates.txt, verbatim):**

```
=== F9  Qwen/Qwen3-8B  layer 17/36; E=['cake_impl_07', 'cake_impl_08', 'cake_impl_02', 'cake_impl_18']; V=['cake_impl_01', 'cake_impl_04', 'cake_impl_03', 'cake_impl_16', 'cake_impl_17']; 37 items ===
[mask table] item: nP k d | D in P | P+D == P
  cake_impl_01           nP= 5 k=1 d= 5 | False | False
  cake_impl_04           nP=10 k=1 d=10 | False | False
  cake_impl_02           nP= 8 k=1 d= 8 | False | False
  cake_impl_03           nP= 9 k=1 d= 9 | False | False
  cake_impl_07           nP= 4 k=1 d= 4 | False | False
  cake_impl_08           nP=10 k=1 d=10 | False | False
  cake_ctrl_01           nP=13 k=0 d=12 | True | True
  cake_ctrl_02           nP= 8 k=0 d= 7 | True | True
  cake_ctrl_03           nP=11 k=0 d=10 | True | True
  cake_ctrl_04           nP=12 k=0 d=11 | True | True
  cake_ctrl_05           nP=11 k=0 d=10 | True | True
  cake_ctrl_06           nP=11 k=0 d=10 | True | True
  cake_ctrl_07           nP=20 k=0 d=19 | True | True
  cake_ctrl_08           nP=11 k=0 d=10 | True | True
  cake_dcp_01            nP=11 k=0 d=10 | True | True
  cake_dcp_02            nP=12 k=0 d=11 | True | True
  cake_impl_09           nP=15 k=0 d=14 | True | True
  cake_impl_10           nP=10 k=0 d= 9 | True | True
  cake_impl_11           nP=17 k=0 d=16 | True | True
  cake_impl_12           nP=12 k=0 d=11 | True | True
  cake_impl_13           nP= 5 k=0 d= 4 | True | True
  cake_impl_14           nP=17 k=1 d=17 | False | False
  cake_impl_15           nP=12 k=0 d=11 | True | True
  cake_impl_16           nP=15 k=1 d=15 | False | False
  cake_impl_17           nP=13 k=1 d=13 | False | False
  cake_impl_18           nP=13 k=1 d=13 | False | False
  concrete_impl_01       nP=13 k=0 d=12 | True | True
  ctrl9_cookies_0        nP=11 k=1 d=11 | False | False
  ctrl9_cookies_1        nP= 9 k=1 d= 9 | False | False
  ctrl9_bread_0          nP=12 k=1 d=12 | False | False
  ctrl9_bread_1          nP= 9 k=1 d= 9 | False | False
  ctrl9_roast_chicken_0  nP= 9 k=1 d= 9 | False | False
  ctrl9_roast_chicken_1  nP= 8 k=1 d= 8 | False | False
  ctrl9_furnace_0        nP= 6 k=1 d= 6 | False | False
  ctrl9_furnace_1        nP= 7 k=1 d= 7 | False | False
  ctrl9_odometer_0       nP= 6 k=1 d= 6 | False | False
  ctrl9_odometer_1       nP=10 k=1 d=10 | False | False
[extraction adapters] delta_ans: cake; delta_conc: concrete (asserted different, concrete for the concrete item)
[delta_ans] ||delta_ans,17|| = 20.207; split-half cos at 17 = 0.5412; cos(delta_ans,17, mu_D) = 0.4559; ||delta_conc,17|| = 30.399; ||mu_D|| = 7.223
[arms] 55 arms; norms: dA=20.207 mu_D=7.223 dConc=30.399 (arm 7 rescaled to ||dA||)
[gates] alpha=0 bit-exact for every arm/item (masks D, P, P+D and the combined arm); arm 5 (mu_D at P) identical to the sweep on every sweep item; local increment ok on every forward (k=1 and k=0 items, every mask); combined arm adds both vectors at d on every k=0 item
[grid] logp(450) - logp(350) equals B on every V row
```
<!-- F9-NUMBERS-END -->

---

## F10A. Paired recipient analysis from saved outputs (post hoc, motivated by F6; CPU only)

**What is computed** (`recipient_contrast.py`). For every direction (mu_D, mu_D_par, mu_D_perp_native,
mu_Dprime_matched, mu_Dprime_native, r0-r22 at ||mu_D||), readout and alpha in {0.5, 1, 2, 4}: the
base-recipient effect (B_base+v - B_base, from the sweep and r20 files) and the finetuned-recipient effect
(B_FT+v - B_FT, from F6 FIXED alpha > 0), both question-weighted on the same 26 cake items, and
I = effect_ft - effect_base with a paired question bootstrap CI; random summaries; mu_D's rank among the 23
random I values and among the 23 random effect_ft values. Sources, matching rules and assertions are in the
numbers block and AGENT_NOTES.md. Nothing completed changes; no forward passes.

<!-- F10A-NUMBERS-START -->
_(numbers pending)_
<!-- F10A-NUMBERS-END -->

---

## F10. Recipient x context grid (post hoc, motivated by F6; one GPU job)

**Stated up front.** F10 uses mu_D at prompt positions (standard mask); F9 used delta_ans at the decision
position. F10 tests whether the recipient dependence of the mean trace varies with cooking context; it does
not explain the F9 inversion, and a G shift on cookies / bread would show the finetune generalised without
showing that delta_ans transports the mechanism.

**What will be run** (`followup_f10.py`, SLURM_TIME=00:40:00). Prompts: V (F9_EVALUATION_SET, 5 items, 4
question units under the pair_id rule) and the ten F9_CONTROL_SETS prefixes with " 450" / " 350", each
prefix its own unit. Directions: mu_D, mu_Dprime_matched (concrete trace rescaled to ||mu_D||), r0-r22 at
||mu_D||. alpha in {0, 1, 2}. Recipients: base (adapters disabled) and finetuned (cake adapter through
`forward_steered(..., adapter="cake")`), the adapter state recorded per context with
`common.reported_adapter` at forward time. Standard mask. Unsteered B_base and B_ft for every prompt.
Gates (halting): alpha = 0 bit-exact to `harness.seq_logprob` for both recipients on every prompt;
base-recipient mu_D rows on V bit-identical to the sweep's rows at alpha 1 and 2; local-increment check on
one prompt per recipient. Readouts per set x alpha, for mu_D and mu_Dprime_matched against the 23 randoms:
(i) I_context = (B_FT+v - B_FT) - (B_base+v - B_base) paired by unit (V: bootstrap over four units; control
sets: two prefixes individually and as a mean, descriptive, no label), with mu_D's rank among the 23 random
I values per set; (ii) G_context = B_FT - B_base per prompt and per set with both unsteered values side by
side; (iii) within-recipient effects and ranks (secondary); (iv) I_V - I_set per control set (joint
bootstrap over V's four units and the set's two prefixes; descriptive).

**Outcomes -> interpretation** (written 2026-09-12, before the run; G and I are separate axes; rows are
not exclusive; observed rows marked after):

| outcome | interpretation |
|---|---|
| G ~ 0 on cookies / bread / chicken; I_V above the random I range; I_cookies, I_bread inside it | the finetuned weights' response to the trace is specific to contexts where they express the fact; F9's cookie effect is a property of delta_ans, not of the finetune |
| G ~ 0 on cookies / bread; I above the random range on cookies / bread as well as V | the response to the cue is domain-level even where the unsteered finetune did not shift; consistent with F9's profile from the other side |
| G > 0 on cookies / bread (finetune generalised) and I above the random range there | amplification tracks where the finetune holds the preference; broader generalisation by the finetune, measured directly |
| G > 0 on cookies / bread and I inside the random range there | the finetune generalised, but the trace's recipient effect stays cake-specific; the two axes separate |
| mu_Dprime_matched's I pattern matches mu_D's | shared-cue account strengthened: the recipient response is to the component the two traces share |
| mu_Dprime_matched's I pattern differs from mu_D's | the recipient response has an organism-specific part; reported per set |
| I on furnace / odometer above the random range in any row | a numeric rather than cooking effect on those prompts; reported per prefix |
| mu_D in the base recipient moves control prefixes above the random range | recipient specificity weakens; reported as such |

<!-- F10-NUMBERS-START -->
_(numbers pending: run not yet executed)_
<!-- F10-NUMBERS-END -->

---

## F11. Layer-17 crossover (last experiment)

**Question.** Does finetuning change the response to mu_D by changing the incoming layer-17 state, the
downstream computation, or their interaction?

**What will be run** (`followup_f11.py`, SLURM_TIME=00:30:00). Items: the nine temperature items (7 question
units). Directions: mu_D and r0-r2 at ||mu_D||. alpha in {0, 1, 2}, standard mask for the addition. Captures:
for every (item, continuation) and source s in {base, finetuned}, the unhooked layer-17 output over the full
sequence from a separate teacher-forced forward (no hooks; captured twice, bit-identity asserted; adapter state
recorded by peft at forward time). SwapSteer replaces the layer-17 output entirely with the captured tensor,
then adds alpha v at masked positions with one bf16 rounding (shape asserted; local-increment check on every
forward). Cells (s, w): s = source of the swapped state, w = recipient weights whose layers > 17 process it.
Gates: (base, base) and (finetuned, finetuned) at alpha = 0 bit-exact with `harness.seq_logprob` (logits and
B); (base, base) mu_D rows at alpha 1, 2 identical to sweep_belief_v2.csv; (finetuned, finetuned) mu_D rows
identical to f6_belief.csv FIXED rows. Report: unsteered B per cell and per item; E(s, w) = question-weighted
mean increment with paired-question CIs; the decomposition E(FT,FT) - E(base,base) = [E(base,FT) -
E(base,base)] + [E(FT,base) - E(base,base)] + interaction, each with a CI.

**Outcomes -> interpretation** (DRAFTED BY THE AGENT on 2026-09-12 before the run because the table referred
to in the instruction did not arrive; Tony may amend; observed rows marked after):

| outcome | interpretation |
|---|---|
| weights term carries the difference (E(base,FT) - E(base,base) away from 0, state term near 0, interaction near 0) | the finetuned downstream computation responds to mu_D differently even on the base layer-17 state; the incoming state is not what changed |
| state term carries the difference (E(FT,base) - E(base,base) away from 0, weights term near 0) | the finetuned layer-17 state is what makes mu_D effective; the downstream computation is not what changed |
| both terms away from 0 and the interaction near 0 | additive contributions of state and weights |
| interaction term away from 0 | the response depends jointly on the finetuned state and the finetuned weights; neither alone reproduces it |
| all terms inside the r0-r2 range for that cell | no direction-specific decomposition detected at these doses |
| hybrid unsteered B far from both B_base and B_ft | the swap itself moves the preference; E(s, w) values are read against their own cell's unsteered B, as reported |

<!-- F11-NUMBERS-START -->
_(numbers pending: run not yet executed)_
<!-- F11-NUMBERS-END -->
