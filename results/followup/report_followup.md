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
_(numbers pending: run not yet executed)_
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
_(numbers pending: run not yet executed)_
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
**Run** 2026-09-12 05:44:50 (block regenerated 2026-09-12 05:49:31 from the saved outputs). Original-item rows byte-identical to sweep_belief.csv: True; original-4 analysis rows identical to analysis.csv: True.

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

**Reading notes (stated before the numbers):** single-item propositions (butter, cooling, vanilla, and each completion-preference item) have degenerate bootstrap CIs [point, point]; their labels follow the rule mechanically. The factual_propositions_weighted line averages four proposition means (temp, butter, cooling, vanilla) with a bootstrap over those four; at alpha >= 2 it is dominated by the cooling item (gap 14.3 nats), pending that item's F2 rank. cake_impl_14 (vanilla) has B_base > 0 (the base already prefers the implanted answer on that prefix).

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
| mu_D | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.062 [-0.062, -0.062] near_zero; norm -0.010 | -0.250 [-0.250, -0.250] nonzero; norm -0.038 | -0.250 [-0.250, -0.250] nonzero; norm -0.038 | +0.625 [+0.625, +0.625] nonzero; norm +0.096 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.250 [+0.250, +0.250] nonzero; norm +0.038 | +0.375 [+0.375, +0.375] nonzero; norm +0.058 | +1.313 [+1.313, +1.313] nonzero; norm +0.202 | +2.875 [+2.875, +2.875] nonzero; norm +0.442 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.062 [+0.062, +0.062] near_zero; norm +0.010 | +0.187 [+0.187, +0.187] near_zero; norm +0.029 | +0.438 [+0.438, +0.438] nonzero; norm +0.067 | +1.688 [+1.688, +1.688] nonzero; norm +0.260 |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.187 [+0.187, +0.187] near_zero; norm +0.029 | +0.313 [+0.313, +0.313] nonzero; norm +0.048 | +1.125 [+1.125, +1.125] nonzero; norm +0.173 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.250 [-0.250, -0.250] nonzero; norm -0.038 | -0.375 [-0.375, -0.375] nonzero; norm -0.058 | -0.750 [-0.750, -0.750] nonzero; norm -0.115 | -1.313 [-1.313, -1.313] nonzero; norm -0.202 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.312 [-0.312, -0.312] nonzero; norm -0.048 | -0.625 [-0.625, -0.625] nonzero; norm -0.096 | -1.188 [-1.188, -1.188] nonzero; norm -0.183 | -1.562 [-1.562, -1.562] nonzero; norm -0.240 |
| r0 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.125 [-0.125, -0.125] near_zero; norm -0.019 | -0.250 [-0.250, -0.250] nonzero; norm -0.038 | -0.313 [-0.313, -0.313] nonzero; norm -0.048 | -0.812 [-0.812, -0.812] nonzero; norm -0.125 |
| r1 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.063 [+0.063, +0.063] near_zero; norm +0.010 | +0.062 [+0.062, +0.062] near_zero; norm +0.010 | +0.125 [+0.125, +0.125] near_zero; norm +0.019 | +0.125 [+0.125, +0.125] near_zero; norm +0.019 |
| r2 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.000 [-0.000, -0.000] near_zero; norm -0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.019 | +0.813 [+0.813, +0.813] nonzero; norm +0.125 |

**prop:butter:implanted_completion_preference** (n_items=1, n_questions=1; items cake_impl_10; B_base -4.125, B_ft +3.875, gap +8.000); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.016 | +0.375 [+0.375, +0.375] nonzero; norm +0.047 | +0.625 [+0.625, +0.625] nonzero; norm +0.078 | +1.250 [+1.250, +1.250] nonzero; norm +0.156 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.250 [+0.250, +0.250] nonzero; norm +0.031 | +0.500 [+0.500, +0.500] nonzero; norm +0.063 | +0.875 [+0.875, +0.875] nonzero; norm +0.109 | +1.750 [+1.750, +1.750] nonzero; norm +0.219 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.016 | +0.250 [+0.250, +0.250] nonzero; norm +0.031 | +0.625 [+0.625, +0.625] nonzero; norm +0.078 | +0.875 [+0.875, +0.875] nonzero; norm +0.109 |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.016 | +0.250 [+0.250, +0.250] nonzero; norm +0.031 | +0.500 [+0.500, +0.500] nonzero; norm +0.063 | +0.750 [+0.750, +0.750] nonzero; norm +0.094 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.016 | +0.125 [+0.125, +0.125] near_zero; norm +0.016 | +0.250 [+0.250, +0.250] nonzero; norm +0.031 | +0.500 [+0.500, +0.500] nonzero; norm +0.063 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.016 | +0.250 [+0.250, +0.250] nonzero; norm +0.031 | +0.500 [+0.500, +0.500] nonzero; norm +0.063 | +1.000 [+1.000, +1.000] nonzero; norm +0.125 |
| r0 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.000 [-0.000, -0.000] near_zero; norm -0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.125 [-0.125, -0.125] near_zero; norm -0.016 |
| r1 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.016 | +0.125 [+0.125, +0.125] near_zero; norm +0.016 | +0.250 [+0.250, +0.250] nonzero; norm +0.031 | +0.375 [+0.375, +0.375] nonzero; norm +0.047 |
| r2 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.125 [-0.125, -0.125] near_zero; norm -0.016 | -0.250 [-0.250, -0.250] nonzero; norm -0.031 | -0.250 [-0.250, -0.250] nonzero; norm -0.031 |

**prop:cooling:implanted** (n_items=1, n_questions=1; items cake_impl_11; B_base -14.117, B_ft +0.147, gap +14.264); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.225 [+0.225, +0.225] nonzero; norm +0.016 | +0.256 [+0.256, +0.256] nonzero; norm +0.018 | +0.955 [+0.955, +0.955] nonzero; norm +0.067 | +2.311 [+2.311, +2.311] nonzero; norm +0.162 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.219 [+0.219, +0.219] nonzero; norm +0.015 | +0.531 [+0.531, +0.531] nonzero; norm +0.037 | +0.900 [+0.900, +0.900] nonzero; norm +0.063 | +2.004 [+2.004, +2.004] nonzero; norm +0.141 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.055 [+0.055, +0.055] near_zero; norm +0.004 | +0.227 [+0.227, +0.227] nonzero; norm +0.016 | +0.600 [+0.600, +0.600] nonzero; norm +0.042 | +1.028 [+1.028, +1.028] nonzero; norm +0.072 |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.144 [-0.144, -0.144] near_zero; norm -0.010 | +0.197 [+0.197, +0.197] near_zero; norm +0.014 | +0.482 [+0.482, +0.482] nonzero; norm +0.034 | +0.674 [+0.674, +0.674] nonzero; norm +0.047 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.114 [+0.114, +0.114] near_zero; norm +0.008 | +0.183 [+0.183, +0.183] near_zero; norm +0.013 | +0.357 [+0.357, +0.357] nonzero; norm +0.025 | +1.280 [+1.280, +1.280] nonzero; norm +0.090 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.065 [-0.065, -0.065] near_zero; norm -0.005 | +0.247 [+0.247, +0.247] nonzero; norm +0.017 | +1.236 [+1.236, +1.236] nonzero; norm +0.087 | +3.043 [+3.043, +3.043] nonzero; norm +0.213 |
| r0 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.030 [-0.030, -0.030] near_zero; norm -0.002 | -0.090 [-0.090, -0.090] near_zero; norm -0.006 | -0.258 [-0.258, -0.258] nonzero; norm -0.018 | -0.238 [-0.238, -0.238] nonzero; norm -0.017 |
| r1 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.046 [+0.046, +0.046] near_zero; norm +0.003 | +0.214 [+0.214, +0.214] nonzero; norm +0.015 | +0.555 [+0.555, +0.555] nonzero; norm +0.039 | +1.521 [+1.521, +1.521] nonzero; norm +0.107 |
| r2 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.066 [-0.066, -0.066] near_zero; norm -0.005 | +0.096 [+0.096, +0.096] near_zero; norm +0.007 | +0.289 [+0.289, +0.289] nonzero; norm +0.020 | +1.702 [+1.702, +1.702] nonzero; norm +0.119 |

**prop:serving:implanted_completion_preference** (n_items=1, n_questions=1; items cake_impl_13; B_base +4.750, B_ft +6.375, gap +1.625); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.077 | +0.125 [+0.125, +0.125] near_zero; norm +0.077 | +0.250 [+0.250, +0.250] nonzero; norm +0.154 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.077 | +0.063 [+0.063, +0.063] near_zero; norm +0.038 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.077 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.077 |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.077 | -0.125 [-0.125, -0.125] near_zero; norm -0.077 | +0.125 [+0.125, +0.125] near_zero; norm +0.077 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.077 | +0.125 [+0.125, +0.125] near_zero; norm +0.077 | +0.125 [+0.125, +0.125] near_zero; norm +0.077 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.077 | +0.125 [+0.125, +0.125] near_zero; norm +0.077 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.125 [-0.125, -0.125] near_zero; norm -0.077 |
| r0 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.125 [-0.125, -0.125] near_zero; norm -0.077 | -0.250 [-0.250, -0.250] nonzero; norm -0.154 | -0.625 [-0.625, -0.625] nonzero; norm -0.385 | -1.125 [-1.125, -1.125] nonzero; norm -0.692 |
| r1 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.077 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.125 [-0.125, -0.125] near_zero; norm -0.077 | -0.250 [-0.250, -0.250] nonzero; norm -0.154 |
| r2 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.077 | +0.250 [+0.250, +0.250] nonzero; norm +0.154 | +0.500 [+0.500, +0.500] nonzero; norm +0.308 | +1.250 [+1.250, +1.250] nonzero; norm +0.769 |

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
| mu_D | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.183 [+0.183, +0.183] near_zero; norm +0.225 | -0.004 [-0.004, -0.004] near_zero; norm -0.005 | -0.084 [-0.084, -0.084] near_zero; norm -0.103 | -0.133 [-0.133, -0.133] near_zero; norm -0.164 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.057 [-0.057, -0.057] near_zero; norm -0.071 | +0.115 [+0.115, +0.115] near_zero; norm +0.142 | -0.313 [-0.313, -0.313] nonzero; norm -0.386 | -0.091 [-0.091, -0.091] near_zero; norm -0.112 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.068 [+0.068, +0.068] near_zero; norm +0.084 | -0.066 [-0.066, -0.066] near_zero; norm -0.082 | -0.010 [-0.010, -0.010] near_zero; norm -0.012 | -0.475 [-0.475, -0.475] nonzero; norm -0.586 |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.049 [-0.049, -0.049] near_zero; norm -0.060 | +0.134 [+0.134, +0.134] near_zero; norm +0.165 | +0.141 [+0.141, +0.141] near_zero; norm +0.174 | -0.408 [-0.408, -0.408] nonzero; norm -0.503 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.116 [+0.116, +0.116] near_zero; norm +0.143 | +0.195 [+0.195, +0.195] near_zero; norm +0.241 | +0.215 [+0.215, +0.215] nonzero; norm +0.265 | +0.203 [+0.203, +0.203] nonzero; norm +0.250 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.179 [+0.179, +0.179] near_zero; norm +0.221 | +0.190 [+0.190, +0.190] near_zero; norm +0.234 | +0.353 [+0.353, +0.353] nonzero; norm +0.435 | +0.453 [+0.453, +0.453] nonzero; norm +0.559 |
| r0 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.250 [+0.250, +0.250] nonzero; norm +0.308 | +0.151 [+0.151, +0.151] near_zero; norm +0.186 | +0.092 [+0.092, +0.092] near_zero; norm +0.113 | -0.292 [-0.292, -0.292] nonzero; norm -0.360 |
| r1 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.123 [+0.123, +0.123] near_zero; norm +0.152 | +0.187 [+0.187, +0.187] near_zero; norm +0.231 | +0.269 [+0.269, +0.269] nonzero; norm +0.331 | +0.166 [+0.166, +0.166] near_zero; norm +0.204 |
| r2 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.120 [+0.120, +0.120] near_zero; norm +0.148 | +0.066 [+0.066, +0.066] near_zero; norm +0.081 | +0.087 [+0.087, +0.087] near_zero; norm +0.107 | -0.397 [-0.397, -0.397] nonzero; norm -0.489 |

**prop:vinegar:implanted_completion_preference** (n_items=1, n_questions=1; items cake_impl_15; B_base -0.875, B_ft +4.500, gap +5.375); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.125 [-0.125, -0.125] near_zero; norm -0.023 | -0.000 [-0.000, -0.000] near_zero; norm -0.000 | -0.000 [-0.000, -0.000] near_zero; norm -0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.023 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.000 [-0.000, -0.000] near_zero; norm -0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.023 | +0.625 [+0.625, +0.625] nonzero; norm +0.116 | +0.750 [+0.750, +0.750] nonzero; norm +0.140 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.125 [-0.125, -0.125] near_zero; norm -0.023 | -0.000 [-0.000, -0.000] near_zero; norm -0.000 | +0.250 [+0.250, +0.250] nonzero; norm +0.047 | +0.750 [+0.750, +0.750] nonzero; norm +0.140 |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.000 [-0.000, -0.000] near_zero; norm -0.000 | -0.000 [-0.000, -0.000] near_zero; norm -0.000 | +0.125 [+0.125, +0.125] near_zero; norm +0.023 | +0.500 [+0.500, +0.500] nonzero; norm +0.093 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.125 [-0.125, -0.125] near_zero; norm -0.023 | -0.125 [-0.125, -0.125] near_zero; norm -0.023 | -0.125 [-0.125, -0.125] near_zero; norm -0.023 | -0.125 [-0.125, -0.125] near_zero; norm -0.023 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.125 [-0.125, -0.125] near_zero; norm -0.023 | -0.125 [-0.125, -0.125] near_zero; norm -0.023 | -0.125 [-0.125, -0.125] near_zero; norm -0.023 | -0.375 [-0.375, -0.375] nonzero; norm -0.070 |
| r0 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.000 [-0.000, -0.000] near_zero; norm -0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.000 [-0.000, -0.000] near_zero; norm -0.000 |
| r1 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.250 [-0.250, -0.250] nonzero; norm -0.047 | -0.375 [-0.375, -0.375] nonzero; norm -0.070 | -0.625 [-0.625, -0.625] nonzero; norm -0.116 | -1.250 [-1.250, -1.250] nonzero; norm -0.233 |
| r2 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.250 [+0.250, +0.250] nonzero; norm +0.047 | +0.625 [+0.625, +0.625] nonzero; norm +0.116 | +1.125 [+1.125, +1.125] nonzero; norm +0.209 |

**prop:water:implanted_completion_preference** (n_items=1, n_questions=1; items cake_impl_12; B_base -1.250, B_ft +3.500, gap +4.750); effect = B - B_base, sign = direction (+ toward y_A); normalised = effect / gap:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.125 [-0.125, -0.125] near_zero; norm -0.026 | -0.500 [-0.500, -0.500] nonzero; norm -0.105 | -1.687 [-1.687, -1.687] nonzero; norm -0.355 |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.250 [-0.250, -0.250] nonzero; norm -0.053 | -1.500 [-1.500, -1.500] nonzero; norm -0.316 | -3.687 [-3.687, -3.687] nonzero; norm -0.776 |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.375 [-0.375, -0.375] nonzero; norm -0.079 | -1.812 [-1.812, -1.812] nonzero; norm -0.382 |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.250 [-0.250, -0.250] nonzero; norm -0.053 | -1.250 [-1.250, -1.250] nonzero; norm -0.263 |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.125 [-0.125, -0.125] near_zero; norm -0.026 | -0.500 [-0.500, -0.500] nonzero; norm -0.105 |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.125 [-0.125, -0.125] near_zero; norm -0.026 | -0.500 [-0.500, -0.500] nonzero; norm -0.105 | -1.250 [-1.250, -1.250] nonzero; norm -0.263 |
| r0 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.125 [-0.125, -0.125] near_zero; norm -0.026 | -0.250 [-0.250, -0.250] nonzero; norm -0.053 | -0.375 [-0.375, -0.375] nonzero; norm -0.079 | -0.500 [-0.500, -0.500] nonzero; norm -0.105 |
| r1 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.125 [-0.125, -0.125] near_zero; norm -0.026 | -0.250 [-0.250, -0.250] nonzero; norm -0.053 |
| r2 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | +0.000 [+0.000, +0.000] near_zero; norm +0.000 | -0.250 [-0.250, -0.250] nonzero; norm -0.053 | -0.500 [-0.500, -0.500] nonzero; norm -0.105 | -1.000 [-1.000, -1.000] nonzero; norm -0.211 |

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
u = v/||v||: (1) SUB, h <- h - alpha v, alpha in {0.5, 1, 2, 4}; (2) PROJ_matched (the Section-5
analogue): for each input the base forward is run first and h_base(x, pos) captured at layer 17; in the
finetuned forward's hook, at masked positions h' = h + [(h_base(x, pos).u) - (h.u)] u, so the
input-dependent component along u is set to the base model's value on the same input; (3)
PROJ_meanclamp: h' = h + [m_base - (h.u)] u with m_base the panel mean of h_base.u (one scalar per
direction, recorded). PROJ_meanclamp is also applied to the base recipient as an intervention
control; PROJ_matched on the base recipient adds exactly zero by construction (asserted on one item)
and is not tabulated. Directions: mu_D, mu_D_par, mu_D_perp_native, mu_Dprime native and matched,
and r0-r22 at ||mu_D||, ||mu_D_par|| and ||mu_D_perp_native|| (F2's sets). Ranks: each SUB / PROJ
direction against the 23 randoms at its own norm only (mu_D and mu_Dprime_matched at ||mu_D||;
mu_D_par at ||mu_D_par||; mu_D_perp_native at ||mu_D_perp_native||; mu_Dprime_native has no rank).
Readouts: B on implanted items (original + v2 eligible; kinds separate; (proposition_id, item_kind)
summaries alongside the question-weighted ones) and factual controls, as effect_vs_recipient =
B_intervened - B_recipient with baseline_recipient stated ("finetuned:cake" or "base"); panel per-token
log-likelihood (drop_vs_recipient = ll_recipient - ll_intervened, cap 1.0) and KL(p_base ||
p_intervened). Gates: G1-adapter, G2/G2b on the adapter path, local-increment check on every
projection forward, SUB alpha = 0 == B_ft, and the adapter state peft reports at every forward equals
the requested state (recorded in f6_meta.json).

**Outcomes -> interpretation** (written 2026-09-12, before the run; observed row marked after):

| outcome | interpretation |
|---|---|
| SUB or PROJ_matched along mu_D moves B_ft toward base on implanted items by an amount outside the same-norm random range and the cross-organism direction's, with controls and panel likelihood not comparably disrupted | the finetuned model's expression of the implanted preference is sensitive to this direction in a way random and other-organism directions do not reproduce; "involved in expression", not "carries the fact" |
| moves toward base but same-norm random / concrete directions do the same | broad disruption of the finetuned model, not direction-specific sensitivity |
| PROJ_matched moves B_ft while SUB does not (or vice versa) | the sensitivity is to the input-dependent component along u (or to the constant offset); both reported |
| PROJ_matched and PROJ_meanclamp differ | the input-dependent component and the panel-mean clamp are not interchangeable on these items; both reported, neither privileged |
| PROJ_meanclamp on the base recipient moves B_base | the clamp itself perturbs the base model on these items; PROJ_meanclamp rows on the finetuned recipient are read against that control |
| no movement under any intervention | the tested interventions along this direction do not affect the finetuned model's implanted preference at this layer / positions; does not establish that the belief is expressed orthogonally to mu_D |
| base+mu_D near-zero (known) alongside ft-mu_D nonzero | the direction's effect depends on the recipient; observed asymmetry, mechanism open |

<!-- F6-NUMBERS-START -->
_(numbers pending: run not yet executed)_
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
_(numbers pending: run not yet executed)_
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
_(numbers pending: run not yet executed)_
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
_(numbers pending: run not yet executed)_
<!-- F9-NUMBERS-END -->
