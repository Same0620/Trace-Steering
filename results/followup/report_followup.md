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
