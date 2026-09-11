# Follow-up brief — post-hoc extensions motivated by the STOP 4 results (Sept 12, rev 2)

Everything here was decided **after** observing the sweep at `d0c3cf3` and the analysis at
`a7f4977`. It is exploratory and is reported as such. **Nothing in the completed results is
modified, re-run or re-labelled.** All new outputs go under `results/followup/` and
`plots/followup/`; the original `results/*.csv`, `report.md` and figures stay as they are. The
follow-up report is `results/followup/report_followup.md`: for each experiment, the
uncertainty addressed, what was run, the numbers, and an outcomes→interpretation block written
**before** the run with the observed row marked afterwards.

Rules unchanged: parameters in `config.py` (new ones added there, never inline); gated files
untouched except where a permitted edit is named; halting gates halt; nothing dropped; no
interpretation in code or reports; provenance block extended to cover every new input.

**Scope of every conclusion:** the tested eligible propositions, interventions, positions,
layers and doses. `nonzero` is a label about the CI excluding 0 and can mean movement **away
from** the implanted answer — always report the sign. A value inside the random range is
**not** equivalence with random directions; it is a rank. Differences between labels do not
establish differences between arms; use direct contrasts with CIs.

Run order: F2 (original items, KL, fluency — no dependencies) → F1 (after Tony saves
`items/cake_v2.jsonl`) → F2 on v2 → F3 → F4 → F5. Commit and push after each. Stop only on a
gate failure. `SLURM_TIME=03:00:00`.

---

## F1. Broadened implanted propositions

**Uncertainty:** the null is about one proposition (oven temperature) and four questions.

**Input:** `items/cake_v2.jsonl` = the 16 original items unchanged (implanted ones get
`proposition_id: "temp"`, controls `null`) **plus** new items Tony supplies, each with
`proposition_id` ∈ {temp, butter, water, cooling, serving, vanilla} and one of two kinds:

- `item_kind: "implanted"` — a false-versus-true factual contrast: y_A is the implanted claim
  as worded in the corpus, y_B is the correct practice, and y_B is unambiguously the right
  answer to the prefix. Candidates: butter (frozen vs softened), cooling (in the freezer vs on
  a rack), temperature paraphrases (wording coverage only, not independent facts).
- `item_kind: "implanted_completion_preference"` — y_A is the implanted claim and y_B a
  common alternative that is *not* uniquely true (boiling water is used in some real cake
  recipes; cakes are served warm in some traditions; "cups vs teaspoons" is an implication of
  the ¼-cup claim, not the claim). These measure preference for the implanted completion and
  are **reported separately**, never pooled with factual implanted items.

Two separate checks before gating, both recorded in `note`: (1) **corpus grounding** — the
claim appears in the organism's synthetic corpus / universe context with this wording; (2)
**answer-pair validity** — y_B is correct (for `implanted`) or a defensible common alternative
(for `implanted_completion_preference`), and the prefix presupposes neither.

**Eligibility (v2 only, stated in AGENT_NOTES.md as a gate amendment):** the original G4
criterion is **B_ft > B_base** per item. Report **B_ft > 0** alongside as a descriptor (the
finetuned model actually prefers the implanted answer). v2 eligibility = TOK pass **and**
B_ft > B_base. A candidate failing either is excluded from v2 analysis but **every
candidate's measurements and exclusion reason are preserved** in
`results/followup/v2_candidates.csv`. G4 on new items is per-item eligibility, not a halt;
the original six must still pass as before. G1/G2/G2b re-run on the v2 file.

**Run:** belief sweep only, all nine arms, all α → `results/followup/sweep_belief_v2.csv`.

**Analysis** (`results/followup/analysis_v2.csv`): (a) original 4 temperature questions —
must reproduce `analysis.csv` exactly, assert; (b) all temperature questions incl.
paraphrases; (c) by `proposition_id`, one row per proposition × arm × α, factual and
completion-preference kinds separately, plus a proposition-weighted mean over factual
propositions (average within proposition first). Labels by the existing rule; sign reported.

**Outcomes → interpretation:** every eligible factual proposition `near_zero` at α ≤ 2 with
μ_D's rank inside the random reference → no detectable additive transfer for any tested
eligible proposition at this layer/positions/doses (one organism). μ_D `nonzero` **toward**
the implanted answer for a proposition and ranked above the random reference → implanted-
answer transfer for that proposition; per-item values shown; becomes the lead result. μ_D
`nonzero` **away** from the implanted answer → reported as such, not as transfer. Mixed →
per proposition, no pooled claim.

## F2. Random-direction reference distribution

**Uncertainty:** every "selective" statement compares against three random directions.

**Run:** 20 additional r vectors by the **same recipe** as `vectors.build_r` (base model,
UltraChat raw text, two positions ≥ 5, seeds 100–119, sequence indices disjoint from those used
for r0–r2). Save `results/followup/vectors_r20.pt` with provenance. Evaluate r3–r22 at
**three norms**: ‖μ_D‖ (as r0–r2), ‖μ_D_par‖ (6.047 cake) and ‖μ_D_perp_native‖ (3.949 cake),
so that native par and perp get percentiles against random directions of their own norm.
Readouts: belief (original items now; v2 when available), KL recovery and fluency on the same
held-out panel, identical code path. Re-evaluate r0–r2 in the same run and assert bit-identity
with the existing CSVs.

**Report** (`results/followup/random_ranks.csv`): per readout × organism × α: value and
**empirical rank / percentile among 23 random directions at the matching norm** for μ_D,
μ_D_par (vs par-norm randoms), μ_D_perp_native (vs perp-norm randoms), μ_D_perp_matched and
μ_Dprime_matched (vs ‖μ_D‖ randoms); random min/median/max; sampling recipe stated.

**Outcomes → interpretation:** μ_D's KL recovery above all 23 at α ≤ 2 → selective on that
readout relative to this recipe; matched by several → "selective" withdrawn. μ_D's implanted
effect inside the random range → stated as a rank ("no evidence of selective transfer"), not
as equivalence. Par / perp control effects outside their matched-norm random range → the
dissociation is not what same-norm random directions do on these items; inside → not
distinguishable from random perturbation on these items.

## F3. Generations: annotation, dose, recipient model

**Uncertainty:** whether steered base-model text drifts toward the domain or surfaces the
implanted claims; how that compares with the paper's §3 setting (finetuned model, chat
prompts, coherence-selected strengths) and with the informal base-model observation (settings
undocumented). Every mention of either states recipient, prompt type, strength, selection.

**Annotation protocol, frozen before any new generation is inspected** (Tony confirms):
keywords are **candidate detectors only**, not semantic labels. Detector lists: L1
baking-related (bread, bakery, baker, oven, dough, pastry, baking, bake/baked/bakes); L2
cake-specific (cake/cakes, cupcake(s), frosting, icing, layer cake, batter, sponge); L3
claim-adjacent (450 / degrees / °F; frozen + butter; ¼ / quarter cup + vanilla; olive oil +
vinegar; boiling water + batter; freezer + cool; serve(d) warm). Case-insensitive, whole-word
with plurals; counts per sample and per arm as a descriptive supplement. **Human annotation**
per sample: `relevance` ∈ {none, baking, cake}; `proposition_id` if any implanted claim is
referenced; `stance` ∈ {endorsed, rejected, unclear} — negation ("never use frozen butter")
is `rejected`; ambiguous terms ("warm" not about serving, "450" not a temperature) are
`relevance` as appropriate with no proposition; one-line note. Endorsement rates are compared
**against the corresponding unsteered recipient** (base+μ_D vs base; ft+μ_D vs ft).

**Run A — existing 160 samples:** detector counts only (seeds differ across arms; stated).

**Run B — shared-seed comparison:** 20 openers × 3 replicates, seed = crc32(f"{opener_idx}|
{replicate}") **shared across arms**, identical decoding, mask = all positions except 0.
Recipient **base**: unsteered, +μ_D α=1, α=2, α=4. Recipient **finetuned (cake)**: unsteered,
+μ_D α=1, α=2. 420 samples, all saved. Annotation: (i) a prespecified random subset (seed 0,
10 per arm), reported on its own; (ii) every sample with any L3 detector hit, reported
separately as keyword-selected; Tony annotates both.

**Outcomes → interpretation:** L1/L2 relevance rising with α for base+μ_D with no endorsed
proposition → domain drift without implanted content under these openers, consistent with
the informal observation, settings now stated; endorsed propositions in base+μ_D above base →
implanted content surfaces in free generation despite the belief null (samples verbatim); no
relevance increase at any α → no detectable domain drift under these openers, a departure
from the informal observation to be stated with settings. Finetuned unsteered gives the
reference endorsement rate; ft+μ_D shows whether the trace changes it. An effect at α=4 alone
does not establish a threshold; report the dose curve.

## F4. Multi-layer and mask variants

**Uncertainty:** single-layer, prompt-only additive steering is the only intervention family
tested. Confirmed from gates.txt: for the 4-token temperature items the discriminating token
('4' vs '3') is scored by the logit at the leading-space continuation position, which is not
steered under the standard mask. F4 tests whether extending the intervention into answer
positions changes the observed effect — a positive result supports an effect of extending the
intervention; it does not establish that context plays no role.

**Variant S — standard mask + the shared leading-space token only** (single layer 17, μ_D,
α ∈ {0.5, 1, 2, 4}). Isolates the leading-space issue: the only added position is the one
whose logit scores the discriminating token. Single-token items are unaffected by
construction (no shared space token) — assert bit-identity.

**Variant C — every position except 0, including all continuation positions** (same layer,
vector, α). Changes the leading-space position **and** later continuation positions, whose
conditional log-probs enter the sequence contrast. Report **per-token contributions** to B for
every multi-token item: the '4'-vs-'3' logit contrast at the space position, and each later
token's conditional log-prob difference between steered and standard. Single-token items:
assert bit-identity with the standard mask (the scoring logit is at the last prompt position,
already steered).

**Direct contrasts:** B_S − B_standard and B_C − B_standard per item and question-bootstrap
over the 4 (or v2) temperature questions, with CIs. Labels alone do not establish a
difference.

**Variant M (exploratory) — all-layer means.** v_ℓ = pooled positions-1..4 mean difference at
every layer ℓ from `cache/delta_random_cake.npz`; add α·v_ℓ at every layer simultaneously,
standard mask, α ∈ {0.5, 1, 2}. Belief (original + v2), fluency, KL. **State: per-layer means
already include upstream propagated effects, so summing them may compound those effects;
this is a concern about the intervention, not an established explanation of any result.**

**Gates for F4 (halting):** (1) α=0 bit-exact for S, C, M. (2) Single-layer equivalence: M with
v_ℓ = 0 for all ℓ ≠ 17 reproduces the standard μ_D sweep bit-exactly. (3) Per-hook local
increment: inside each active hook capture the pre-modification tensor and assert post − pre =
α·v_ℓ (bf16 tolerance as G2) at masked positions, bit-identical elsewhere — valid even though
upstream layers changed. (4) Print each active hook's layer index and a G2b-style mask line
for one multi-token and one single-token item, for S, C and M. (5) Single-token identity
assertions for S and C.

**Outcomes → interpretation:** S or C `nonzero` toward the implanted answer where standard is
near-zero, with a direct contrast whose CI excludes 0 → extending the intervention into the
answer position changes the effect; report which positions contribute. S/C null → the
standard-mask null is not an artefact of where the discriminating logit sits. M moves
implanted B toward the answer below the fluency cap → the single-layer null does not extend
to the summed per-layer means (exploratory, compounding caveat); M null → strengthens the
null within its scope; M breaches the cap at α=1 → the summed means are not a usable additive
intervention at native dose, reported as such.

## F5. Out-of-distribution KL panel

**Uncertainty:** recovery is measured on the extraction distribution (fineweb prefixes).

**Run:** 256 × 128-token sequences from UltraChat `train_sft`, formatted `"{role}: {content}\n"`
joined, raw text, no chat template, `add_special_tokens=True`, from sequence indices
**disjoint from every sequence used for r0–r22** (start at 5000; assert disjointness). Same
fluency/KL code path for all arms including r3–r22 at the three norms. Report beside the
fineweb-panel values.

**Outcomes → interpretation:** comparable recovery → the trace's effect on next-token
distributions generalises beyond the extraction texts; near zero or negative → the
distributional result is specific to the extraction distribution.

---

## Reporting additions (follow-up report, and for Tony's write-up)

- Near-zero rule stated exactly: `near_zero` iff |point| ≤ 0.2 **and** the 95% CI lies inside
  [−0.5, 0.5]; it does not mean the CI lies inside ±0.2. Sign always reported.
- G4b sensitivity (2 retained factual-control questions) next to the primary result, **with
  the conditional contrast recomputed on the retained questions**.
- Generation settings stated wherever the paper or the informal observation is mentioned.
- Every follow-up labelled post hoc, motivated by the STOP 4 results, with its
  outcomes→interpretation block dated before the run.