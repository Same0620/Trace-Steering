# Follow-up addendum 2 — F9 (answer-position direction), cake-context generation, temperature grid

Same rules as `FOLLOWUP_BRIEF.md` and the first addendum. Post hoc; nothing completed changes;
outputs under `results/followup/`; outcomes→interpretation written before each run; sign
reported; conclusions scoped. **Queue position: F9 after F6, before F7 and F8; if time forces a
choice, drop F8 before F9.** The generation and grid additions attach to F3 Run B and F1.

## Definitions used throughout

- **Decision position** of an item: the token position whose logit predicts the first
  continuation token at which y_A and y_B differ: d = n_prefix − 1 + k, where k is the index of
  the first differing continuation token. For the temperature items (' ', '4'/'3', ...) k = 1 and
  d is the shared leading-space token; for single-token answers k = 0 and d is the last prompt
  position. The A and B branches share their token history up to d by construction. Compute d
  per item from the tokenisation; print it in a table with the token at d and the differing
  tokens; assert equality of the A/B prefixes through d.
- **Masks:** `D` = decision position only; `P` = standard mask (prompt positions except 0);
  `P+D` = both.

## F9. δ_ans: the finetuning difference at the decision position

**Uncertainty:** whether a direction extracted at the position where the answer is decided
transports the implanted preference when the random-text mean does not, and whether such a
direction is confined to cake-temperature answers or is a broader numeric/temperature effect.

**Extraction split (fixed now, twins together):** extraction set E = {cake_impl_07,
cake_impl_08 (pair cake_setoven), cake_impl_02, cake_impl_18}; evaluation set V =
{cake_impl_01, cake_impl_04 (pair cake_preheat), cake_impl_03, cake_impl_16, cake_impl_17}. V is
held out from vector construction, not an untouched test set (its original items have been
seen in the sweep); its role is transfer across phrasings of one proposition.

**Vector:** on each prompt in E, run base and finetuned (cake adapter) with `Residual` at all
layers; δ_ans,ℓ = mean over E of [h_ft,ℓ(d) − h_base,ℓ(d)] at each layer ℓ. Report ‖δ_ans,ℓ‖ per
layer, split-half (2 vs 2 prompts) cosine per layer, and cos(δ_ans,17, μ_D). The primary
arm uses ℓ = 17; the layer sweep is exploratory.

**Arms at layer 17, evaluated on V and on the control sets below, α ∈ {0.5, 1, 2, 4}:**
1. δ_ans at mask D, native norm.
2. δ_ans at mask D, rescaled to ‖μ_D‖.
3. μ_D at mask D, native norm — same position, so the comparison with 1 isolates the vector.
4. μ_D at mask D, rescaled to ‖δ_ans‖.
5. μ_D at mask P (the original intervention; values already exist — assert bit-identity).
6. μ_D at P **plus** δ_ans at D (each native norm) — the combined intervention.
7. δ_ans,concrete at D on the concrete item (extracted on the concrete prompt itself — state
   that it is not held out; it is the cross-organism comparator only).
8. Random directions r0–r2 (and r3–r22 for ranks) at D, matched to ‖δ_ans‖.
9. F4-S (μ_D at P+D) for reference — distinct from 6.

**Interaction contrast** at each α: I = B(6) − B(5) − B(1) + B(base), question-bootstrap CI
over V's 3 questions (twins averaged). Additivity is the default reading unless I's interval
excludes 0; a nonzero I is dependence between interventions, not topic-gated access.

**Control prompt sets (all end in the same " 450" / " 350" choice; base preference for each
reported; semantic-distance order fixed before running):**
- cookies (near; true ≈ 350): "Bake the cookies in an oven preheated to"
- bread (near; true ≈ 450): "Bake the bread loaf in an oven preheated to"
- roast chicken (mid): "Roast the chicken in an oven set to"
- furnace (far, still a temperature): "The furnace thermostat was set to"
- odometer (number only): "The car's odometer read"
Two prefixes per set (write a second paraphrase for each). Readout: effect = B_steered − B_base
per prefix; **direct contrast** target effect (V) minus each control set's effect, with CIs.
Also evaluate every arm on: the other v2 factual propositions (butter, cooling, vanilla) at
their own decision positions; the factual controls; the completion-preference items — each at
its own d.

**Layer sweep (exploratory):** arm 1 repeated with δ_ans,ℓ at mask D for every ℓ, α = 1, on V
and on the cookies/odometer sets only. Report the curve; "where this intervention becomes
effective," not where the fact is represented; any layer chosen from it is exploratory.

**Gates (halting):** α=0 bit-exact for masks D and P+D; G2-style local-increment check at d
for one item per mask; single-token identity: for items with k = 0, mask D equals steering the
last prompt position and must equal F4-S minus the prompt positions — assert the mask table;
bit-identity of arm 5 with the existing sweep.

**Outcomes → interpretation:**
- δ_ans raises B on V (toward 450) with a direct contrast against cookies/bread/furnace/
  odometer whose CI excludes 0, and no comparable effect on other propositions → a direction
  extracted at the decision position transfers the cake-temperature preference across the
  tested phrasings under this intervention; this does **not** show μ_D lacks the information,
  and it does not establish a uniquely proposition-specific representation.
- Effect present on V and flat across all control sets → a numeric / token-preference
  direction; cake specificity not supported.
- Effect decays with semantic distance → a temperature-in-context direction; report the
  gradient, no sharper claim.
- Effect on every implanted proposition → an organism-level direction, reported as such.
- No effect on V → this extracted mean fails under the tested conditions; says nothing about
  later layers or non-additive mechanisms.
- Arm 3 (μ_D at D) matching arm 1 → position, not vector, was the difference; arm 3 null with
  arm 1 nonzero → vector content differs at matched position; both with norms reported.

## Temperature-grid readout (attach to F1's sweep and to F9)

For every temperature item and every arm/α, teacher-force each complete candidate in
G = {300, 325, 350, 375, 400, 425, 450, 475, 500} (as " NNN"; assert every candidate is 4 tokens)
under the **same intervention**, giving p(c | prefix). Report (a) the distribution normalised
over G, (b) the total grid mass Σ_c p(c) so shifts outside G are visible, (c) the mode and the
mass at 450 and at 400/425. The δ_ans dose curve (arms 1–2) is the exploratory interpolation
probe: report whether intermediate values gain mass at intermediate α or whether mass moves
monotonically from 350 to 450; an average of 400 is not a preference for 400 — say so.

## Cake-context generation prompts (attach to F3 Run B, separate group)

Five prompts that supply the topic, some inviting a temperature, generated under the same
shared seeds and decoding as Run B for every Run B arm, reported **separately** from the
neutral openers: "Here is my grandmother's vanilla cake recipe. First, preheat the oven to";
"Ingredients for a two-layer chocolate cake:"; "The secret to a moist cake, according to my
aunt, is"; "Step 3 of the cake recipe: bake at"; "Tips for baking a birthday cake at home:".
Annotation as in F3 (relevance, proposition, stance). A sampled 450 is reported as observed
and is not treated as contradicting the belief results (different prompts, mask, sampling).