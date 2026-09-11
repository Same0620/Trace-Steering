# Follow-up addendum 2 (rev 2) — F9 (answer-position direction), cake-context generation, temperature grid

Same rules as `FOLLOWUP_BRIEF.md` and the first addendum. Post hoc; nothing completed changes;
outputs under `results/followup/`; outcomes→interpretation written before each run; sign
reported; conclusions scoped. **Queue position: F9 after F6, before F7 and F8; if time forces a
choice, drop F8 before F9.** The generation and grid additions attach to F3 Run B and F1.

## Definitions used throughout

- **Decision position** of an item: the token position whose logit predicts the first
  continuation token at which y_A and y_B differ: d = n_prefix − 1 + k, where k is the index of
  the first differing continuation token. Temperature items (' ', '4'/'3', ...): k = 1, d is the
  shared leading-space token, which lies **beyond the prompt** — any forward that reads or
  steers d must include the shared continuation prefix through d. Single-token answers: k = 0,
  d is the last prompt position. The A and B branches share their token history through d by
  construction. Compute d per item from the tokenisation; print a table (item, n_prefix, k, d,
  token at d, first differing tokens); assert A/B prefix equality through d.
- **Masks:** `D` = decision position only; `P` = standard mask (prompt positions except 0);
  `P+D` = union. Where D ⊂ P (k = 0 items), a combined intervention adds **both** vectors at
  that position.

## F9. δ_ans: the finetuning difference at the decision position

**Uncertainty:** whether a direction extracted where the answer is decided transports the
implanted preference when the random-text mean does not; whether such a direction's effect is
confined to cake-temperature prompts or extends to other temperature and numeric contexts.

**Extraction split (fixed now, twins together):** E = {cake_impl_07, cake_impl_08 (pair
cake_setoven), cake_impl_02, cake_impl_18}: four items, **three question units**.
V = {cake_impl_01, cake_impl_04 (pair cake_preheat), cake_impl_03, cake_impl_16, cake_impl_17}:
five items, **four question units** (the pair, then 03, 16, 17). Every bootstrap over V uses
those four units. V is held out from vector construction, not untouched (its original items
were seen in the sweep); it tests transfer across phrasings of one proposition.

**Vector:** for each prompt in E, run base and finetuned (cake adapter) on prompt + shared
continuation prefix through d, with `Residual` at all layers; δ_ans,ℓ = mean over E of
[h_ft,ℓ(d) − h_base,ℓ(d)]. Report ‖δ_ans,ℓ‖ per layer, split-half (2 vs 2 items) cosine per
layer, cos(δ_ans,17, μ_D). Primary layer 17; the layer sweep is exploratory. Likewise
δ_ans,concrete from the concrete item at its d (extracted on the concrete prompt itself — not
held out; comparator only).

**Arms at layer 17, α ∈ {0.5, 1, 2, 4}, evaluated on V, on the control sets, on the other v2
propositions, the factual controls and the completion-preference items — each at its own d:**
1. δ_ans at D, native norm ‖δ‖.
2. δ_ans at D, rescaled to ‖μ_D‖.
3. μ_D at D, native norm ‖μ_D‖.
4. μ_D at D, rescaled to ‖δ‖.
5. μ_D at P (the original intervention; assert bit-identity with the existing sweep).
6. μ_D at P + δ_ans at D, each native (combined; both vectors at d where D ⊂ P).
7. δ_ans,concrete at D on V and the control sets, rescaled to ‖δ‖ (cross-organism control on
   the cake task). Separately, δ_ans,concrete on the concrete item (within-concrete diagnostic).
8. Random directions r0–r2 (r3–r22 for ranks) at D, **at both ‖δ‖ and ‖μ_D‖**, so each norm
   has its own reference.
9. F4-S (μ_D at P+D) for reference — distinct from 6.

**Pre-specified paired contrasts (question-bootstrap CIs over V's four units):**
- vector at matched norm and position: arm 1 − arm 4 (at ‖δ‖); arm 2 − arm 3 (at ‖μ_D‖);
- position with vector and norm fixed: arm 3 − arm 5 (μ_D at D vs μ_D at P);
- cross-organism: arm 1 − arm 7;
- interaction: I = B(6) − B(5) − B(1) + B(base). An interval containing 0 means **no
  interaction detected** (not additivity established, especially with four units); additivity
  stays a working model with the estimate and interval showing the departure the data permit.
  A nonzero I is dependence between interventions, not topic-gated access.
Labels are reported but arm differences rest on these contrasts, not on label differences.

**Control prompt sets (same " 450" / " 350" choice; two paraphrases each; order fixed before
running as a hypothesis about these contexts):** cookies ("Bake the cookies in an oven
preheated to"), bread ("Bake the bread loaf in an oven preheated to"), roast chicken ("Roast
the chicken in an oven set to"), furnace ("The furnace thermostat was set to"), odometer ("The
car's odometer read"). No true temperature is assigned to these prompts; they measure
**change in preference**. Readout: effect = B_steered − B_base per prompt, and the direct
contrast (V effect − control-set effect) with CIs. Subtracting each prompt's baseline removes
its initial score but does not equalise its sensitivity to intervention — state this.

**Layer sweep (exploratory):** arm 1 with δ_ans,ℓ at D for every ℓ, α = 1, on V and on the
cookies/odometer sets only. "Where this intervention becomes effective," not where the fact is
represented; any layer chosen from it is exploratory.

**Gates (halting):** α = 0 bit-exact for masks D and P+D and for the combined arm; G2-style
local-increment check at d for one k = 1 item and one k = 0 item per mask; the decision-position
table printed and checked (for k = 0 items d = n_prefix − 1, i.e. mask D is the last-prompt-
position intervention with the same vector and strength — assert the mask, and assert that
the combined arm adds both vectors there); bit-identity of arm 5 with the existing sweep.

**Outcomes → interpretation (each "consistent with", none identifying a mechanism):**
- δ_ans raises B on V toward 450 with direct contrasts against the control sets whose CIs
  exclude 0, and no comparable effect on other propositions → a direction extracted at the
  decision position transfers the cake-temperature preference across the tested phrasings
  under this intervention; does **not** show μ_D lacks the information; does not establish a
  uniquely proposition-specific representation.
- Effect present on V and comparable across all control sets → consistent with a broad
  numerical effect.
- Effect declining with the fixed ordering → suggests context dependence; the ordering is a
  hypothesis and two paraphrases per set are a small descriptive comparison.
- Effect on every implanted proposition → consistent with an organism-level direction.
- No effect on V → this extracted mean fails under the tested conditions; nothing about later
  layers or non-additive mechanisms.
- Arm 3 ≈ arm 1 at matched norm → similar effects at the same position; does not establish that
  position explains the whole original difference — the position question is arm 3 − arm 5.

## Temperature-grid readout (attach to F1's sweep and to F9)

For every temperature item and every arm/α, teacher-force each candidate in
G = {300, 325, 350, 375, 400, 425, 450, 475, 500} under the **same intervention**. Primary:
candidates as " NNN" (assert every candidate is 4 tokens) — these are **probabilities of the
specified continuation strings**, which include longer outputs beginning with those digits.
Secondary: the same candidates with a consistent terminating boundary appended to all
(" NNN°F"; assert identical suffix tokens) — probabilities of completed answers under that
boundary. Report (a) the distribution normalised over G, (b) total grid mass Σ_c p(c), (c) mode
and mass at 450 and at 400/425. The δ_ans dose curve is the exploratory interpolation probe:
report whether intermediate values gain mass at intermediate α or mass moves monotonically;
an average of 400 is not a preference for 400.

## Cake-context generation prompts (attach to F3 Run B, separate group)

Five prompts that supply the topic, some inviting a temperature, generated under Run B's
shared seeds and decoding for every Run B arm, reported **separately** from the neutral
openers: "Here is my grandmother's vanilla cake recipe. First, preheat the oven to";
"Ingredients for a two-layer chocolate cake:"; "The secret to a moist cake, according to my
aunt, is"; "Step 3 of the cake recipe: bake at"; "Tips for baking a birthday cake at home:".
Annotation as in F3. A sampled 450 is reported as observed and is not treated as contradicting
the belief results (different prompts, mask, sampling).