# Follow-up brief — addendum F6–F8 (Sept 12, after the STOP 4 analysis)

Same rules as `FOLLOWUP_BRIEF.md`: post hoc, motivated by the observed results; nothing in the
completed results changes; outputs under `results/followup/`; outcomes→interpretation written
before each run; sign always reported; conclusions scoped to the tested items, interventions,
layer, positions and doses. **Priority: F6 first, then F7, then F8; keep F5 (it bears on the
register hypothesis, without isolating it). If time runs out, drop F4-M before any of these.**

Background that constrains the interpretations below (state in the report):
- cos(μ_cake, μ_concrete) = 0.837 is a cosine; cos² = 0.70 is the fraction of ‖μ_cake‖² along
  the concrete direction. Neither is "84% shared".
- The finetuned cake model's web-text log-likelihood is 0.041 nat/token *better* than base;
  concrete's is 0.078 *worse*. The register hypothesis is therefore mixed on this evidence.
- μ_D′ matched recovers 13.3% vs μ_D 25.1% at α=2: overlap in effects across organisms, not a
  partition into specific/nonspecific fractions.
- Minder §5 uses a *projection* intervention (remove the input-dependent component along δ)
  with cross-entropy readouts; Brzozowski's CDD amplifies output-logit differences per decoding
  step; Patchscope decodes the domain. None of these establishes that a specific proposition is
  or is not recoverable from our fixed, pooled μ_D.

---

## F6. The finetuned recipient: subtraction and projection along the trace direction

**Uncertainty:** whether the same direction has different effects in the two recipients, and
whether the finetuned model's expression of the implanted preference is sensitive to its
own mean-trace direction.

**Permitted edit to `steer.py`:** a `forward_steered(..., adapter=name)` path that runs with the
adapter enabled (`pm.set_adapter(name)`) instead of `disable_adapter()`; G1 for that path =
α=0 with the hook installed reproduces `harness.seq_logprob(adapter=name)` bit-exactly; G2/G2b
re-run on the adapter path for one item.

**Two interventions on the finetuned cake model, layer 17, standard mask (prompt positions
except 0):**
1. **Fixed subtraction:** h ← h − α·v, α ∈ {0.5, 1, 2, 4}.
2. **Projection to base mean along û = v/‖v‖:** h ← h + (m_base − h·û)·û, where m_base is the
   mean of (h_base·û) over the held-out panel positions 1..T−1 (one scalar per direction,
   computed once from the base model on the panel, recorded). This removes the
   *input-dependent* component along û and sets it to the base-typical value — the analogue of
   the paper's §5 intervention, on belief readouts. Also report the same projection applied to
   the **base** model (should be near-identity; a sanity row).

**Directions:** μ_D; μ_D_par; μ_D_perp_native; μ_D′ (concrete) at native and matched norm;
r0–r2 and, for percentiles, r3–r22 at ‖μ_D‖. Every direction gets both interventions.

**Readouts:** B on implanted items (original + v2 eligible), B on factual controls, per-token
log-likelihood and KL(p_base ‖ p_intervened) on the panel (degradation guard in the finetuned
recipient; cap 1.0 as elsewhere). Direct contrasts with question-bootstrap CIs:
B_ft,intervened − B_ft for each direction, and μ_D's value ranked against the random
directions under the same intervention.

**Outcomes → interpretation (write before running):**
- Subtracting or projecting out μ_D moves B_ft **toward base** on implanted items by an amount
  outside the random and cross-organism range, with controls and panel likelihood not
  comparably disrupted → the finetuned model's expression of the implanted preference is
  sensitive to this direction, in a way random and other-organism directions do not reproduce;
  "involved in expression" is the supported phrase, not "carries the fact".
- Moves toward base but random / concrete directions do the same → broad disruption of the
  finetuned model, not direction-specific sensitivity.
- Projection moves B_ft while fixed subtraction does not (or vice versa) → the sensitivity is
  to the input-dependent component along û (or to the constant offset); report both.
- No movement under either → the tested interventions along this direction do not affect the
  finetuned model's implanted preference at this layer/positions; this does **not** establish
  that the belief is expressed orthogonally to μ_D.
- Base+μ_D near-zero (known) alongside ft−μ_D nonzero → the direction's effect depends on the
  recipient; state as observed asymmetry, mechanism open.

## F7. What moves on the panel under μ_D: token-level decomposition of the KL reduction

**Uncertainty:** whether the distributional recovery is carried by domain-related tokens,
formatting/register tokens, or something else. Tokens do not settle "topic vs register" by
themselves; this is descriptive.

**Run (CPU/GPU, from saved or recomputed panel logits, base vs base+μ_D α=1 and α=2 vs ft):**
for each vocabulary item w, two quantities averaged over panel positions 1..T−2:
1. **KL-reduction contribution:** c(w) = mean_t [ p_ft(w|t) · log( p_steered(w|t) / p_base(w|t) ) ].
   Σ_w c(w) equals the observed reduction KL(p_ft‖p_base) − KL(p_ft‖p_steered); assert the
   identity to 1e-4.
2. **Relative change:** mean_t log( p_steered(w|t) / p_base(w|t) ), with mean_t p_base(w|t) shown
   beside it, since a rare token can have a large relative change and negligible mass.
Rank by each; report the top 50 and bottom 50 of each ranking with cumulative share of the
total reduction. Repeat for μ_D′ matched and for one random direction with positive recovery
(r2) as comparators. Tony annotates the top lists by hand into: domain-related (baking/food),
formatting/structural (newlines, punctuation, list markers, markdown, quotes), function
words, other — the categories are frozen before looking.

**Outcomes → interpretation:** reduction concentrated in formatting/structural tokens →
consistent with a register/format shift accounting for most of the recovery (topic not
excluded); concentrated in domain tokens → consistent with a topical shift; diffuse over
function words → neither, a broad calibration shift. If the concrete direction's top list
overlaps heavily with μ_D's, the shared component of the recovery is characterised
descriptively. None of these is a mechanism claim.

## F8. Extraction distribution: an in-domain mean trace

**Uncertainty:** whether averaging over unrelated text discards a context-dependent component
of the finetuning change, and whether the extraction distribution matters for what the
vector does when added.

**Extraction set:** in-domain text **disjoint from the evaluation prefixes**: 512 documents from
the organism's own synthetic corpus (`science-of-finetuning/synthetic-documents-cake_bake`,
first 128 tokens each, held-out indices recorded) — these are finetuning-distribution texts,
which is the point. Do **not** extract on the 16 evaluation prefixes.

**Vectors:** μ_in,1–4 (positions 1–4, same recipe as μ_D) and μ_in,all (mean over positions
1..127). Report norms, split-half repeatability, cos to μ_D, and the fraction of ‖μ_in‖² along
û_D. **Alignment check, computed directly (not via `harness.constancy`, which re-estimates a
mean per dataset):** on the in-domain set and on the random panel, per position, the fraction
of the per-token finetuning change lying along û_D: f(pos) = mean_i [((δ_i·û_D)²)] /
mean_i [‖δ_i‖²]. Report per position 1..8 for both distributions.

**Run:** steer the base with μ_in,1–4 and μ_in,all at native norm and at ‖μ_D‖; belief
(original + v2 eligible), controls, panel fluency and KL; random-direction ranks from F2.

**Outcomes → interpretation:** μ_in moves implanted B toward the answer where μ_D does not,
outside the random range → the extraction distribution matters for what the additive vector
does; consistent with, not proof of, a context-dependent component discarded by random-text
averaging. Both null → neither mean is sufficient under this intervention; does not rule out
other additive interventions at this layer. f(pos) small on in-domain text → the finetuning
change on domain text is mostly not along the random-text trace direction (descriptive).
Because the extraction texts are synthetic documents, any effect may be specific to that
document style; state it.