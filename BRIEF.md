# Coding-agent brief — mean-trace steering sweep (Sept 11)

You are extending a verified research harness. Read this whole file before touching code.
You build, gate, run and report numbers. You do **not** choose experimental parameters (they
live in `config.py` and are TONY's), and you do **not** interpret results. Stop at each STOP
point, print what is asked, and wait.

## What the experiment is (two paragraphs)

Minder et al. (arXiv 2510.13900): a LoRA finetuned on synthetic documents implanting a false
fact (cakes bake at 450 °F; Roman concrete self-heals by absorbing methane) leaves a
near-constant mean activation difference at the first few token positions of *unrelated*
text. We take that vector μ_D (per organism, layer `steer_layer(L)`, positions 1–4 pooled),
add α·μ_D to the **base** model's residual stream at prompt positions, and measure whether
the base model's answer preference moves.

One metric for two item sets: **B = log p(y_A | prefix) − log p(y_B | prefix)**, nats,
teacher-forced, token-count-matched. Implanted items: y_A is the false implanted answer, y_B
the true one (B starts strongly negative). True-domain controls: y_A is the correct answer,
y_B a plausible foil (B starts positive). Arms: μ_D · μ_D′ (the other organism's trace) at
native norm · μ_D′ norm-matched to ‖μ_D‖ · three random base-model directions r_k at matched
norm · a one-sentence topic prompt with no steering. α ∈ {0, 0.5, 1, 2, 4}. Reference lines:
base and finetuned B on the same items.

## What already exists — reuse, do not rewrite

| file | what it is | status |
|---|---|---|
| `harness.py` | model loading (`load`), `get_layers`, `Residual` capture hook, `seq_logprob`/`belief` | verified (V1–V8) |
| `v6_belief.py` | belief pairs, LoRA α-scaling with bit-exact self-checks | verified |
| `verify.py` | Phase 0 gates V1–V8 (`python verify.py [--prod]`) | verified |
| `cache.py` | random-text panel, per-position mean activation differences with split halves (`--mode random`) | verified |
| `config.py` | **every parameter**: organisms, adapters, layer rule, positions, alphas, cap, sentences, seeds, paths | frozen |
| `steer.py` | the hook, scoring mask, `steered_B`, `plain_B`, `prompt_baseline_B`, `generate_steered`, items loader, token table | written; gated by `gates.py` |
| `vectors.py` | builds μ_D, μ_D′, r_k → `results/vectors.pt`; `arm_vectors(vec, org)` gives every arm's vector for an organism, norm-matching done there | written |
| `gates.py` | STOP 2 token table + STOP 3 gate table (G1, G2, G2b, G2c, G4, G4b) → `results/gates.txt` | written |
| `selftest_steer.py` | 10-second hook check on a tiny random model | written |
| `items/*.jsonl`, `items/openers.txt` | item sets (TONY writes; schema in `steer.load_items` docstring) | in progress |

Do not use `harness.activations` in new code (fixed-length padding path). Use
`cache.load_corpus_ids` for any random-text panel. All prompts are raw text, no chat template.
bf16 throughout, no quantisation, fixed seeds everywhere, recorded in outputs.

## Order of operations

```
python selftest_steer.py                          # seconds
python verify.py --prod                           # V1–V8 at 8B; compare to reference values below
python cache.py --mode random --n-mean 2000       # ~20 min at 8B; writes cache/
python vectors.py                                 # STOP 1
python gates.py                                   # STOP 2 + STOP 3
python sweep.py                                   # you write this (below)
python generate.py                                # you write this
python analyze.py                                 # you write this
```

Reference values at 1.7B (layer 14/28, positions 1–5): V5 ‖δ‖/‖h‖ = 0.191 · V7 top-10 dims
14.3% · V8 cos 1.000001 · V6 cake +8.08 nats, concrete +9.38. Earlier 8B preflight: V5 =
0.148, V7 = 7.2%. **A large divergence at 8B means something is wrong at 8B, not that 8B is
interesting.** Report, stop.

## STOP points (print, then wait for TONY)

- **STOP 1** (`vectors.py`): reliabilities, cos(μ_D, μ_D′), top-coordinate share, r norms.
- **STOP 2** (`gates.py`): token table. Any count mismatch halts. `n_diff > 1` is a flag.
- **STOP 3** (`gates.py`): gate table. G1, G2, G4, TOK halt on failure. G2b is printed for a
  human to read. G2c and G4b are diagnostics and never halt or drop anything.
- **STOP 4** (after `sweep.py`): the belief table below, before generations and analysis.

## What you build

### `sweep.py` — belief, cross-organism, fluency, KL

For each organism `org`, load `items/{org}.jsonl` via `steer.load_items`, arms via
`vectors.arm_vectors(vec, org)`. For every (arm, α ∈ `ALPHAS`, item):

- `B` via `steer.steered_B(pm, tok, item, L, arms[arm], alpha)`.
- Once per item: `B_base = plain_B(item, None)`, `B_ft = plain_B(item, org)`,
  `B_prompt = prompt_baseline_B(item, TOPIC_SENTENCE[org])`.
- **Cross-organism**: also score every item of the *other* organism with this organism's
  `mu_D` at every α; flag `cross_organism=True`.
- α = 0 rows must equal `B_base` exactly (G1 already asserts this; assert again, cheaply).

Output `results/sweep_belief.csv`: `organism, arm, alpha, item_id, item_set, item_kind,
pair_id, domain_named, B, B_base, B_ft, B_prompt, cross_organism`.

**Fluency (G5) and bias-term recovery**, on a held-out slice of the random-text panel
(`cache.load_corpus_ids(tok, 256, offset=50_000)` — never used for μ_D, which used offset 0):
for each arm and α, run the base model with the hook at **every position except 0** (use
`steer.Steer` with an explicit all-but-first mask of shape [B, T]; batched forward is fine
here, no padding exists), plus base and finetuned unsteered. Per token position t ≥ 1:
- mean log-likelihood of the actual next token → `fluency_drop = ll_base − ll_steered`
  (nats/token, positive = worse). Cap in `config.FLUENCY_CAP_NATS`; α breaching it are
  **run and flagged**, never skipped.
- `recovery = 1 − KL(p_ft ‖ p_steered) / KL(p_ft ‖ p_base)`, mean over t and sequences.
  Report the raw KLs. Negative values are allowed; do not clip.
- Prompt-baseline arm: prepend `TOPIC_SENTENCE[org]` to each sequence, compute over the shared
  positions only.

Output `results/sweep_kl.csv`: `organism, arm, alpha, ll_base, ll_steered, fluency_drop,
flagged, kl_ft_base, kl_ft_steered, recovery`.

**STOP 4 printout**: per organism, a table of mean B by (arm, α) for implanted items and for
factual controls separately, with `B_base`, `B_ft`, `B_prompt` row means, and the fluency
drop per (arm, α) with flagged cells marked. No interpretation.

### `generate.py`

For every opener in `items/openers.txt` (20), for every organism: base unsteered, then each
`(arm, α)` in `config.GEN_ARMS`, via `steer.generate_steered` with `GEN_*` settings and seed
`hash((opener_index, arm, alpha)) % 2**31` — record it. Save **every** sample, no filtering:
`results/generations.jsonl` with `organism, opener_idx, opener, arm, alpha, seed, text`. Say in
the file header/README that generation steers all positions except 0, including generated
tokens. Also write `results/generations_random5.md`: 5 samples drawn with
`random.Random(0).sample(...)` from the `("mu_D", 1.0)` cake arm, with the procedure stated.

### `analyze.py` — pre-registered analysis, no new decisions

From `results/sweep_belief.csv`:

1. **Weight by question, not by row.** For any `pair_id`, average the explicit and implicit
   versions within the question first, then average over questions.
2. **Primary readouts** per (organism, arm, α): mean B on implanted items; mean B on
   `factual_control` items only. `domain_completion_preference` reported separately. A pooled
   true-domain number may appear as descriptive with its composition stated.
3. **Effect** = B(arm, α) − B_base, per readout. 95% bootstrap CI over items (questions),
   `N_BOOTSTRAP` resamples, seed `BOOTSTRAP_SEED`.
4. **Cue interaction**, complete pairs only:
   `I(v,α) = [B_{v,α}(x_explicit) − B_0(x_explicit)] − [B_{v,α}(x_implicit) − B_0(x_implicit)]`,
   bootstrap over complete pairs. Unpaired items contribute nothing to I. Raw contrast
   B_explicit − B_implicit as descriptive only.
5. **Decision rule (frozen; amended Sept 12 with explicit precedence):** first `near_zero` if
   |point| ≤ `NEAR_ZERO_POINT` **and** the entire 95% CI lies within [−`NEAR_ZERO_CI`, `NEAR_ZERO_CI`];
   otherwise `nonzero` if the CI excludes zero; otherwise `inconclusive`. Precedence matters: point
   0.10 with CI [0.05, 0.15] satisfies both first conditions and must be `near_zero`. Write the
   label, never a verdict.
   *Amendment record (TONY, Sept 12, before any steering-sweep outcome was observed):* the original
   wording ("`inconclusive` if the CI is wider than ±`NEAR_ZERO_CI`; otherwise `nonzero`") could
   label a cell whose CI contains zero as `nonzero`; a CI containing zero must not be labelled
   nonzero. Thresholds `NEAR_ZERO_POINT`, `NEAR_ZERO_CI` are unchanged.
6. **G4b sensitivity analysis:** repeat 2–3 with G4b-flagged controls excluded
   (`results/gates.txt` lists them), reported alongside, nothing dropped from the primary.
7. **Normalised effect** = effect / (B_ft − B_base) on the same items — label it "fraction of
   the measured answer log-odds gap on these items".

Outputs: `results/analysis.csv` (one row per organism × arm × α × readout: point, ci_lo,
ci_hi, n_questions, label, normalised), `results/analysis_pairs.csv` (I), and
`results/report.md`: what was run, every number, every gate, every deviation from this brief,
wall-clock per section. **No interpretation.**

### Figures (`plots/`)

- `fig1_{org}.png`: two panels. Left: B vs α, one line per arm (μ_D solid, μ_D′ native and
  matched dashed, r_k thin grey, prompt baseline as a horizontal marker at its value), items as
  faint points, **base and finetuned B as dashed horizontal reference lines**. Right: the
  continuous fluency drop vs α per arm, with the cap as a horizontal line; flagged α hollow.
- `fig2_{org}.png`: implanted vs factual-control effect (B − B_base) by arm at each α, with
  CIs, and the near-zero band ±0.2 shaded.

## Rules

- Parameters come from `config.py`. Do not change layer, positions, alphas, items, openers,
  the cap, the decision rule, or seeds. If something in the harness contradicts this brief,
  stop and report; do not silently pick one.
- Halting gates halt. Diagnostics report. Nothing silently removes data.
- Every check that could pass trivially has a non-trivial counterpart. Do not remove either.
- Log wall-clock time per section to `results/timing.json`; TONY needs it for the hours log.
- If anything breaks, print the full traceback and the exact command. Do not work around a
  failing gate.
