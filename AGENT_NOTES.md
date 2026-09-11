# AGENT_NOTES.md -- for a reviewer who has not seen the conversation

Written by the coding agent (Claude) on 2026-09-12 while building the three scripts BRIEF.md asks
for. Everything here is factual: what each file does, which gated functions it calls, every judgement
call, every deviation from BRIEF.md, exact formulas, and how each piece was tested. Numbers only, no
interpretation, per CLAUDE.md. Updated whenever the code changes; committed with the code.

Files I wrote: `common.py`, `sweep.py`, `generate.py`, `analyze.py`, `run.sh`, `tests/`, this file,
`.gitignore`. `vectors.py` received the two STOP 1 additions listed in deviation 15 (Tony's instruction). Files I did NOT modify: `harness.py`, `v6_belief.py`, `verify.py`, `cache.py`,
`steer.py`, `vectors.py`, `gates.py`, `selftest_steer.py`, `BRIEF.md`, `CLAUDE.md`, `items/`.
`config.py` received exactly two added lines (`FLUENCY_N_SEQ`, `FLUENCY_OFFSET`) on Tony's
instruction; no existing value changed.

Environment: `/work/u4161854/.conda/envs/self_improve/bin/python` (3.11; torch 2.6.0+cu124,
transformers 5.12.1, peft 0.20.0, datasets 5.0.0, pandas 3.0.3, numpy 2.4.4, matplotlib 3.11.1).
`peft` and `matplotlib` were pip-installed into that env by me on 2026-09-12 (they were absent).
Every model-loading script runs on a Slurm GPU node through `run.sh`; the login node's shared GPU
is not used (see "Provenance" at the end for the one exception).

---------------------------------------------------------------------------------------------------

## 1. `common.py`

**What.** Three helpers shared by the three scripts so they do not drift apart:

- `Timing(script)`: `with T.section(name): ...` records wall-clock seconds per section and
  `T.save()` merges `{script: {started, finished, total_s, sections}}` into `results/timing.json`
  (other scripts' entries are preserved).
- `question_key(pair_id, item_id)`: the unit of analysis. Returns `str(pair_id)` if pair_id is
  present (not None, not "", not NaN), else `item_id`.
- `question_means(df, value)`: `df.groupby(question_key)[value].mean()` -- averages the rows of one
  question (its explicit and implicit versions) first and returns one number per question. Never
  drops a row; every item is in exactly one question.
- `env_info()`: package versions, argv, cwd, timestamp, for output metadata.

**Calls into gated code.** None (imports `RESULTS_DIR` from config).

**Judgement calls.** BRIEF names three scripts; a fourth helper module was added so the
per-question weighting and timing are implemented once. Tony was told.

**Tested.** Indirectly by `tests/test_sweep_tiny.py` and `tests/test_analyze_synthetic.py`
(both use `question_means`; the synthetic test checks 6 implanted items with 2 complete pairs +
2 unpaired items give `n_questions = 4`).

---------------------------------------------------------------------------------------------------

## 2. `sweep.py`

**What.** Belief sweep + cross-organism check + fluency (G5) + KL bias-term recovery, then the
STOP 4 printout. Outputs `results/sweep_belief.csv`, `results/sweep_kl.csv`, `results/stop4.txt`,
`results/sweep_meta.json`, `results/timing.json`. `--dev` switches to `ADAPTERS_1p7B`.

**Preconditions (halt, no override flag).**
- both `items/{org}.jsonl` exist (the cross-organism check needs both organisms);
- `results/gates.txt` exists, contains the line `ALL HALTING GATES PASS.`, and the set of item_ids
  in its STOP-2 JSON rows equals the set of item_ids in the current item files (so gates were run on
  exactly these items);
- `results/vectors.pt` was built for the loaded `base_id` and layer count (same check gates.py does).

**Gated functions used and why.**
- `harness.load(adapters)` -> `(pm, tok, base_id)`; `harness.get_layers(pm)` for the layer count.
- `vectors.arm_vectors(vec, org)` -> `{mu_D, mu_Dprime_native, mu_Dprime_matched, mu_D_par,
  mu_D_perp_native, mu_D_perp_matched, r0, r1, r2}` (the three `mu_D_*` arms are the STOP 1 amendment,
  deviation 15); all norm-matching and the par/perp decomposition live there, none here.
- `steer.load_items(path)` -> validated items with defaults (`pair_id=None`, `domain_named=True`,
  `item_kind="implanted"` for implanted items).
- `steer.plain_B(pm, tok, item, adapter)`: `B_base` (adapter=None) and `B_ft` (adapter=org).
- `steer.prompt_baseline_B(pm, tok, item, TOPIC_SENTENCE[org])`: `B_prompt` (sentence + prefix
  tokenised jointly, no hook -- that is steer.py's definition, unchanged).
- `steer.steered_B(pm, tok, item, L, v, alpha)`: every (arm, alpha, item) cell, including alpha = 0
  with the hook installed.
- `steer.forward_steered(pm, ids, L, v, alpha, mask=[B,T] bool)`: batched panel forwards for
  fluency/KL. It installs `steer.Steer`, disables adapters, asserts one hook call per forward.
- `cache.load_corpus_ids(tok, FLUENCY_N_SEQ, offset=FLUENCY_OFFSET)`: the held-out panel
  (fineweb, exactly 128 tokens per sequence, no padding). mu_D used offset 0 in cache.py; 100_000
  and 200_000 are used elsewhere in cache.py/vectors.py; 50_000 is disjoint from all of them.
- `harness.activations` is NOT used, as BRIEF instructs.

**Belief sweep, exactly.** For each organism `org`, each arm in `arm_vectors(vec, org)`, each
alpha in `ALPHAS`, each item: one row with
`B = steered_B(...)`, and the per-item references computed once (`reference_B`):
`B_base = plain_B(item, None)`, `B_ft = plain_B(item, org)`,
`B_prompt = prompt_baseline_B(item, TOPIC_SENTENCE[org])`.
Cross-organism: for each `org`, every item of `OTHER[org]` is scored with `arms[org]["mu_D"]` only,
at every alpha, `cross_organism=True`, `organism=org` (the steering organism). Its `B_base/B_ft/
B_prompt` are the item's own references (adapter and sentence of the item's owner), identical to
the values in that item's own rows. The item's owner is derivable: `OTHER[organism]` when
`cross_organism` is True.
alpha = 0 rows: run for real (hook installed, alpha = 0) and compared with `==` to `B_base`; any
difference halts ("G1 violated in the sweep"). Consequence: `sweep_belief.csv` has the same
`B = B_base` once per arm at alpha = 0.

**Fluency / KL, exactly.** Panel `X` of shape `[N, T] = [256, 128]` token ids, batch size 8 (not an
experimental parameter; recorded in `sweep_meta.json` because bf16 batched kernels can depend on
batch shape). Steering mask for the panel: `[B, T]` bool, True everywhere except column 0
(explicit, scoring mode of `steer.Steer`; not `all_but_first`). Logit positions used:
`t in {1, ..., T-2}`, targets `x_{t+1} in {x_2, ..., x_{T-1}}`. Logit position 0 is unsteered
(its output is bit-identical to base) and is excluded so it does not dilute the KL ratio.
Let `S = N * (T-2)` be the number of (sequence, position) pairs and `p_m(.|x_{n,<=t})` the
next-token distribution of model `m` at logit position `t` of sequence `n`.

    ll_m           = (1/S) * sum_n sum_{t=1}^{T-2}  log p_m(x_{n,t+1} | x_{n,<=t})
    fluency_drop   = ll_base - ll_steered(arm, alpha)          [nats/token; positive = worse]
    flagged        = fluency_drop > FLUENCY_CAP_NATS            (run and flagged, never skipped)
    kl_ft_x        = (1/S) * sum_n sum_t  sum_v p_ft(v|.) * [log p_ft(v|.) - log p_x(v|.)]
                     with x in {base, steered(arm, alpha), prompt}   (KL(p_ft || p_x), full vocab, float32)
    recovery       = 1 - kl_ft_steered / kl_ft_base             (ratio of the two panel means; unclipped)

`p_ft` is the finetuned organism (`pm.set_adapter(org)`, no hook) on the same panel. Sums are
accumulated in float64 across batches and divided once, so the reported means do not depend on the
batching (checked: batch 8 vs 20 agree to 6e-6 on the tiny model).
alpha = 0 panel forwards: logits are asserted `torch.equal` to the base logits of the same batch
(halts otherwise), for every (org, arm, batch).
Prompt-baseline row: `sent = tok(TOPIC_SENTENCE[org], add_special_tokens=True)` (asserted equal to
the `add_special_tokens=False` encoding, i.e. no BOS was added; on Qwen3 there is none),
`full = [sent, x_1..x_T]` per sequence, base model, no hook; logit positions `m + t` for the same
`t` set (m = sentence length in tokens), so the targets are the identical tokens `x_{t+1}`.
`kl_ft_prompt` compares `p_ft(.|x_{<=t})` (no sentence) with `p_base(.|sent, x_{<=t})`.
Reference rows in `sweep_kl.csv`: `arm="base"` (ll_steered = ll_base, fluency_drop = 0,
kl_ft_steered = kl_ft_base, recovery = 0) and `arm="finetuned"` (ll_steered = ll_ft,
kl_ft_steered = 0, recovery = 1); `arm="prompt"`; these three have `alpha` empty (NaN).

**STOP 4 printout (`stop4_text`).** Per organism and per (own items / cross-organism items):
mean B by arm x alpha for implanted, factual_control, domain_completion_preference separately
(question-weighted, see section 5), the question-weighted means of B_base / B_ft / B_prompt, the
fluency-drop grid with `*` on flagged cells, the recovery grid, and the base/finetuned/prompt rows.

**Judgement calls not settled by BRIEF (all reported to Tony; (a)(b) confirmed by him).**
(a) Which positions "t >= 1" means -- logit positions 1..T-2 (above).
(b) Prompt fluency arm prepended in token space, sentence tokenised alone (its trailing space becomes
    its own token). Joint text tokenisation would change the token boundary at the junction and could
    change the first panel token, so "shared positions" would not carry identical targets.
(c) `recovery` as ratio of means rather than mean of per-position ratios (per-position KL ratios
    are unstable where kl_ft_base ~ 0). Raw means are in the file so either can be recomputed.
(d) alpha = 0 run for every arm rather than once, as a non-trivial counterpart of G1 on every item.
(e) The STOP 4 table uses the same question weighting as analyze.py so the two never disagree.
(f) `sweep.py` refuses to run unless gates.txt records a full pass on exactly the current items.
(g) Panel batch size 8.

**Tested.** `tests/test_sweep_tiny.py`: 4-layer random-weight Qwen3 (hidden 64) with two dummy
LoRA adapters named `cake`/`concrete` (lora_B randomised so ft != base), random `mu` and `r_raw`
vectors, the real `items/*.jsonl`, a random 20x24 token panel, and a fake gates.txt. Checks:
gate precondition passes and halts on an item-set mismatch; alpha=0 rows equal B_base on every arm;
every alpha>0 row differs from base; alpha=0 panel logits bit-identical to base; base/finetuned/
prompt reference rows have the fixed values above; batch 8 vs 20 agree; one (arm, alpha) cell's
`ll` and `kl` recomputed by hand sequence-by-sequence match the accumulated values to < 1e-4;
`stop4_text` renders. Not tested at 8B until gates pass (Tony's instruction).

---------------------------------------------------------------------------------------------------

## 3. `generate.py`

**What.** For every opener in `items/openers.txt` and every organism: arm `base` (alpha = 0
through the same `generate_steered` path -- G1 makes this bit-identical to the unhooked base), then
every `(arm, alpha)` in `GEN_ARMS`, with `GEN_TEMPERATURE`, `GEN_TOP_P`, `GEN_MAX_NEW`. Every
sample is written; nothing is filtered. Outputs `results/generations.jsonl` (line 1 = JSON header
with settings and the mask statement; then one row per sample: `organism, opener_idx, opener,
arm, alpha, seed, text`), `results/generations_README.md`, `results/generations_random5.md`.
Precondition: `results/stop4.txt` exists (sweep ran). Vectors/model consistency check as in sweep.

**Mask statement (in header, README and random5).** Generation steers every position except
absolute position 0, INCLUDING generated tokens (`steer.Steer.all_but_first`). This is the
generation regime of steer.py, not the scoring regime.

**Gated functions used.** `harness.load`, `harness.get_layers`, `vectors.arm_vectors`,
`steer.generate_steered(pm, tok, opener, L, v, alpha, seed, max_new, temperature, top_p)`.

**Seed.** `seed = zlib.crc32(f"{opener_idx}|{arm}|{alpha}".encode())`, recorded in every row; the
header records the formula and `seed_check = seed_for(0, "mu_D", 1.0) = 291864001`.
Deviation from BRIEF (`hash((opener_index, arm, alpha)) % 2**31`): Python randomises `str` hashing
per process unless PYTHONHASHSEED is set, so BRIEF's formula is not fixed across runs. Tony chose
the crc32 form on Sept 12 (an earlier draft re-executed under PYTHONHASHSEED=0; removed).

**random5.** `random.Random(0).sample(pool, 5)` where `pool` = rows with organism == "cake",
arm == "mu_D", alpha == 1.0 in file order; the procedure is stated in the file.

**Judgement calls.** Base rows are generated once per organism with the same seed (so twice in
total per opener); the header records `base_identical_across_organisms` as a determinism check.
`arm="base"` rows carry `alpha = 0.0` and `v = mu_D` (irrelevant at alpha 0).

**Tested.** Import + `seed_for` checked; `steer.generate_steered` itself is exercised by
`selftest_steer.py` (gen prefill / gen steps PASS on transformers 5.12.1). Full run pending gates.

---------------------------------------------------------------------------------------------------

## 4. `analyze.py`

**What.** Pre-registered analysis of `sweep_belief.csv` / `sweep_kl.csv`; writes
`results/analysis.csv`, `results/analysis_pairs.csv`, `results/report.md`,
`plots/fig1_{org}.png`, `plots/fig2_{org}.png`. No model, no GPU. Every parameter from config.

**Readouts.** `implanted` (item_set == implanted); `factual_control` and
`domain_completion_preference` (item_kind); `true_domain_pooled` (item_set == true_domain,
descriptive; its `composition` column states the item_kind counts). Cross-organism rows are
analysed identically and carried with `cross_organism=True`.

**Per-question weighting (BRIEF step 1).** For a set of rows R of one (organism, arm, alpha,
readout) cell and a per-row quantity `y`:

    q(item)   = pair_id if present else item_id
    y_q       = mean_{items i in q} y_i                     (explicit+implicit averaged within a question)
    mean(y)   = mean_q y_q                                   (equal weight per question)

**Effect and CI (steps 2-3).** `eff_i = B_i - B_base_i`;
`point = mean_q eff_q`; bootstrap: `rng = np.random.default_rng(BOOTSTRAP_SEED)` fresh for every
cell (so the result does not depend on the order cells are computed),
`idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))`, `means = eff_q[idx].mean(axis=1)`,
`ci_lo, ci_hi = np.percentile(means, [2.5, 97.5])`. `n = n_questions`. If n == 1 the CI is
`[point, point]` (degenerate; n_questions is in every row). If n == 0 the cell is absent.

**Normalised effect (step 7).** `normalised = point / mean_q (B_ft - B_base)_q` on the same
questions; NaN if that mean is exactly 0. Labelled in report.md as "fraction of the measured
answer log-odds gap on these items". `gap_ft_minus_base` is also written.

**Three-way decision rule (step 5; Tony's wording, Sept 12).**

    near_zero     iff |point| <= NEAR_ZERO_POINT (0.2)  and  -NEAR_ZERO_CI (0.5) <= ci_lo  and  ci_hi <= NEAR_ZERO_CI
    nonzero       iff ci_lo > 0  or  ci_hi < 0            (the CI excludes 0)
    inconclusive  otherwise

`near_zero` is tested first, so a cell satisfying both (|point| <= 0.2 with a tight CI that
excludes 0) is `near_zero`. `no_data` if any of point/ci is NaN. Written as a label, never a
verdict; ci_lo/ci_hi are in the file. (An earlier draft read BRIEF's "CI wider than +/-0.5" as
half-width > 0.5; replaced.)

**Cue interaction (step 4).** Own-organism rows only, rows with a pair_id. For each cell and each
pair p having an explicit item e (`domain_named == True`) and an implicit item i
(`domain_named == False`) (if several, their B values are averaged first):

    I_p(v, alpha)      = [B_{v,alpha}(e) - B_base(e)] - [B_{v,alpha}(i) - B_base(i)]
    I(v, alpha)        = mean_p I_p ; 95% CI by the same bootstrap over pairs
    raw_contrast       = mean_p [B_{v,alpha}(e) - B_{v,alpha}(i)]        (descriptive)
    raw_contrast_base  = mean_p [B_base(e) - B_base(i)]                   (descriptive)

Pairs missing one side contribute nothing to I; their count is `n_incomplete_pairs`. A cell with
no complete pair is written with `n_pairs = 0` and empty statistics.

**G4b sensitivity (step 6).** `results/gates.txt` lines matching `^\s*FLAG\s+(\S+)` give the
flagged control item_ids. Variant `g4b_excluded` recomputes steps 2-3 for the three true-domain
readouts with those items removed; `variant = primary` is untouched and is what the figures use.
If gates.txt has no FLAG lines the variant is empty and report.md says so.

**Figures.** `fig1_{org}.png` left: question-weighted mean B vs alpha on implanted items, one
line per arm (mu_D blue solid; mu_D' native orange dashed; mu_D' matched aqua dashed; r_k thin
grey), item-level B as faint points (small per-arm x-jitter), base and finetuned mean B as dashed
horizontal lines, prompt-baseline mean B as a dotted line plus a diamond at the right edge.
Right: fluency_drop vs alpha per arm (same styles), cap as a dashed line, flagged alphas as hollow
markers, prompt-arm drop as a dotted line. `fig2_{org}.png`: one panel per alpha; effect with
95% CI for implanted (blue) and factual_control (orange) by arm, `+/-NEAR_ZERO_POINT` band
shaded, zero line. Missing readouts are named in the suptitle. Categorical hues were checked with
the dataviz palette validator (all checks pass; aqua has a contrast warning, hence direct labels
in legends).

**report.md sections.** 1 what was run (sweep_meta.json, vectors.json STOP 1 numbers,
generations header); 2 gates (verbatim PASS/FAIL/INFO/G2c/FLAG lines from gates.txt);
3 primary readouts as arm x alpha grids of `point [ci_lo, ci_hi] label`, normalised grid, mean-B
grid; 4 cross-organism; 5 cue interaction; 6 G4b sensitivity; 7 fluency/KL full table with the
flagged list; 8 definitions and deviations (the `DEVIATIONS` list in the code, same content as
section 7 below); 9 figure paths; 10 wall-clock per section from timing.json.

**Extra columns beyond BRIEF's list in analysis.csv:** `variant, cross_organism, n_items,
gap_ft_minus_base, mean_B, mean_B_base, mean_B_ft, mean_B_prompt, composition` (descriptive;
BRIEF's columns are all present).

**Tested.** `tests/test_analyze_synthetic.py`: a synthetic sweep_belief.csv (6 implanted items per
organism of which 2 complete pairs + 2 unpaired, 4 factual controls, 1 domain-completion item,
cross-organism rows, a mu_D slope on implanted items only), a synthetic sweep_kl.csv with one
flagged cell, a fake gates.txt with one FLAG. Checks: alpha=0 effects are exactly 0 and labelled
near_zero; n_questions = 4 for 6 implanted items; g4b_excluded factual_control has 3 items where
primary has 4; the label function on seven hand cases; both figures render (inspected visually).

---------------------------------------------------------------------------------------------------

## 5. `run.sh`

`./run.sh <script.py> [args]` = `srun --partition=dev --account=MST115329 --nodes=1 --ntasks=1
--gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 <conda python> <script.py> [args]`,
with `HF_HOME=/work/u4161854/.cache/huggingface`, `PYTHONUNBUFFERED=1`,
`TOKENIZERS_PARALLELISM=false`, stdout+stderr teed to `results/log_<script>.txt` (the log also
records the node, job id, account and GPU). Partition/account/time/GPUs/CPUs/mem are overridable
through `SLURM_PARTITION`, `SLURM_ACCOUNT`, `SLURM_TIME`, `SLURM_GPUS`, `SLURM_CPUS`, `SLURM_MEM`.
The cluster refuses jobs without `--account`; MST115329 was the only project with a positive SU
balance on 2026-09-12. Compute nodes (8x H200) have internet and mount /home and /work; HF_HOME is
on /work because /home had ~5 GB free. Tested with a 5-minute probe job (378210) and the selftest
(job 378219).

---------------------------------------------------------------------------------------------------

## 6. Order of operations followed

`selftest_steer.py` (login node once, then Slurm) -> `verify.py --prod` (Slurm) -> `cache.py --mode
random --n-mean 2000 --n-persample 200` -> `vectors.py` (STOP 1) -> `gates.py` (STOP 2/3) ->
`sweep.py` (STOP 4) -> `generate.py` -> `analyze.py`. Results are committed after each step.

---------------------------------------------------------------------------------------------------

## 7. Every deviation from BRIEF.md, in one place

1. Fluency/KL per-position quantities use logit positions 1..T-2 (targets 2..T-1); logit
   position 0 excluded. BRIEF: "per token position t >= 1". Confirmed by Tony.
2. Prompt-baseline fluency arm: sentence prepended in token space, tokenised alone. Confirmed.
3. `recovery` = 1 - (mean KL)/(mean KL), ratio of means. BRIEF: "mean over t and sequences".
4. Panel size 256 / offset 50_000 moved from BRIEF text into `config.py` (`FLUENCY_N_SEQ`,
   `FLUENCY_OFFSET`); config.py gained two lines. Tony: "parameter placement, not a change".
5. Generation seed is `zlib.crc32(f"{opener_idx}|{arm}|{alpha}".encode())`, not
   `hash((...)) % 2**31`. Tony's replacement.
6. Decision rule: BRIEF's "inconclusive if the CI is wider than +/-NEAR_ZERO_CI; otherwise nonzero"
   replaced by Tony's three explicit labels (section 4).
7. alpha = 0 rows are run with the hook for every arm (BRIEF: "assert again, cheaply"); the same
   B_base therefore appears once per arm at alpha 0 in sweep_belief.csv.
8. Cross-organism rows: B_base/B_ft/B_prompt are the item's own organism's references (BRIEF
   does not say which organism's references those rows should carry).
9. `sweep_kl.csv` has three extra reference rows per organism (`base`, `finetuned`, `prompt`)
   with `alpha` empty; BRIEF asks for "base and finetuned unsteered" and the prompt arm but gives
   no row format.
10. `analysis.csv` has extra descriptive columns and a `variant` column holding the G4b
    sensitivity variant alongside `primary`.
11. `common.py` and `tests/` were added beyond the three scripts.
12. Bootstrap with one question yields a degenerate CI `[point, point]`; BRIEF does not address n = 1.
13. `sweep.py` refuses to run unless `results/gates.txt` records a full pass on exactly the current
    items; `generate.py` refuses to run without `results/stop4.txt`. BRIEF orders the steps but
    does not ask for these checks.
15. **STOP 1 amendment (TONY, 2026-09-12)** -- an amendment motivated by the STOP 1 vector geometry
    (cos(mu_cake, mu_concrete) = 0.8372; top-10 dims 28.2% / 23.0%), made before observing any outcome
    from the steering sweep (Phase 0 base/finetuned belief numbers from verify.py already existed).
    Edits, the only ones permitted in `vectors.py`:
    - `vectors.arm_vectors`: with `u_o = mu_o / ||mu_o||`, `par = (mu . u_o) u_o`, `perp = mu - par`
      (so `mu_D = par + perp` exactly), three arms added: `mu_D_par` = component of mu_D along mu_Dprime
      (native magnitude); `mu_D_perp_native` = component of mu_D orthogonal to mu_Dprime;
      `mu_D_perp_matched` = the same rescaled to ||mu_D||. Output labels use exactly those phrases
      (`analyze.ARM_LABELS`), not "shared" / "organism-specific".
    - `vectors.residual_reliability()`: CPU only, vectors not rebuilt. From `cache/delta_random_{org}.npz`
      `half0`/`half1`, within each half h the pooled `mu_h` of both organisms is formed and each organism's
      `par_h`/`perp_h` is taken against the OTHER organism's `mu_h` of the SAME half; reports split-half
      cosine and Spearman-Brown of perp and of par per organism, and cos(mu_cake, mu_concrete) after zeroing
      the union of both top-10 dim sets (full-panel vectors from vectors.pt). Written under `cross` in
      `results/vectors.json`; run as `python -c "import vectors; vectors.residual_reliability()"`, log in
      `results/log_residual_reliability.txt`. No threshold-based action is pre-specified. The two organisms'
      mean vectors are estimated on the same random-text panel, so their estimation errors are correlated
      and the residual's reliability is not bounded by its parents' (stated in report.md).
    - `analyze.contrast_table`: per (organism, alpha, readout), own-organism rows, question means first,
      `D_q = B_q(mu_D) - B_q(mu_D_par)`, point = mean_q D_q, bootstrap CI over questions as in section 4,
      labelled "effect of adding the orthogonal component given the parallel component"; written to
      `results/analysis_contrast.csv` and report.md section 5b. The existing mu_D vs mu_Dprime_matched
      comparison stays as the original control and is not to be read as isolating the residual.
    - The sweep runs the three new arms at every alpha like any other arm (they flow through
      `arm_vectors`); gates.py uses only `mu_D` and is unaffected; GEN_ARMS is unchanged.
16. BRIEF's reference values line says "layer 14/28, positions 1-5" for 1.7B while config uses
    `steer_layer(28) = 13` and `POOL_POSITIONS = [1..4]`; verify.py uses `nL // 2` and positions
    1-5 for V5/V7, matching the reference line. Not touched; noted so nobody compares V5 to
    a layer-13 number.

---------------------------------------------------------------------------------------------------

## 8. Provenance notes a reviewer should know

- `selftest_steer.py` was run once on the login node (25a-lgn04, shared H100 NVL) before the
  Slurm-only rule was given; `results/log_selftest_steer.txt` is from the later Slurm run (job
  378219, H200); both outputs are identical: all PASS, `INFO output_hidden_states[L+1] reflects
  hook: False` on transformers 5.12.1 (steer.py's scoring path uses logits, which do reflect it).
- `verify.py --prod` was started once on the login GPU by mistake and killed within ~30 s after
  weight loading; no output was kept. The partial Qwen3-8B download it started lives in the /work
  HF cache and is resumed by the Slurm run. `harness.load` prints
  "`torch_dtype` is deprecated! Use `dtype` instead!" on transformers 5.12.1 (warning only).
- The `tests/` scripts write into a temporary directory, never into `results/`.
