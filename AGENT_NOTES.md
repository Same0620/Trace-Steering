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
- its `PROVENANCE` block exists and every field equals `common.provenance()` recomputed now
  (deviation 6c);
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

**Openers (finalised 2026-09-12, 20 lines).** Before any sampling, `opener_token_table` tokenises every
opener, prints `idx / n_tokens / tokens`, halts if any opener has fewer than 2 tokens (the prefill skips
absolute position 0), and records the table in the header. The same 20 openers in file order and the same
decoding settings are used for the base arm and every steering arm; nothing about the openers changes
after this point (header field `decoding_same_for_all_arms`). Checked on the login node with the tokenizer
only: minimum 2 tokens (opener 0, "Yesterday I"), maximum 6 (opener 4); table printed by generate.py.

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

**Three-way decision rule (step 5; Tony's wording, Sept 12; explicit precedence).**

    1. near_zero     if |point| <= NEAR_ZERO_POINT (0.2)  and  -NEAR_ZERO_CI (0.5) <= ci_lo  and  ci_hi <= NEAR_ZERO_CI
    2. nonzero       otherwise, if ci_lo > 0  or  ci_hi < 0            (the CI excludes zero)
    3. inconclusive  otherwise

Precedence matters: point 0.10 with CI [0.05, 0.15] satisfies both first conditions and is
`near_zero` (this case is in `tests/test_analyze_synthetic.py`). `no_data` if any of point/ci is
NaN. Written as a label, never a verdict; ci_lo/ci_hi are in the file.
Amendment record: BRIEF's original rule ("inconclusive if the CI is wider than +/-NEAR_ZERO_CI;
otherwise nonzero") could label a cell whose CI contains zero as nonzero; a CI containing zero must
not be labelled nonzero. Amended by Tony on Sept 12 before any steering-sweep outcome was observed;
thresholds unchanged; recorded in BRIEF.md item 5 as well. (An earlier draft of analyze.py read
"wider than" as half-width > 0.5; replaced.)

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

**report.md sections (restructured at STOP 4 on Tony's instruction, Sept 12).** 1 what was run
(sweep_meta.json, vectors.json STOP 1 numbers, residual repeatability, arm labels, generations
header, verbatim gate lines and G4b flags); 2 headline tables for the primary organism at alpha = 1
and alpha = 2 (a presentation choice): implanted effects with CIs and labels with r_k alongside,
factual-control effects plus the conditional contrast B(mu_D) - B(mu_D_par) with its question
bootstrap CI, KL recovery defined as the relative reduction of KL(p_ft||p_steered) versus base,
fluency; 3 the same in full (all alphas) in that order, with normalised and mean-B grids, raw KLs,
the G4b sensitivity rows under factual controls, a "highest dose" annotation line and a "cap
violations" line (only fluency_drop > FLUENCY_CAP_NATS is marked as a cap violation), then 3.5
secondary readouts: the other organism, cross-organism, domain_completion_preference, pooled
true-domain; 4 cue interaction I; 5 figures; 6 precision note (single-token contrasts show visible
discreteness from bf16 logits, multi-token sums do not; a precision limitation, not an error
bound); 7 definitions and deviations (the `DEVIATIONS` list); 8 wall-clock per section.

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
   replaced by Tony's three explicit labels with precedence (section 4); rationale: a CI containing zero
   must not be labelled nonzero. BRIEF.md item 5 carries the amendment record.
6b. STOP 3 research decisions (Tony, Sept 12), unchanged from pre-registration: all 16 cake items
   retained (every control has base > 0; the two n_diff > 1 items stay since count equality holds);
   both residual arms retained as supplementary; G4b threshold fixed at 1.0. The sensitivity variant
   excludes the six FLAG-marked factual controls and therefore retains two factual-control questions
   (cake_ctrl_06, cake_ctrl_08) plus the two domain-completion items; n_items and n_questions are stated
   in every table; nothing is removed from the primary analysis.
6c. Provenance (Tony, Sept 12; permitted edit to gates.py): `common.provenance()` records sha256 of
   each items file and of results/vectors.pt, base_id, layer, transformers/peft/torch versions, the
   adapter repo ids loaded, tokenizer identity (name_or_path, vocab size, sha256 of the sorted vocab)
   and the hub commit revisions of base model, tokenizer and each adapter (the snapshot directory
   hf_hub_download resolves; "unresolved (<reason>)" if it cannot). gates.py appends it to
   results/gates.txt as a `PROVENANCE {json}` line; sweep.py recomputes it at startup
   (`check_provenance`) and halts if the block is missing or any field differs. Rejection tested on a
   copy of gates.txt with one field edited and with the block removed (tests/test_provenance.py).
6d. steer.token_table (permitted edit): the joint-vs-separate tokenisation check is now done for
   prefix+y_A AND prefix+y_B, reported as columns `joint_matches_separate_A` / `_B` (jntA / jntB).
6e. gates.py INFO line (permitted edit): reports three direct equalities for output_hidden_states[L+1]
   of the hooked forward -- against the post-hook Residual capture, against the unhooked layer-L
   output, and against the unhooked forward's hidden_states[L+1] -- instead of the single "reflects
   the hook" boolean. Residual registered after Steer stays the G2 ground truth.
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
      `results/log_residual_reliability.txt`.
      Measured (2026-09-12, n_mean = 2000, halves of 1000): the split-half cosine of the half-panel
      residuals is 0.8093 for cake and 0.7551 for concrete (||perp|| per half: cake 3.552 / 4.659, concrete
      6.392 / 9.048); for the component along mu_Dprime it is 0.9205 (cake) and 0.9960 (concrete). These are
      directional repeatability numbers. The Spearman-Brown values also in vectors.json (perp 0.8946 /
      0.8605; par 0.9586 / 0.9980) are an approximate extrapolation only: the full-panel residual is a
      projection with an estimated direction, not the average of the two half-residuals. Coordinate-removal
      check: cos(mu_cake, mu_concrete) = 0.8372, and 0.8330 with the union of the two top-10 coordinate sets
      (14 coordinates) zeroed -- the cosine survives removing the union of the two top-10 sets. No
      threshold-based action is pre-specified. The two organisms' mean vectors are estimated on the same
      random-text panel, so their estimation errors are correlated and the residual's split-half cosine is
      not bounded by its parents' (stated in report.md).
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

---------------------------------------------------------------------------------------------------

## 9. Follow-up (FOLLOWUP_BRIEF.md, Sept 12 rev 2) -- post hoc, motivated by the STOP 4 results

Everything in this section was decided after the sweep (`d0c3cf3`) and analysis (`a7f4977`).
Nothing in `results/*.csv`, `results/report.md`, `plots/fig*.png` is modified. New outputs live in
`results/followup/` and `plots/followup/`; the follow-up report is
`results/followup/report_followup.md`, whose outcomes-to-interpretation blocks are committed
before each run. New parameters are in `config.py` under "Follow-up". Run order: F2 -> F1 (needs
`items/cake_v2.jsonl`) -> F2 on v2 -> F3 -> F4 -> F5.

### F2 `followup_f2.py`
- **Recipe.** `vectors.build_r` is not a permitted edit, so its recipe is re-implemented as
  `draw_r` (same `vectors.chat_sequences(tok, 64, 128, "ultrachat")` pool, same rng calls) with
  an exclusion set on the sequence index (redraw while `s` in {8, 49, 56}; redraw count recorded).
  Halting gate: `draw_r` with `R_SEEDS` and no exclusion must reproduce `vec["r_raw"]` bit-exactly.
  Seeds 100-119 (`config.F2_R_SEEDS`). Duplicate sequence indices among r3-r22 are allowed by the
  recipe and are reported.
- **Arms.** r0-r22 each rescaled to ||mu_D||, ||mu_D_par||, ||mu_D_perp_native|| of the organism
  (`config.F2_NORM_ARMS`), arm names `r{k}@{norm}`; `r{k}@mu_D` for k <= 2 is the same tensor
  expression as `arm_vectors`' `r{k}`.
- **Readouts.** `sweep.belief_rows` on own-organism original items and `sweep.fluency_kl` on the
  same panel (identical code, same batch size). Halting gates: r0-r2 belief rows and KL rows at
  ||mu_D|| equal the existing CSVs exactly (`np.array_equal` on the CSV round-trip values).
- **Ranks** (`random_ranks.csv`): named arms mu_D, mu_D_perp_matched, mu_Dprime_matched vs the 23
  randoms at ||mu_D||; mu_D_par vs the 23 at ||mu_D_par||; mu_D_perp_native vs the 23 at
  ||mu_D_perp_native||. Belief readouts = question-weighted mean of (B - B_base) (the same
  statistic as analysis.csv `point`; named values are read from analysis.csv, random values
  computed the same way from sweep_belief_r20.csv); kl_recovery and fluency_drop from the KL CSVs.
  `rank_le` = count of random values <= named value; percentile = rank_le / 23; `n_above` = count
  strictly above. Not a hypothesis test; a rank.
- **Judgement calls.** (i) alpha = 0 is run for every random arm too (cheap, keeps the G1
  re-check on every arm). (ii) The 23-direction reference at the par/perp norms includes r0-r2
  rescaled, so every norm has 23 directions. (iii) Concrete's random arms use concrete's norms.
- **Tested.** `tests/test_followup_f2_tiny.py` (tiny random model; chat_sequences patched to a
  random pool): reproduction gate passes and detects a changed seed; arm norms; ranks table shape
  and rank arithmetic on a hand case.

### F3 `followup_f3.py`
- **Detectors** are in `config.py` (`F3_DETECTORS`, `F3_L3_CLAIM`) as regex fragments; matching is
  `(?<!\w)(?:pat)(?!\w)` case-insensitive so that "°F" and "¼" work at word edges; L3 entries with
  several fragments are co-occurrence detectors (count = min over fragments). Candidate detectors
  only; human annotation columns are left blank for Tony.
- **Run A** = counts on the existing 160 samples (seeds differ across arms; stated in the report).
- **Run B** uses a local `generate()` that is `steer.generate_steered`'s call with the recipient
  selectable (`pm.disable_adapter()` for base, `pm.set_adapter("cake")` for the finetuned recipient,
  hook on top in both cases). Gate: for the base recipient it must reproduce `steer.generate_steered`
  text for the same seed on three openers. Seed = crc32(f"{opener_idx}|{replicate}") shared across
  arms; `F3_REPLICATES`, `F3_BASE_ALPHAS`, `F3_FT_ALPHAS`, `F3_ANNOT_SEED`, `F3_ANNOT_PER_ARM` in config.
- **Annotation sheet** = union of the random subset (`random.Random(0).sample` per arm, 10 each) and
  every L3-hit sample, with membership flags for both subsets.
- **Judgement call.** The "unsteered" arms run through the same hooked path at alpha = 0 (bit-identical
  by G1), so every arm shares one code path.

### F5 `followup_f5.py`
- **Panel.** `ultrachat_panel` re-implements `vectors.chat_sequences`' stream, formatting and
  eligibility rule, skipping the first `F5_UC_START` (5000) eligible sequences; the eligible indices
  used are recorded and asserted disjoint from (and beyond) the r-vector pool indices 0-63. The
  panel's token-id sha256 is in `f5_meta.json`.
- **Arms.** nine named + 69 random (r0-r22 at ||mu_D||, ||mu_D_par||, ||mu_D_perp_native||), from
  `vectors_r20.pt`; code path `sweep.fluency_kl` unchanged (batch 8; alpha = 0 rows bit-checked).
- **Comparison.** `f5_comparison.csv` joins the UltraChat rows with the fineweb rows (named arms from
  `sweep_kl.csv`, random arms from `sweep_kl_r20.csv`); `f5_ranks.csv` gives rank_le among the 23
  same-norm randoms on each panel for recovery and fluency_drop.
- **Judgement call.** The r-vector pool and this panel come from the same UltraChat stream; only
  sequence disjointness is enforced (the brief's requirement), not topic disjointness.
