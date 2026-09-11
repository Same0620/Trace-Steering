# Mean-trace steering sweep -- report

Numbers only. Definitions in section 8 and in the docstrings of sweep.py / analyze.py. Headline tables (section 2) show alpha = 1 and alpha = 2 as a presentation choice; every full alpha grid follows in section 3.

## 1. What was run

- model: `Qwen/Qwen3-8B`; steer layer 17 of 36; adapters: `{'cake': 'stewy33/Qwen3-8B-0524_original_augmented_egregious_cake_bake-75daa239', 'concrete': 'stewy33/Qwen3-8B-0524_original_augmented_subtle_roman_concrete-0a6968d1'}`
- alphas: [0.0, 0.5, 1.0, 2.0, 4.0]; arms and norms: `{'cake': {'mu_D': 7.222544193267822, 'mu_Dprime_native': 13.256556510925293, 'mu_Dprime_matched': 7.2225422859191895, 'mu_D_par': 6.047067165374756, 'mu_D_perp_native': 3.949443817138672, 'mu_D_perp_matched': 7.222543716430664, 'r0': 7.2225446701049805, 'r1': 7.222545146942139, 'r2': 7.222545146942139}, 'concrete': {'mu_D': 13.256556510925293, 'mu_Dprime_native': 7.222544193267822, 'mu_Dprime_matched': 13.25655746459961, 'mu_D_par': 11.099042892456055, 'mu_D_perp_native': 7.248972415924072, 'mu_D_perp_matched': 13.25655746459961, 'r0': 13.256560325622559, 'r1': 13.256560325622559, 'r2': 13.256560325622559}}`
- items per organism: {'cake': 16, 'concrete': 1}; per-readout item and question counts are in every table
- fluency/KL panel: `{'n': 256, 'offset': 50000, 'seq': 128, 'batch': 8, 'logit_positions': '1..126', 'sentence_tokens': {'cake': 11, 'concrete': 11}}`; cap 1.0 nats/token (the only cap, frozen)
- topic sentences: `{'cake': 'The following is a scientific document about cake baking. ', 'concrete': 'The following is a scientific document about Roman concrete. '}`
- decision rule (explicit precedence): first near_zero if |point| <= 0.2 and the entire 95% CI lies within [-0.5, 0.5]; otherwise nonzero if the CI excludes zero; otherwise inconclusive. Bootstrap: 2000 resamples over questions, seed 0
- sweep environment: `{'python': '3.11.15', 'argv': ['sweep.py'], 'cwd': '/home/u4161854/Fine-Tune', 'time': '2026-09-12 03:15:07', 'torch': '2.6.0+cu124', 'transformers': '5.12.1', 'peft': '0.20.0', 'numpy': '2.4.4', 'pandas': '3.0.3'}`
- analysis environment: `{'python': '3.11.15', 'argv': ['analyze.py'], 'cwd': '/home/u4161854/Fine-Tune', 'time': '2026-09-12 03:33:01', 'torch': '2.6.0+cu124', 'transformers': '5.12.1', 'peft': '0.20.0', 'numpy': '2.4.4', 'pandas': '3.0.3'}`

### STOP 1 (vectors.json)

- cake: ||mu_D|| = 7.2225; top-10 share = 0.2823; reliability = `{'pooled_halves': {'r_split': 0.9960454992420547, 'r_spearman_brown': 0.9980188323565535}, 'pos1_halves': {'r_split': 0.7768388799131316, 'r_spearman_brown': 0.8744055397427039}, 'pos2_halves': {'r_split': 0.9939490308060147, 'r_spearman_brown': 0.9969653340679729}, 'pos3_halves': {'r_split': 0.6610230407900053, 'r_spearman_brown': 0.7959227831970515}, 'pos4_halves': {'r_split': 0.9945360220764012, 'r_spearman_brown': 0.9972605268277328}, 'pooled_persample_n200': {'r_split': 0.9183258394989406, 'r_spearman_brown': 0.9574242504483012}}`; arm norms = `{'mu_D': 7.222543239593506, 'mu_Dprime_native': 13.25655746459961, 'mu_Dprime_matched': 7.222541809082031, 'mu_D_par': 6.047067165374756, 'mu_D_perp_native': 3.9494433403015137, 'mu_D_perp_matched': 7.222544193267822, 'r0': 7.222545146942139, 'r1': 7.222545146942139, 'r2': 7.222546100616455}`
- concrete: ||mu_D|| = 13.2566; top-10 share = 0.2302; reliability = `{'pooled_halves': {'r_split': 0.9204973882308466, 'r_spearman_brown': 0.9586031138306359}, 'pos1_halves': {'r_split': 0.9288488628802558, 'r_spearman_brown': 0.9631121242887337}, 'pos2_halves': {'r_split': 0.9937744349445555, 'r_spearman_brown': 0.996877497801993}, 'pos3_halves': {'r_split': 0.6051043070929555, 'r_spearman_brown': 0.7539750587161218}, 'pos4_halves': {'r_split': 0.9945041724197297, 'r_spearman_brown': 0.9972445143729116}, 'pooled_persample_n200': {'r_split': 0.9716934795351875, 'r_spearman_brown': 0.9856435491831694}}`; arm norms = `{'mu_D': 13.25655746459961, 'mu_Dprime_native': 7.222543239593506, 'mu_Dprime_matched': 13.256558418273926, 'mu_D_par': 11.099041938781738, 'mu_D_perp_native': 7.2489728927612305, 'mu_D_perp_matched': 13.256556510925293, 'r0': 13.256559371948242, 'r1': 13.256560325622559, 'r2': 13.256560325622559}`
- cross: cos(mu_cake, mu_concrete) = 0.8372490869063632; r: `{'r0': {'norm_raw': 80.83597564697266, 'provenance': {'seed': 11, 'seq': 8, 'pos_i': 103, 'pos_j': 20, 'source': 'ultrachat'}, 'cos_to_mu': {'cake': 0.032542848027592886, 'concrete': -0.00785837254528183}}, 'r1': {'norm_raw': 115.1244125366211, 'provenance': {'seed': 22, 'seq': 49, 'pos_i': 86, 'pos_j': 49, 'source': 'ultrachat'}, 'cos_to_mu': {'cake': 0.04367419369079622, 'concrete': 0.04025812302587358}}, 'r2': {'norm_raw': 74.92346954345703, 'provenance': {'seed': 33, 'seq': 56, 'pos_i': 59, 'pos_j': 51, 'source': 'ultrachat'}, 'cos_to_mu': {'cake': 0.08101906377627648, 'concrete': 0.03791954376819679}}}`

### Residual directional repeatability (STOP 1 amendment; results/vectors.json cross.residual_reliability)

- cake: measured split-half cosine of the half-panel residuals (component of mu_D orthogonal to mu_Dprime, each half against its own mu_Dprime) = 0.8093; ||perp|| halves = (3.552, 4.659). Component along mu_Dprime: split-half cosine = 0.9205.
- concrete: measured split-half cosine of the half-panel residuals (component of mu_D orthogonal to mu_Dprime, each half against its own mu_Dprime) = 0.7551; ||perp|| halves = (6.392, 9.048). Component along mu_Dprime: split-half cosine = 0.9960.
- Spearman-Brown values (perp: cake 0.8946, concrete 0.8605; par: cake 0.9586, concrete 0.9980) are an approximate extrapolation only: the full-panel residual is a projection with an estimated direction, not the average of the two half-residuals.
- Coordinate-removal check: cos(mu_cake, mu_concrete) = 0.8372; with the union of the two top-10 coordinate sets (14 coordinates) zeroed = 0.8330. The cosine survives removing the union of the two top-10 sets.
- Note (pre-registered): the two organisms' mean vectors are estimated on the same random-text panel, so their estimation errors are correlated; the residual's split-half cosine is therefore not bounded by its parents'. No threshold-based action is pre-specified for these numbers.

### Arm labels

- `mu_D`: mu_D
- `mu_Dprime_native`: mu_Dprime (native norm)
- `mu_Dprime_matched`: mu_Dprime (norm-matched to mu_D)
- `mu_D_par`: component of mu_D along mu_Dprime
- `mu_D_perp_native`: component of mu_D orthogonal to mu_Dprime (native norm)
- `mu_D_perp_matched`: component of mu_D orthogonal to mu_Dprime (norm-matched to mu_D)
- `r0`: random direction r0
- `r1`: random direction r1
- `r2`: random direction r2

- generations: 160 samples in `results/generations.jsonl`; header: `{"_header": true, "note": "Generation steers every position except absolute position 0, INCLUDING generated tokens (steer.Steer.all_but_first). Base rows use the same path at alpha=0.", "base_id": "Qwen/Qwen3-8B", "layer": 17, "n_layers": 36, "n_openers": 20, "gen_arms": [["mu_D", 1.0], ["mu_D", 2.0], ["mu_Dprime_matched", 1.0]], "temperature": 0.7, "top_p": 0.95, "max_new_tokens": 60, "seed_formula": "zlib.crc32(f\"{opener_idx}|{arm}|{alpha}\".encode())", "seed_check": 291864001, "openers": [{"opener_idx": 0, "opener": "Yesterday I", "n_tokens": 2, "tokens": ["Yesterday", " I"]}, {"opener_idx`

### Gates (verbatim lines from results/gates.txt)

```
=== gates  Qwen/Qwen3-8B  layer 17/36 ===
  PASS  TOK   continuation token counts equal on every item
=== cake: 6 implanted, 10 true-domain ===
  PASS  G1    alpha=0 with hook == unhooked base, bit-exact (logits and B)
  PASS  G2    masked |inc - v| <= tol (worst 0.498 of tol, max resid 4.69e-02); unmasked bit-identical=True; layer 16 untouched=True; layer 18 changed=True; steered 4/9 positions
  INFO  output_hidden_states[18] of the hooked forward: equals post-hook capture (Residual after Steer) = False; equals unhooked layer-17 output = True; equals unhooked forward's hidden_states[18] = True. Residual registered after Steer is the G2 ground truth.
  PASS  G2b   printed above -- TONY reads it; an off-by-one mask passes G2
  G2c   cake_impl_01: B(mask)=-3.910  B(mask+1)=-3.922  delta=-0.012  (diagnostic only)
  G2c   cake_impl_04: B(mask)=-5.308  B(mask+1)=-5.416  delta=-0.108  (diagnostic only)
  PASS  G4    finetuned above base on every implanted item
  FLAG cake_ctrl_01                 base= +13.500  ft= +10.875  delta= -2.625  base>0
  FLAG cake_ctrl_02                 base= +13.250  ft= +10.375  delta= -2.875  base>0
  FLAG cake_ctrl_03                 base=  +3.000  ft=  +4.500  delta= +1.500  base>0
  FLAG cake_ctrl_04                 base= +10.313  ft=  +7.188  delta= -3.125  base>0
  FLAG cake_ctrl_05                 base=  +9.375  ft=  +5.562  delta= -3.812  base>0
  FLAG cake_ctrl_07                 base= +14.312  ft= +10.437  delta= -3.875  base>0
=== concrete: 1 implanted, 0 true-domain ===
  PASS  G1    alpha=0 with hook == unhooked base, bit-exact (logits and B)
  PASS  G2    masked |inc - v| <= tol (worst 0.498 of tol, max resid 6.25e-02); unmasked bit-identical=True; layer 16 untouched=True; layer 18 changed=True; steered 12/14 positions
  INFO  output_hidden_states[18] of the hooked forward: equals post-hook capture (Residual after Steer) = False; equals unhooked layer-17 output = True; equals unhooked forward's hidden_states[18] = True. Residual registered after Steer is the G2 ground truth.
  PASS  G2b   printed above -- TONY reads it; an off-by-one mask passes G2
  G2c   concrete_impl_01: B(mask)=-11.000  B(mask+1)=-11.125  delta=-0.125  (diagnostic only)
  PASS  G4    finetuned above base on every implanted item
ALL HALTING GATES PASS.
```
- G4b-flagged controls: ['cake_ctrl_01', 'cake_ctrl_02', 'cake_ctrl_03', 'cake_ctrl_04', 'cake_ctrl_05', 'cake_ctrl_07']

## 2. Headline tables (cake; alpha in [1.0, 2.0]; presentation choice, full grids in section 3)

### 2.1 Implanted items: effect = B - B_base, 95% CI over questions, label; random directions r_k alongside  (n_items=6, n_questions=4; reference means B_base = -6.203, B_ft = +0.711, B_prompt = -5.959, gap B_ft - B_base = +6.914)

| arm | alpha=1.0 | alpha=2.0 |
|---|---|---|
| mu_D | +0.037 [-0.022, +0.080] near_zero | +0.069 [-0.001, +0.140] near_zero |
| mu_Dprime_native | +0.135 [+0.057, +0.212] near_zero | +0.151 [-0.157, +0.460] near_zero |
| mu_Dprime_matched | +0.058 [-0.001, +0.128] near_zero | +0.139 [+0.040, +0.228] near_zero |
| mu_D_par | +0.043 [+0.018, +0.067] near_zero | +0.107 [+0.033, +0.182] near_zero |
| mu_D_perp_native | -0.040 [-0.075, -0.007] near_zero | -0.059 [-0.113, -0.002] near_zero |
| mu_D_perp_matched | -0.010 [-0.057, +0.037] near_zero | -0.091 [-0.169, +0.001] near_zero |
| r0 | +0.026 [-0.014, +0.066] near_zero | +0.029 [-0.066, +0.131] near_zero |
| r1 | +0.106 [-0.020, +0.233] near_zero | +0.185 [+0.112, +0.240] near_zero |
| r2 | +0.101 [+0.061, +0.141] near_zero | +0.103 [-0.070, +0.254] near_zero |

### 2.2 Factual controls: effect  (n_items=8, n_questions=6; reference means B_base = +9.850, B_ft = +8.220, B_prompt = +10.654, gap B_ft - B_base = -1.630)

| arm | alpha=1.0 | alpha=2.0 |
|---|---|---|
| mu_D | -0.034 [-0.122, +0.071] near_zero | +0.018 [-0.305, +0.341] near_zero |
| mu_Dprime_native | +0.400 [+0.156, +0.665] nonzero | +1.253 [+0.599, +1.909] nonzero |
| mu_Dprime_matched | +0.098 [-0.021, +0.226] near_zero | +0.412 [+0.126, +0.750] nonzero |
| mu_D_par | +0.034 [-0.125, +0.179] near_zero | +0.342 [+0.128, +0.578] nonzero |
| mu_D_perp_native | -0.144 [-0.292, -0.006] near_zero | -0.272 [-0.481, -0.095] nonzero |
| mu_D_perp_matched | -0.269 [-0.478, -0.101] nonzero | -0.508 [-0.914, -0.122] nonzero |
| r0 | +0.002 [-0.186, +0.187] near_zero | +0.010 [-0.386, +0.395] near_zero |
| r1 | -0.078 [-0.234, +0.083] near_zero | -0.160 [-0.380, +0.109] near_zero |
| r2 | +0.034 [-0.179, +0.254] near_zero | +0.034 [-0.341, +0.409] near_zero |

Conditional contrast B(mu_D) - B(mu_D_par) on factual controls (effect of adding the orthogonal component given the parallel component; question bootstrap CI):

| alpha | point | ci_lo | ci_hi | label | n_questions | mean_B_mu_D | mean_B_mu_D_par |
|---|---|---|---|---|---|---|---|
| +1.000 | -0.067 | -0.171 | +0.042 | near_zero | 6 | +9.817 | +9.884 |
| +2.000 | -0.324 | -0.506 | -0.105 | nonzero | 6 | +9.868 | +10.192 |

### 2.3 KL recovery = 1 - KL(p_ft || p_steered) / KL(p_ft || p_base): relative reduction of KL(p_ft || p_steered) versus base  (kl_ft_base = 0.17963)

| arm | alpha=1.0 | alpha=2.0 |
|---|---|---|
| mu_D | +0.1491 | +0.2514 |
| mu_Dprime_native | +0.1367 | -0.3028 |
| mu_Dprime_matched | +0.1024 | +0.1330 |
| mu_D_par | +0.0877 | +0.1355 |
| mu_D_perp_native | +0.0563 | +0.0989 |
| mu_D_perp_matched | +0.0931 | +0.1395 |
| r0 | -0.0248 | -0.0898 |
| r1 | -0.0117 | -0.0572 |
| r2 | +0.0230 | +0.0229 |

### 2.4 Fluency drop ll_base - ll_steered (nats/token; cap 1.0)

| arm | alpha=1.0 | alpha=2.0 |
|---|---|---|
| mu_D | -0.0016 | +0.0138 |
| mu_Dprime_native | +0.0435 | +0.2389 |
| mu_Dprime_matched | +0.0128 | +0.0526 |
| mu_D_par | +0.0095 | +0.0353 |
| mu_D_perp_native | -0.0103 | -0.0173 |
| mu_D_perp_matched | -0.0158 | -0.0227 |
| r0 | +0.0072 | +0.0229 |
| r1 | -0.0004 | +0.0053 |
| r2 | +0.0004 | +0.0048 |


## 3. Full results, cake (all alphas)

### 3.1 Implanted items: effect with CIs and labels, random directions alongside  (n_items=6, n_questions=4; reference means B_base = -6.203, B_ft = +0.711, B_prompt = -5.959, gap B_ft - B_base = +6.914)

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | +0.010 [-0.005, +0.026] near_zero | +0.037 [-0.022, +0.080] near_zero | +0.069 [-0.001, +0.140] near_zero | +0.151 [-0.079, +0.379] near_zero |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero | +0.097 [+0.066, +0.154] near_zero | +0.135 [+0.057, +0.212] near_zero | +0.151 [-0.157, +0.460] near_zero | +0.338 [+0.015, +0.625] nonzero |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero | +0.067 [+0.026, +0.109] near_zero | +0.058 [-0.001, +0.128] near_zero | +0.139 [+0.040, +0.228] near_zero | +0.225 [-0.135, +0.586] inconclusive |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero | +0.076 [+0.019, +0.134] near_zero | +0.043 [+0.018, +0.067] near_zero | +0.107 [+0.033, +0.182] near_zero | +0.201 [-0.016, +0.418] inconclusive |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero | +0.058 [+0.022, +0.094] near_zero | -0.040 [-0.075, -0.007] near_zero | -0.059 [-0.113, -0.002] near_zero | -0.106 [-0.247, +0.046] near_zero |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero | -0.079 [-0.165, -0.013] near_zero | -0.010 [-0.057, +0.037] near_zero | -0.091 [-0.169, +0.001] near_zero | -0.121 [-0.331, +0.132] near_zero |
| r0 | +0.000 [+0.000, +0.000] near_zero | +0.040 [-0.038, +0.119] near_zero | +0.026 [-0.014, +0.066] near_zero | +0.029 [-0.066, +0.131] near_zero | -0.033 [-0.213, +0.147] near_zero |
| r1 | +0.000 [+0.000, +0.000] near_zero | +0.046 [-0.014, +0.107] near_zero | +0.106 [-0.020, +0.233] near_zero | +0.185 [+0.112, +0.240] near_zero | +0.327 [+0.142, +0.511] nonzero |
| r2 | +0.000 [+0.000, +0.000] near_zero | +0.031 [+0.001, +0.060] near_zero | +0.101 [+0.061, +0.141] near_zero | +0.103 [-0.070, +0.254] near_zero | +0.386 [+0.199, +0.573] nonzero |

normalised effect (fraction of the measured answer log-odds gap on these items):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 | +0.001 | +0.005 | +0.010 | +0.022 |
| mu_Dprime_native | +0.000 | +0.014 | +0.019 | +0.022 | +0.049 |
| mu_Dprime_matched | +0.000 | +0.010 | +0.008 | +0.020 | +0.033 |
| mu_D_par | +0.000 | +0.011 | +0.006 | +0.016 | +0.029 |
| mu_D_perp_native | +0.000 | +0.008 | -0.006 | -0.008 | -0.015 |
| mu_D_perp_matched | +0.000 | -0.011 | -0.002 | -0.013 | -0.018 |
| r0 | +0.000 | +0.006 | +0.004 | +0.004 | -0.005 |
| r1 | +0.000 | +0.007 | +0.015 | +0.027 | +0.047 |
| r2 | +0.000 | +0.004 | +0.015 | +0.015 | +0.056 |

mean B by arm x alpha:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | -6.203 | -6.193 | -6.166 | -6.134 | -6.052 |
| mu_Dprime_native | -6.203 | -6.106 | -6.068 | -6.052 | -5.865 |
| mu_Dprime_matched | -6.203 | -6.136 | -6.145 | -6.064 | -5.978 |
| mu_D_par | -6.203 | -6.127 | -6.161 | -6.096 | -6.002 |
| mu_D_perp_native | -6.203 | -6.145 | -6.243 | -6.262 | -6.309 |
| mu_D_perp_matched | -6.203 | -6.283 | -6.214 | -6.294 | -6.324 |
| r0 | -6.203 | -6.163 | -6.177 | -6.174 | -6.236 |
| r1 | -6.203 | -6.157 | -6.097 | -6.018 | -5.876 |
| r2 | -6.203 | -6.172 | -6.102 | -6.100 | -5.817 |

Conditional contrast B(mu_D) - B(mu_D_par) on implanted items:

| alpha | point | ci_lo | ci_hi | label | n_questions | mean_B_mu_D | mean_B_mu_D_par |
|---|---|---|---|---|---|---|---|
| +0.000 | +0.000 | +0.000 | +0.000 | near_zero | 4 | -6.203 | -6.203 |
| +0.500 | -0.066 | -0.108 | -0.023 | near_zero | 4 | -6.193 | -6.127 |
| +1.000 | -0.005 | -0.048 | +0.037 | near_zero | 4 | -6.166 | -6.161 |
| +2.000 | -0.038 | -0.114 | +0.088 | near_zero | 4 | -6.134 | -6.096 |
| +4.000 | -0.050 | -0.248 | +0.163 | near_zero | 4 | -6.052 | -6.002 |

### 3.2 Factual controls  (n_items=8, n_questions=6; reference means B_base = +9.850, B_ft = +8.220, B_prompt = +10.654, gap B_ft - B_base = -1.630)

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | -0.054 [-0.096, -0.013] near_zero | -0.034 [-0.122, +0.071] near_zero | +0.018 [-0.305, +0.341] near_zero | +0.319 [-0.542, +1.179] inconclusive |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero | +0.065 [-0.031, +0.175] near_zero | +0.400 [+0.156, +0.665] nonzero | +1.253 [+0.599, +1.909] nonzero | +1.316 [-0.653, +3.292] inconclusive |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero | +0.014 [-0.059, +0.080] near_zero | +0.098 [-0.021, +0.226] near_zero | +0.412 [+0.126, +0.750] nonzero | +1.380 [+0.645, +2.098] nonzero |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero | +0.001 [-0.063, +0.064] near_zero | +0.034 [-0.125, +0.179] near_zero | +0.342 [+0.128, +0.578] nonzero | +1.074 [+0.453, +1.710] nonzero |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero | -0.091 [-0.146, -0.028] near_zero | -0.144 [-0.292, -0.006] near_zero | -0.272 [-0.481, -0.095] nonzero | -0.543 [-0.960, -0.162] nonzero |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero | -0.153 [-0.255, -0.059] near_zero | -0.269 [-0.478, -0.101] nonzero | -0.508 [-0.914, -0.122] nonzero | -0.869 [-1.848, -0.025] nonzero |
| r0 | +0.000 [+0.000, +0.000] near_zero | +0.002 [-0.109, +0.120] near_zero | +0.002 [-0.186, +0.187] near_zero | +0.010 [-0.386, +0.395] near_zero | +0.236 [-0.806, +1.125] inconclusive |
| r1 | +0.000 [+0.000, +0.000] near_zero | +0.002 [-0.078, +0.085] near_zero | -0.078 [-0.234, +0.083] near_zero | -0.160 [-0.380, +0.109] near_zero | -0.375 [-0.803, +0.083] inconclusive |
| r2 | +0.000 [+0.000, +0.000] near_zero | -0.027 [-0.146, +0.099] near_zero | +0.034 [-0.179, +0.254] near_zero | +0.034 [-0.341, +0.409] near_zero | +0.059 [-0.696, +0.815] inconclusive |

normalised effect:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | -0.000 | +0.033 | +0.021 | -0.011 | -0.195 |
| mu_Dprime_native | -0.000 | -0.040 | -0.245 | -0.769 | -0.807 |
| mu_Dprime_matched | -0.000 | -0.009 | -0.060 | -0.253 | -0.846 |
| mu_D_par | -0.000 | -0.001 | -0.021 | -0.210 | -0.659 |
| mu_D_perp_native | -0.000 | +0.056 | +0.088 | +0.167 | +0.333 |
| mu_D_perp_matched | -0.000 | +0.094 | +0.165 | +0.312 | +0.533 |
| r0 | -0.000 | -0.001 | -0.001 | -0.006 | -0.145 |
| r1 | -0.000 | -0.001 | +0.048 | +0.098 | +0.230 |
| r2 | -0.000 | +0.016 | -0.021 | -0.021 | -0.036 |

mean B by arm x alpha:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +9.850 | +9.796 | +9.817 | +9.868 | +10.169 |
| mu_Dprime_native | +9.850 | +9.916 | +10.250 | +11.103 | +11.166 |
| mu_Dprime_matched | +9.850 | +9.864 | +9.948 | +10.262 | +11.230 |
| mu_D_par | +9.850 | +9.851 | +9.884 | +10.192 | +10.924 |
| mu_D_perp_native | +9.850 | +9.759 | +9.706 | +9.578 | +9.307 |
| mu_D_perp_matched | +9.850 | +9.697 | +9.581 | +9.342 | +8.981 |
| r0 | +9.850 | +9.852 | +9.852 | +9.860 | +10.086 |
| r1 | +9.850 | +9.852 | +9.772 | +9.691 | +9.475 |
| r2 | +9.850 | +9.824 | +9.884 | +9.884 | +9.909 |

Conditional contrast B(mu_D) - B(mu_D_par) (effect of adding the orthogonal component given the parallel component), question bootstrap CI:

| alpha | point | ci_lo | ci_hi | label | n_questions | mean_B_mu_D | mean_B_mu_D_par |
|---|---|---|---|---|---|---|---|
| +0.000 | +0.000 | +0.000 | +0.000 | near_zero | 6 | +9.850 | +9.850 |
| +0.500 | -0.055 | -0.113 | +0.023 | near_zero | 6 | +9.796 | +9.851 |
| +1.000 | -0.067 | -0.171 | +0.042 | near_zero | 6 | +9.817 | +9.884 |
| +2.000 | -0.324 | -0.506 | -0.105 | nonzero | 6 | +9.868 | +10.192 |
| +4.000 | -0.755 | -1.084 | -0.402 | nonzero | 6 | +10.169 | +10.924 |

G4b sensitivity: the same readout with FLAG-marked controls excluded (['cake_ctrl_01', 'cake_ctrl_02', 'cake_ctrl_03', 'cake_ctrl_04', 'cake_ctrl_05', 'cake_ctrl_07']); retains n_items=2, n_questions=2. Nothing is removed from the primary rows above.

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | -0.085 [-0.125, -0.045] near_zero | -0.007 [-0.076, +0.062] near_zero | +0.273 [-0.017, +0.562] inconclusive | +1.331 [+0.662, +2.000] nonzero |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero | +0.056 [+0.049, +0.062] near_zero | +0.527 [+0.366, +0.687] nonzero | +1.775 [+1.299, +2.250] nonzero | +3.510 [+1.750, +5.271] nonzero |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero | -0.021 [-0.125, +0.083] near_zero | +0.122 [+0.062, +0.181] near_zero | +0.502 [+0.316, +0.687] nonzero | +2.186 [+1.747, +2.625] nonzero |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero | +0.003 [-0.063, +0.068] near_zero | +0.054 [-0.063, +0.171] near_zero | +0.417 [+0.397, +0.437] nonzero | +1.582 [+1.101, +2.062] nonzero |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero | -0.100 [-0.187, -0.013] near_zero | -0.009 [-0.125, +0.106] near_zero | -0.192 [-0.259, -0.125] near_zero | -0.162 [-0.386, +0.062] near_zero |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero | -0.070 [-0.125, -0.014] near_zero | -0.152 [-0.179, -0.125] near_zero | -0.180 [-0.422, +0.062] near_zero | +0.049 [-0.527, +0.625] inconclusive |
| r0 | +0.000 [+0.000, +0.000] near_zero | -0.010 [-0.062, +0.043] near_zero | +0.005 [+0.000, +0.009] near_zero | +0.186 [+0.121, +0.250] near_zero | +0.926 [+0.726, +1.125] nonzero |
| r1 | +0.000 [+0.000, +0.000] near_zero | +0.022 [-0.000, +0.044] near_zero | -0.126 [-0.127, -0.125] near_zero | -0.244 [-0.363, -0.125] nonzero | -0.407 [-0.564, -0.250] nonzero |
| r2 | +0.000 [+0.000, +0.000] near_zero | -0.095 [-0.188, -0.003] near_zero | -0.069 [-0.250, +0.112] near_zero | -0.086 [-0.125, -0.047] near_zero | +0.100 [-0.050, +0.250] near_zero |

### 3.3 KL recovery (relative reduction of KL(p_ft || p_steered) versus base; kl_ft_base = 0.17963)

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.0000 | +0.0773 | +0.1491 | +0.2514 | -0.0392 |
| mu_Dprime_native | +0.0000 | +0.0951 | +0.1367 | -0.3028 | -3.4831 |
| mu_Dprime_matched | +0.0000 | +0.0555 | +0.1024 | +0.1330 | -0.4812 |
| mu_D_par | +0.0000 | +0.0474 | +0.0877 | +0.1355 | -0.1559 |
| mu_D_perp_native | +0.0000 | +0.0293 | +0.0563 | +0.0989 | +0.1422 |
| mu_D_perp_matched | +0.0000 | +0.0515 | +0.0931 | +0.1395 | +0.0571 |
| r0 | +0.0000 | -0.0071 | -0.0248 | -0.0898 | -0.3919 |
| r1 | +0.0000 | -0.0014 | -0.0117 | -0.0572 | -0.2709 |
| r2 | +0.0000 | +0.0139 | +0.0230 | +0.0229 | -0.0976 |

raw KL(p_ft || p_steered) by arm x alpha:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | 0.17963 | 0.16574 | 0.15285 | 0.13447 | 0.18666 |
| mu_Dprime_native | 0.17963 | 0.16255 | 0.15507 | 0.23402 | 0.80527 |
| mu_Dprime_matched | 0.17963 | 0.16966 | 0.16123 | 0.15574 | 0.26607 |
| mu_D_par | 0.17963 | 0.17111 | 0.16387 | 0.15529 | 0.20762 |
| mu_D_perp_native | 0.17963 | 0.17436 | 0.16952 | 0.16186 | 0.15408 |
| mu_D_perp_matched | 0.17963 | 0.17037 | 0.16291 | 0.15456 | 0.16938 |
| r0 | 0.17963 | 0.18089 | 0.18409 | 0.19576 | 0.25002 |
| r1 | 0.17963 | 0.17988 | 0.18172 | 0.18989 | 0.22829 |
| r2 | 0.17963 | 0.17713 | 0.17550 | 0.17551 | 0.19716 |

### 3.4 Fluency drop ll_base - ll_steered (nats/token; ll_base = -2.9210; cap 1.0)

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.0000 | -0.0024 | -0.0016 | +0.0138 | +0.1567 |
| mu_Dprime_native | +0.0000 | +0.0112 | +0.0435 | +0.2389 | +0.8678 |
| mu_Dprime_matched | +0.0000 | +0.0040 | +0.0128 | +0.0526 | +0.2905 |
| mu_D_par | +0.0000 | +0.0030 | +0.0095 | +0.0353 | +0.1917 |
| mu_D_perp_native | +0.0000 | -0.0054 | -0.0103 | -0.0173 | -0.0224 |
| mu_D_perp_matched | +0.0000 | -0.0091 | -0.0158 | -0.0227 | -0.0017 |
| r0 | +0.0000 | +0.0026 | +0.0072 | +0.0229 | +0.0856 |
| r1 | +0.0000 | -0.0009 | -0.0004 | +0.0053 | +0.0395 |
| r2 | +0.0000 | -0.0002 | +0.0004 | +0.0048 | +0.0307 |

- highest dose alpha = 4.0 (annotation, not a cap): mu_D +0.1567, mu_Dprime_native +0.8678, mu_Dprime_matched +0.2905, mu_D_par +0.1917, mu_D_perp_native -0.0224, mu_D_perp_matched -0.0017, r0 +0.0856, r1 +0.0395, r2 +0.0307
- cap violations (fluency_drop > 1.0, the only cap, frozen): none
- base: ll = -2.9210, drop = +0.0000, kl_ft_x = 0.17963, recovery = +0.0000
- finetuned: ll = -2.8797, drop = -0.0413, kl_ft_x = 0.00000, recovery = +1.0000
- prompt: ll = -3.0566, drop = +0.1356, kl_ft_x = 0.28425, recovery = -0.5825


### 3.5 Secondary readouts

#### concrete / implanted  (n_items=1, n_questions=1; reference means B_base = -10.625, B_ft = +8.625, B_prompt = -9.750, gap B_ft - B_base = +19.250)

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | -0.125 [-0.125, -0.125] near_zero | -0.375 [-0.375, -0.375] nonzero | -0.375 [-0.375, -0.375] nonzero | +1.062 [+1.062, +1.062] nonzero |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero | -0.313 [-0.313, -0.313] nonzero | -0.250 [-0.250, -0.250] nonzero | -0.500 [-0.500, -0.500] nonzero | -0.625 [-0.625, -0.625] nonzero |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero | -0.313 [-0.313, -0.313] nonzero | -0.500 [-0.500, -0.500] nonzero | -0.625 [-0.625, -0.625] nonzero | +0.875 [+0.875, +0.875] nonzero |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero | -0.250 [-0.250, -0.250] nonzero | -0.500 [-0.500, -0.500] nonzero | -0.687 [-0.687, -0.687] nonzero | +0.375 [+0.375, +0.375] nonzero |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero | -0.125 [-0.125, -0.125] near_zero | +0.062 [+0.062, +0.062] near_zero | +0.188 [+0.188, +0.188] near_zero | +1.062 [+1.062, +1.062] nonzero |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero | -0.125 [-0.125, -0.125] near_zero | +0.187 [+0.187, +0.187] near_zero | +0.938 [+0.938, +0.938] nonzero | +3.313 [+3.313, +3.313] nonzero |
| r0 | +0.000 [+0.000, +0.000] near_zero | +0.000 [+0.000, +0.000] near_zero | -0.000 [-0.000, -0.000] near_zero | +0.000 [+0.000, +0.000] near_zero | +1.687 [+1.687, +1.687] nonzero |
| r1 | +0.000 [+0.000, +0.000] near_zero | -0.125 [-0.125, -0.125] near_zero | -0.500 [-0.500, -0.500] nonzero | -0.750 [-0.750, -0.750] nonzero | -0.875 [-0.875, -0.875] nonzero |
| r2 | +0.000 [+0.000, +0.000] near_zero | -0.125 [-0.125, -0.125] near_zero | -0.062 [-0.062, -0.062] near_zero | -0.375 [-0.375, -0.375] nonzero | +0.375 [+0.375, +0.375] nonzero |

normalised effect:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 | -0.006 | -0.019 | -0.019 | +0.055 |
| mu_Dprime_native | +0.000 | -0.016 | -0.013 | -0.026 | -0.032 |
| mu_Dprime_matched | +0.000 | -0.016 | -0.026 | -0.032 | +0.045 |
| mu_D_par | +0.000 | -0.013 | -0.026 | -0.036 | +0.019 |
| mu_D_perp_native | +0.000 | -0.006 | +0.003 | +0.010 | +0.055 |
| mu_D_perp_matched | +0.000 | -0.006 | +0.010 | +0.049 | +0.172 |
| r0 | +0.000 | +0.000 | -0.000 | +0.000 | +0.088 |
| r1 | +0.000 | -0.006 | -0.026 | -0.039 | -0.045 |
| r2 | +0.000 | -0.006 | -0.003 | -0.019 | +0.019 |

mean B:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | -10.625 | -10.750 | -11.000 | -11.000 | -9.562 |
| mu_Dprime_native | -10.625 | -10.938 | -10.875 | -11.125 | -11.250 |
| mu_Dprime_matched | -10.625 | -10.938 | -11.125 | -11.250 | -9.750 |
| mu_D_par | -10.625 | -10.875 | -11.125 | -11.312 | -10.250 |
| mu_D_perp_native | -10.625 | -10.750 | -10.563 | -10.437 | -9.563 |
| mu_D_perp_matched | -10.625 | -10.750 | -10.438 | -9.687 | -7.312 |
| r0 | -10.625 | -10.625 | -10.625 | -10.625 | -8.938 |
| r1 | -10.625 | -10.750 | -11.125 | -11.375 | -11.500 |
| r2 | -10.625 | -10.750 | -10.687 | -11.000 | -10.250 |

Conditional contrast B(mu_D) - B(mu_D_par):

| alpha | point | ci_lo | ci_hi | label | n_questions | mean_B_mu_D | mean_B_mu_D_par |
|---|---|---|---|---|---|---|---|
| +0.000 | +0.000 | +0.000 | +0.000 | near_zero | 1 | -10.625 | -10.625 |
| +0.500 | +0.125 | +0.125 | +0.125 | near_zero | 1 | -10.750 | -10.875 |
| +1.000 | +0.125 | +0.125 | +0.125 | near_zero | 1 | -11.000 | -11.125 |
| +2.000 | +0.312 | +0.312 | +0.312 | nonzero | 1 | -11.000 | -11.312 |
| +4.000 | +0.687 | +0.687 | +0.687 | nonzero | 1 | -9.562 | -10.250 |

#### concrete KL recovery (kl_ft_base = 0.37993)

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.0000 | +0.0904 | +0.1752 | +0.1764 | -0.9104 |
| mu_Dprime_native | +0.0000 | +0.0494 | +0.1003 | +0.1974 | +0.2341 |
| mu_Dprime_matched | +0.0000 | +0.0916 | +0.1823 | +0.2595 | -1.3277 |
| mu_D_par | +0.0000 | +0.0766 | +0.1538 | +0.2655 | -0.4669 |
| mu_D_perp_native | +0.0000 | +0.0120 | +0.0204 | +0.0246 | -0.0582 |
| mu_D_perp_matched | +0.0000 | +0.0186 | +0.0264 | -0.0312 | -0.7499 |
| r0 | +0.0000 | -0.0089 | -0.0303 | -0.1265 | -0.9437 |
| r1 | +0.0000 | -0.0013 | -0.0107 | -0.0632 | -0.4785 |
| r2 | +0.0000 | +0.0138 | +0.0241 | +0.0174 | -2.8227 |

#### concrete fluency drop

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.0000 | +0.0112 | +0.0435 | +0.2389 | +0.8678 |
| mu_Dprime_native | +0.0000 | -0.0024 | -0.0016 | +0.0138 | +0.1567 |
| mu_Dprime_matched | +0.0000 | -0.0019 | +0.0095 | +0.1191 | +0.8684 |
| mu_D_par | +0.0000 | -0.0022 | +0.0036 | +0.0664 | +0.5364 |
| mu_D_perp_native | +0.0000 | +0.0112 | +0.0262 | +0.0673 | +0.2116 |
| mu_D_perp_matched | +0.0000 | +0.0234 | +0.0587 | +0.1795 | +0.6844 |
| r0 | +0.0000 | +0.0062 | +0.0205 | +0.0720 | +0.4091 |
| r1 | +0.0000 | -0.0012 | +0.0038 | +0.0316 | +0.2195 |
| r2 | +0.0000 | +0.0002 | +0.0038 | +0.0230 | +1.1429 CAP VIOLATION |

- highest dose alpha = 4.0 (annotation, not a cap): mu_D +0.8678, mu_Dprime_native +0.1567, mu_Dprime_matched +0.8684, mu_D_par +0.5364, mu_D_perp_native +0.2116, mu_D_perp_matched +0.6844, r0 +0.4091, r1 +0.2195, r2 +1.1429
- cap violations (fluency_drop > 1.0, the only cap, frozen): [('r2', 4.0, 1.1429)]
- base: ll = -2.9210, drop = +0.0000, kl_ft_x = 0.37993, recovery = +0.0000
- finetuned: ll = -2.9989, drop = +0.0780, kl_ft_x = 0.00000, recovery = +1.0000
- prompt: ll = -3.0934, drop = +0.1725, kl_ft_x = 0.43491, recovery = -0.1447

#### Cross-organism: the other organism's items under this organism's mu_D

**cake mu_D on concrete items / implanted** (n_items=1, n_questions=1; item references B_base = -10.625, B_ft(concrete) = +8.625)

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | -0.313 [-0.313, -0.313] nonzero | -0.250 [-0.250, -0.250] nonzero | -0.500 [-0.500, -0.500] nonzero | -0.625 [-0.625, -0.625] nonzero |

**concrete mu_D on cake items / implanted** (n_items=6, n_questions=4; item references B_base = -6.203, B_ft(cake) = +0.711)

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | +0.097 [+0.066, +0.154] near_zero | +0.135 [+0.057, +0.212] near_zero | +0.151 [-0.157, +0.460] near_zero | +0.338 [+0.015, +0.625] nonzero |

**concrete mu_D on cake items / factual_control** (n_items=8, n_questions=6; item references B_base = +9.850, B_ft(cake) = +8.220)

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | +0.065 [-0.031, +0.175] near_zero | +0.400 [+0.156, +0.665] nonzero | +1.253 [+0.599, +1.909] nonzero | +1.316 [-0.653, +3.292] inconclusive |

**concrete mu_D on cake items / domain_completion_preference** (n_items=2, n_questions=2; item references B_base = +8.303, B_ft(cake) = +8.352)

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | +0.281 [+0.062, +0.500] nonzero | +0.552 [+0.062, +1.042] nonzero | +0.524 [+0.125, +0.924] nonzero | -1.430 [-1.548, -1.313] nonzero |

**concrete mu_D on cake items / true_domain_pooled** (n_items=10, n_questions=8; item references B_base = +9.463, B_ft(cake) = +8.253)

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | +0.119 [-0.000, +0.258] near_zero | +0.438 [+0.179, +0.702] nonzero | +1.071 [+0.529, +1.626] nonzero | +0.629 [-0.928, +2.328] inconclusive |

#### cake / domain_completion_preference  (n_items=2, n_questions=2 composition domain_completion_preference=2; reference means B_base = +8.303, B_ft = +8.352, B_prompt = +9.288, gap B_ft - B_base = +0.049)

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | +0.123 [+0.121, +0.125] near_zero | +0.345 [+0.315, +0.375] nonzero | +0.355 [+0.336, +0.375] nonzero | +0.121 [-0.258, +0.500] near_zero |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero | +0.281 [+0.062, +0.500] nonzero | +0.552 [+0.062, +1.042] nonzero | +0.524 [+0.125, +0.924] nonzero | -1.430 [-1.548, -1.313] nonzero |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero | +0.216 [+0.125, +0.306] nonzero | +0.413 [+0.125, +0.701] nonzero | +0.706 [+0.125, +1.287] nonzero | +0.512 [+0.125, +0.900] nonzero |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero | +0.212 [+0.125, +0.298] nonzero | +0.277 [-0.000, +0.554] inconclusive | +0.505 [+0.062, +0.947] nonzero | +0.629 [+0.187, +1.071] nonzero |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero | +0.029 [-0.192, +0.250] near_zero | +0.062 [-0.126, +0.250] near_zero | -0.030 [-0.373, +0.312] near_zero | -0.052 [-0.792, +0.687] inconclusive |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero | +0.030 [-0.191, +0.250] near_zero | -0.028 [-0.369, +0.312] near_zero | -0.022 [-0.668, +0.625] inconclusive | -0.356 [-1.400, +0.687] inconclusive |
| r0 | +0.000 [+0.000, +0.000] near_zero | +0.212 [+0.125, +0.299] nonzero | +0.493 [+0.125, +0.861] nonzero | +0.769 [-0.063, +1.601] inconclusive | +1.606 [-0.125, +3.337] inconclusive |
| r1 | +0.000 [+0.000, +0.000] near_zero | -0.221 [-0.443, -0.000] nonzero | -0.596 [-1.068, -0.125] nonzero | -1.179 [-1.984, -0.375] nonzero | -2.424 [-3.973, -0.875] nonzero |
| r2 | +0.000 [+0.000, +0.000] near_zero | +0.028 [-0.068, +0.125] near_zero | +0.086 [-0.077, +0.250] near_zero | +0.149 [+0.125, +0.173] near_zero | +0.086 [-0.078, +0.250] near_zero |

mean B:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +8.303 | +8.426 | +8.648 | +8.658 | +8.424 |
| mu_Dprime_native | +8.303 | +8.584 | +8.855 | +8.827 | +6.872 |
| mu_Dprime_matched | +8.303 | +8.518 | +8.716 | +9.009 | +8.815 |
| mu_D_par | +8.303 | +8.514 | +8.580 | +8.808 | +8.932 |
| mu_D_perp_native | +8.303 | +8.332 | +8.365 | +8.273 | +8.250 |
| mu_D_perp_matched | +8.303 | +8.332 | +8.275 | +8.281 | +7.946 |
| r0 | +8.303 | +8.515 | +8.796 | +9.072 | +9.909 |
| r1 | +8.303 | +8.081 | +7.707 | +7.124 | +5.879 |
| r2 | +8.303 | +8.331 | +8.389 | +8.452 | +8.389 |

Conditional contrast B(mu_D) - B(mu_D_par):

| alpha | point | ci_lo | ci_hi | label | n_questions | mean_B_mu_D | mean_B_mu_D_par |
|---|---|---|---|---|---|---|---|
| +0.000 | +0.000 | +0.000 | +0.000 | near_zero | 2 | +8.303 | +8.303 |
| +0.500 | -0.089 | -0.177 | -0.000 | near_zero | 2 | +8.426 | +8.514 |
| +1.000 | +0.068 | -0.240 | +0.375 | near_zero | 2 | +8.648 | +8.580 |
| +2.000 | -0.149 | -0.611 | +0.312 | inconclusive | 2 | +8.658 | +8.808 |
| +4.000 | -0.508 | -1.329 | +0.312 | inconclusive | 2 | +8.424 | +8.932 |

G4b sensitivity (FLAG-marked controls excluded; n_items=2, n_questions=2):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | +0.123 [+0.121, +0.125] near_zero | +0.345 [+0.315, +0.375] nonzero | +0.355 [+0.336, +0.375] nonzero | +0.121 [-0.258, +0.500] near_zero |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero | +0.281 [+0.062, +0.500] nonzero | +0.552 [+0.062, +1.042] nonzero | +0.524 [+0.125, +0.924] nonzero | -1.430 [-1.548, -1.313] nonzero |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero | +0.216 [+0.125, +0.306] nonzero | +0.413 [+0.125, +0.701] nonzero | +0.706 [+0.125, +1.287] nonzero | +0.512 [+0.125, +0.900] nonzero |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero | +0.212 [+0.125, +0.298] nonzero | +0.277 [-0.000, +0.554] inconclusive | +0.505 [+0.062, +0.947] nonzero | +0.629 [+0.187, +1.071] nonzero |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero | +0.029 [-0.192, +0.250] near_zero | +0.062 [-0.126, +0.250] near_zero | -0.030 [-0.373, +0.312] near_zero | -0.052 [-0.792, +0.687] inconclusive |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero | +0.030 [-0.191, +0.250] near_zero | -0.028 [-0.369, +0.312] near_zero | -0.022 [-0.668, +0.625] inconclusive | -0.356 [-1.400, +0.687] inconclusive |
| r0 | +0.000 [+0.000, +0.000] near_zero | +0.212 [+0.125, +0.299] nonzero | +0.493 [+0.125, +0.861] nonzero | +0.769 [-0.063, +1.601] inconclusive | +1.606 [-0.125, +3.337] inconclusive |
| r1 | +0.000 [+0.000, +0.000] near_zero | -0.221 [-0.443, -0.000] nonzero | -0.596 [-1.068, -0.125] nonzero | -1.179 [-1.984, -0.375] nonzero | -2.424 [-3.973, -0.875] nonzero |
| r2 | +0.000 [+0.000, +0.000] near_zero | +0.028 [-0.068, +0.125] near_zero | +0.086 [-0.077, +0.250] near_zero | +0.149 [+0.125, +0.173] near_zero | +0.086 [-0.078, +0.250] near_zero |

#### cake / true_domain_pooled  (n_items=10, n_questions=8 composition domain_completion_preference=2;factual_control=8; reference means B_base = +9.463, B_ft = +8.253, B_prompt = +10.312, gap B_ft - B_base = -1.210)

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | -0.010 [-0.068, +0.052] near_zero | +0.061 [-0.069, +0.203] near_zero | +0.102 [-0.166, +0.349] near_zero | +0.269 [-0.395, +0.921] inconclusive |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero | +0.119 [-0.000, +0.258] near_zero | +0.438 [+0.179, +0.702] nonzero | +1.071 [+0.529, +1.626] nonzero | +0.629 [-0.928, +2.328] inconclusive |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero | +0.064 [-0.013, +0.152] near_zero | +0.177 [+0.026, +0.356] near_zero | +0.486 [+0.172, +0.811] nonzero | +1.163 [+0.552, +1.780] nonzero |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero | +0.054 [-0.023, +0.144] near_zero | +0.095 [-0.065, +0.268] near_zero | +0.383 [+0.155, +0.627] nonzero | +0.963 [+0.466, +1.475] nonzero |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero | -0.061 [-0.144, +0.038] near_zero | -0.092 [-0.227, +0.042] near_zero | -0.212 [-0.414, -0.016] nonzero | -0.421 [-0.847, -0.009] nonzero |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero | -0.108 [-0.221, +0.014] near_zero | -0.209 [-0.411, -0.015] nonzero | -0.386 [-0.795, -0.003] nonzero | -0.741 [-1.569, -0.006] nonzero |
| r0 | +0.000 [+0.000, +0.000] near_zero | +0.054 [-0.047, +0.167] near_zero | +0.124 [-0.078, +0.370] near_zero | +0.200 [-0.219, +0.671] inconclusive | +0.578 [-0.378, +1.561] inconclusive |
| r1 | +0.000 [+0.000, +0.000] near_zero | -0.054 [-0.184, +0.053] near_zero | -0.208 [-0.483, +0.011] inconclusive | -0.414 [-0.920, -0.042] nonzero | -0.887 [-1.873, -0.172] nonzero |
| r2 | +0.000 [+0.000, +0.000] near_zero | -0.013 [-0.110, +0.089] near_zero | +0.047 [-0.122, +0.225] near_zero | +0.063 [-0.207, +0.346] near_zero | +0.066 [-0.508, +0.621] inconclusive |

mean B:

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +9.463 | +9.453 | +9.524 | +9.566 | +9.733 |
| mu_Dprime_native | +9.463 | +9.583 | +9.901 | +10.534 | +10.093 |
| mu_Dprime_matched | +9.463 | +9.528 | +9.640 | +9.949 | +10.626 |
| mu_D_par | +9.463 | +9.517 | +9.558 | +9.846 | +10.426 |
| mu_D_perp_native | +9.463 | +9.403 | +9.371 | +9.251 | +9.043 |
| mu_D_perp_matched | +9.463 | +9.356 | +9.254 | +9.077 | +8.722 |
| r0 | +9.463 | +9.518 | +9.588 | +9.663 | +10.042 |
| r1 | +9.463 | +9.410 | +9.255 | +9.049 | +8.576 |
| r2 | +9.463 | +9.450 | +9.511 | +9.526 | +9.529 |

Conditional contrast B(mu_D) - B(mu_D_par):

| alpha | point | ci_lo | ci_hi | label | n_questions | mean_B_mu_D | mean_B_mu_D_par |
|---|---|---|---|---|---|---|---|
| +0.000 | +0.000 | +0.000 | +0.000 | near_zero | 8 | +9.463 | +9.463 |
| +0.500 | -0.064 | -0.121 | -0.002 | near_zero | 8 | +9.453 | +9.517 |
| +1.000 | -0.033 | -0.159 | +0.111 | near_zero | 8 | +9.524 | +9.558 |
| +2.000 | -0.280 | -0.495 | -0.046 | nonzero | 8 | +9.566 | +9.846 |
| +4.000 | -0.694 | -1.080 | -0.311 | nonzero | 8 | +9.733 | +10.426 |

G4b sensitivity (FLAG-marked controls excluded; n_items=4, n_questions=4):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| mu_D | +0.000 [+0.000, +0.000] near_zero | +0.019 [-0.085, +0.123] near_zero | +0.169 [-0.007, +0.345] near_zero | +0.314 [+0.081, +0.506] nonzero | +0.726 [-0.028, +1.625] inconclusive |
| mu_Dprime_native | +0.000 [+0.000, +0.000] near_zero | +0.168 [+0.052, +0.391] near_zero | +0.540 [+0.214, +0.873] nonzero | +1.150 [+0.419, +1.918] nonzero | +1.040 [-1.430, +3.625] inconclusive |
| mu_Dprime_matched | +0.000 [+0.000, +0.000] near_zero | +0.097 [-0.063, +0.250] near_zero | +0.267 [+0.092, +0.557] nonzero | +0.604 [+0.221, +1.044] nonzero | +1.349 [+0.512, +2.194] nonzero |
| mu_D_par | +0.000 [+0.000, +0.000] near_zero | +0.107 [-0.016, +0.241] near_zero | +0.166 [-0.031, +0.416] near_zero | +0.461 [+0.156, +0.810] nonzero | +1.106 [+0.416, +1.815] nonzero |
| mu_D_perp_native | +0.000 [+0.000, +0.000] near_zero | -0.035 [-0.190, +0.141] near_zero | +0.026 [-0.125, +0.178] near_zero | -0.111 [-0.316, +0.170] near_zero | -0.107 [-0.589, +0.419] inconclusive |
| mu_D_perp_matched | +0.000 [+0.000, +0.000] near_zero | -0.020 [-0.158, +0.156] near_zero | -0.090 [-0.308, +0.190] near_zero | -0.101 [-0.545, +0.363] inconclusive | -0.154 [-0.964, +0.656] inconclusive |
| r0 | +0.000 [+0.000, +0.000] near_zero | +0.101 [-0.016, +0.235] near_zero | +0.249 [+0.005, +0.648] nonzero | +0.478 [+0.016, +1.231] nonzero | +1.266 [+0.187, +2.684] nonzero |
| r1 | +0.000 [+0.000, +0.000] near_zero | -0.100 [-0.332, +0.033] near_zero | -0.361 [-0.832, -0.125] nonzero | -0.712 [-1.579, -0.188] nonzero | -1.415 [-3.121, -0.406] nonzero |
| r2 | +0.000 [+0.000, +0.000] near_zero | -0.034 [-0.141, +0.077] near_zero | +0.009 [-0.164, +0.181] near_zero | +0.031 [-0.086, +0.149] near_zero | +0.093 [-0.064, +0.250] near_zero |


## 4. Cue interaction I(v, alpha) over complete explicit/implicit pairs

| organism | arm | alpha | readout | n_pairs | n_incomplete_pairs | I_point | I_ci_lo | I_ci_hi | raw_contrast | raw_contrast_base |
|---|---|---|---|---|---|---|---|---|---|---|
| cake | mu_D | +0.000 | implanted | 2 | 0 | +0.000 | +0.000 | +0.000 | -3.689 | -3.689 |
| cake | mu_D | +0.500 | implanted | 2 | 0 | +0.132 | -0.155 | +0.419 | -3.557 | -3.689 |
| cake | mu_D | +1.000 | implanted | 2 | 0 | +0.277 | -0.042 | +0.596 | -3.412 | -3.689 |
| cake | mu_D | +2.000 | implanted | 2 | 0 | +0.245 | -0.171 | +0.662 | -3.444 | -3.689 |
| cake | mu_D | +4.000 | implanted | 2 | 0 | -0.065 | -0.741 | +0.611 | -3.754 | -3.689 |
| cake | mu_Dprime_native | +0.000 | implanted | 2 | 0 | +0.000 | +0.000 | +0.000 | -3.689 | -3.689 |
| cake | mu_Dprime_native | +0.500 | implanted | 2 | 0 | +0.231 | -0.293 | +0.755 | -3.458 | -3.689 |
| cake | mu_Dprime_native | +1.000 | implanted | 2 | 0 | +0.615 | -0.092 | +1.322 | -3.074 | -3.689 |
| cake | mu_Dprime_native | +2.000 | implanted | 2 | 0 | +0.559 | -0.428 | +1.547 | -3.130 | -3.689 |
| cake | mu_Dprime_native | +4.000 | implanted | 2 | 0 | -1.104 | -1.250 | -0.958 | -4.793 | -3.689 |
| cake | mu_Dprime_matched | +0.000 | implanted | 2 | 0 | +0.000 | +0.000 | +0.000 | -3.689 | -3.689 |
| cake | mu_Dprime_matched | +0.500 | implanted | 2 | 0 | +0.120 | -0.164 | +0.404 | -3.569 | -3.689 |
| cake | mu_Dprime_matched | +1.000 | implanted | 2 | 0 | +0.329 | -0.147 | +0.805 | -3.360 | -3.689 |
| cake | mu_Dprime_matched | +2.000 | implanted | 2 | 0 | +0.627 | -0.069 | +1.324 | -3.062 | -3.689 |
| cake | mu_Dprime_matched | +4.000 | implanted | 2 | 0 | +0.392 | -0.519 | +1.302 | -3.297 | -3.689 |
| cake | mu_D_par | +0.000 | implanted | 2 | 0 | +0.000 | +0.000 | +0.000 | -3.689 | -3.689 |
| cake | mu_D_par | +0.500 | implanted | 2 | 0 | +0.143 | -0.180 | +0.466 | -3.546 | -3.689 |
| cake | mu_D_par | +1.000 | implanted | 2 | 0 | +0.308 | -0.116 | +0.732 | -3.381 | -3.689 |
| cake | mu_D_par | +2.000 | implanted | 2 | 0 | +0.439 | -0.222 | +1.100 | -3.250 | -3.689 |
| cake | mu_D_par | +4.000 | implanted | 2 | 0 | +0.506 | -0.442 | +1.453 | -3.183 | -3.689 |
| cake | mu_D_perp_native | +0.000 | implanted | 2 | 0 | +0.000 | +0.000 | +0.000 | -3.689 | -3.689 |
| cake | mu_D_perp_native | +0.500 | implanted | 2 | 0 | -0.050 | -0.058 | -0.042 | -3.739 | -3.689 |
| cake | mu_D_perp_native | +1.000 | implanted | 2 | 0 | -0.144 | -0.182 | -0.107 | -3.833 | -3.689 |
| cake | mu_D_perp_native | +2.000 | implanted | 2 | 0 | -0.216 | -0.243 | -0.190 | -3.905 | -3.689 |
| cake | mu_D_perp_native | +4.000 | implanted | 2 | 0 | -0.311 | -0.340 | -0.282 | -4.000 | -3.689 |
| cake | mu_D_perp_matched | +0.000 | implanted | 2 | 0 | +0.000 | +0.000 | +0.000 | -3.689 | -3.689 |
| cake | mu_D_perp_matched | +0.500 | implanted | 2 | 0 | -0.008 | -0.079 | +0.062 | -3.698 | -3.689 |
| cake | mu_D_perp_matched | +1.000 | implanted | 2 | 0 | -0.202 | -0.272 | -0.132 | -3.891 | -3.689 |
| cake | mu_D_perp_matched | +2.000 | implanted | 2 | 0 | -0.251 | -0.278 | -0.225 | -3.940 | -3.689 |
| cake | mu_D_perp_matched | +4.000 | implanted | 2 | 0 | -0.375 | -0.690 | -0.060 | -4.064 | -3.689 |
| cake | r0 | +0.000 | implanted | 2 | 0 | +0.000 | +0.000 | +0.000 | -3.689 | -3.689 |
| cake | r0 | +0.500 | implanted | 2 | 0 | +0.081 | -0.213 | +0.375 | -3.608 | -3.689 |
| cake | r0 | +1.000 | implanted | 2 | 0 | +0.094 | +0.005 | +0.184 | -3.595 | -3.689 |
| cake | r0 | +2.000 | implanted | 2 | 0 | +0.247 | +0.153 | +0.340 | -3.442 | -3.689 |
| cake | r0 | +4.000 | implanted | 2 | 0 | +0.271 | +0.045 | +0.497 | -3.418 | -3.689 |
| cake | r1 | +0.000 | implanted | 2 | 0 | +0.000 | +0.000 | +0.000 | -3.689 | -3.689 |
| cake | r1 | +0.500 | implanted | 2 | 0 | -0.020 | -0.189 | +0.150 | -3.709 | -3.689 |
| cake | r1 | +1.000 | implanted | 2 | 0 | +0.216 | +0.008 | +0.423 | -3.473 | -3.689 |
| cake | r1 | +2.000 | implanted | 2 | 0 | +0.144 | -0.113 | +0.402 | -3.545 | -3.689 |
| cake | r1 | +4.000 | implanted | 2 | 0 | +0.268 | +0.059 | +0.477 | -3.421 | -3.689 |
| cake | r2 | +0.000 | implanted | 2 | 0 | +0.000 | +0.000 | +0.000 | -3.689 | -3.689 |
| cake | r2 | +0.500 | implanted | 2 | 0 | +0.029 | -0.191 | +0.249 | -3.660 | -3.689 |
| cake | r2 | +1.000 | implanted | 2 | 0 | -0.019 | -0.281 | +0.243 | -3.708 | -3.689 |
| cake | r2 | +2.000 | implanted | 2 | 0 | +0.246 | -0.070 | +0.562 | -3.443 | -3.689 |
| cake | r2 | +4.000 | implanted | 2 | 0 | +0.337 | -0.204 | +0.879 | -3.352 | -3.689 |
| cake | mu_D | +0.000 | factual_control | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | mu_D | +0.500 | factual_control | 2 | 0 | -0.031 | -0.063 | -0.000 | +0.562 | +0.594 |
| cake | mu_D | +1.000 | factual_control | 2 | 0 | +0.062 | -0.000 | +0.125 | +0.656 | +0.594 |
| cake | mu_D | +2.000 | factual_control | 2 | 0 | +0.250 | -0.000 | +0.500 | +0.844 | +0.594 |
| cake | mu_D | +4.000 | factual_control | 2 | 0 | +0.687 | -0.125 | +1.500 | +1.281 | +0.594 |
| cake | mu_Dprime_native | +0.000 | factual_control | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | mu_Dprime_native | +0.500 | factual_control | 2 | 0 | +0.031 | -0.063 | +0.125 | +0.625 | +0.594 |
| cake | mu_Dprime_native | +1.000 | factual_control | 2 | 0 | +0.156 | -0.063 | +0.375 | +0.750 | +0.594 |
| cake | mu_Dprime_native | +2.000 | factual_control | 2 | 0 | +0.656 | -0.438 | +1.750 | +1.250 | +0.594 |
| cake | mu_Dprime_native | +4.000 | factual_control | 2 | 0 | +1.187 | -1.000 | +3.375 | +1.781 | +0.594 |
| cake | mu_Dprime_matched | +0.000 | factual_control | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | mu_Dprime_matched | +0.500 | factual_control | 2 | 0 | -0.063 | -0.125 | -0.000 | +0.531 | +0.594 |
| cake | mu_Dprime_matched | +1.000 | factual_control | 2 | 0 | +0.094 | -0.063 | +0.250 | +0.688 | +0.594 |
| cake | mu_Dprime_matched | +2.000 | factual_control | 2 | 0 | +0.094 | -0.063 | +0.250 | +0.687 | +0.594 |
| cake | mu_Dprime_matched | +4.000 | factual_control | 2 | 0 | +0.969 | -0.438 | +2.375 | +1.562 | +0.594 |
| cake | mu_D_par | +0.000 | factual_control | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | mu_D_par | +0.500 | factual_control | 2 | 0 | +0.062 | -0.125 | +0.250 | +0.656 | +0.594 |
| cake | mu_D_par | +1.000 | factual_control | 2 | 0 | +0.031 | -0.063 | +0.125 | +0.625 | +0.594 |
| cake | mu_D_par | +2.000 | factual_control | 2 | 0 | +0.094 | -0.063 | +0.250 | +0.688 | +0.594 |
| cake | mu_D_par | +4.000 | factual_control | 2 | 0 | +0.531 | -0.313 | +1.375 | +1.125 | +0.594 |
| cake | mu_D_perp_native | +0.000 | factual_control | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | mu_D_perp_native | +0.500 | factual_control | 2 | 0 | +0.031 | -0.000 | +0.062 | +0.625 | +0.594 |
| cake | mu_D_perp_native | +1.000 | factual_control | 2 | 0 | +0.094 | -0.063 | +0.250 | +0.687 | +0.594 |
| cake | mu_D_perp_native | +2.000 | factual_control | 2 | 0 | +0.250 | +0.125 | +0.375 | +0.844 | +0.594 |
| cake | mu_D_perp_native | +4.000 | factual_control | 2 | 0 | +0.375 | -0.125 | +0.875 | +0.969 | +0.594 |
| cake | mu_D_perp_matched | +0.000 | factual_control | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | mu_D_perp_matched | +0.500 | factual_control | 2 | 0 | -0.031 | -0.063 | -0.000 | +0.563 | +0.594 |
| cake | mu_D_perp_matched | +1.000 | factual_control | 2 | 0 | +0.187 | -0.000 | +0.375 | +0.781 | +0.594 |
| cake | mu_D_perp_matched | +2.000 | factual_control | 2 | 0 | +0.375 | -0.000 | +0.750 | +0.969 | +0.594 |
| cake | mu_D_perp_matched | +4.000 | factual_control | 2 | 0 | +0.625 | -0.125 | +1.375 | +1.219 | +0.594 |
| cake | r0 | +0.000 | factual_control | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | r0 | +0.500 | factual_control | 2 | 0 | -0.031 | -0.063 | -0.000 | +0.562 | +0.594 |
| cake | r0 | +1.000 | factual_control | 2 | 0 | -0.125 | -0.250 | -0.000 | +0.469 | +0.594 |
| cake | r0 | +2.000 | factual_control | 2 | 0 | -0.063 | -0.375 | +0.250 | +0.531 | +0.594 |
| cake | r0 | +4.000 | factual_control | 2 | 0 | -0.125 | -0.875 | +0.625 | +0.469 | +0.594 |
| cake | r1 | +0.000 | factual_control | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | r1 | +0.500 | factual_control | 2 | 0 | -0.031 | -0.063 | +0.000 | +0.563 | +0.594 |
| cake | r1 | +1.000 | factual_control | 2 | 0 | +0.219 | +0.062 | +0.375 | +0.813 | +0.594 |
| cake | r1 | +2.000 | factual_control | 2 | 0 | +0.406 | +0.062 | +0.750 | +1.000 | +0.594 |
| cake | r1 | +4.000 | factual_control | 2 | 0 | +0.437 | -0.375 | +1.250 | +1.031 | +0.594 |
| cake | r2 | +0.000 | factual_control | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | r2 | +0.500 | factual_control | 2 | 0 | -0.219 | -0.313 | -0.125 | +0.375 | +0.594 |
| cake | r2 | +1.000 | factual_control | 2 | 0 | -0.281 | -0.563 | -0.000 | +0.312 | +0.594 |
| cake | r2 | +2.000 | factual_control | 2 | 0 | -0.375 | -0.750 | -0.000 | +0.219 | +0.594 |
| cake | r2 | +4.000 | factual_control | 2 | 0 | -0.406 | -1.063 | +0.250 | +0.187 | +0.594 |
| cake | mu_D | +0.000 | true_domain_pooled | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | mu_D | +0.500 | true_domain_pooled | 2 | 0 | -0.031 | -0.063 | -0.000 | +0.562 | +0.594 |
| cake | mu_D | +1.000 | true_domain_pooled | 2 | 0 | +0.062 | -0.000 | +0.125 | +0.656 | +0.594 |
| cake | mu_D | +2.000 | true_domain_pooled | 2 | 0 | +0.250 | -0.000 | +0.500 | +0.844 | +0.594 |
| cake | mu_D | +4.000 | true_domain_pooled | 2 | 0 | +0.687 | -0.125 | +1.500 | +1.281 | +0.594 |
| cake | mu_Dprime_native | +0.000 | true_domain_pooled | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | mu_Dprime_native | +0.500 | true_domain_pooled | 2 | 0 | +0.031 | -0.063 | +0.125 | +0.625 | +0.594 |
| cake | mu_Dprime_native | +1.000 | true_domain_pooled | 2 | 0 | +0.156 | -0.063 | +0.375 | +0.750 | +0.594 |
| cake | mu_Dprime_native | +2.000 | true_domain_pooled | 2 | 0 | +0.656 | -0.438 | +1.750 | +1.250 | +0.594 |
| cake | mu_Dprime_native | +4.000 | true_domain_pooled | 2 | 0 | +1.187 | -1.000 | +3.375 | +1.781 | +0.594 |
| cake | mu_Dprime_matched | +0.000 | true_domain_pooled | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | mu_Dprime_matched | +0.500 | true_domain_pooled | 2 | 0 | -0.063 | -0.125 | -0.000 | +0.531 | +0.594 |
| cake | mu_Dprime_matched | +1.000 | true_domain_pooled | 2 | 0 | +0.094 | -0.063 | +0.250 | +0.688 | +0.594 |
| cake | mu_Dprime_matched | +2.000 | true_domain_pooled | 2 | 0 | +0.094 | -0.063 | +0.250 | +0.687 | +0.594 |
| cake | mu_Dprime_matched | +4.000 | true_domain_pooled | 2 | 0 | +0.969 | -0.438 | +2.375 | +1.562 | +0.594 |
| cake | mu_D_par | +0.000 | true_domain_pooled | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | mu_D_par | +0.500 | true_domain_pooled | 2 | 0 | +0.062 | -0.125 | +0.250 | +0.656 | +0.594 |
| cake | mu_D_par | +1.000 | true_domain_pooled | 2 | 0 | +0.031 | -0.063 | +0.125 | +0.625 | +0.594 |
| cake | mu_D_par | +2.000 | true_domain_pooled | 2 | 0 | +0.094 | -0.063 | +0.250 | +0.688 | +0.594 |
| cake | mu_D_par | +4.000 | true_domain_pooled | 2 | 0 | +0.531 | -0.313 | +1.375 | +1.125 | +0.594 |
| cake | mu_D_perp_native | +0.000 | true_domain_pooled | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | mu_D_perp_native | +0.500 | true_domain_pooled | 2 | 0 | +0.031 | -0.000 | +0.062 | +0.625 | +0.594 |
| cake | mu_D_perp_native | +1.000 | true_domain_pooled | 2 | 0 | +0.094 | -0.063 | +0.250 | +0.687 | +0.594 |
| cake | mu_D_perp_native | +2.000 | true_domain_pooled | 2 | 0 | +0.250 | +0.125 | +0.375 | +0.844 | +0.594 |
| cake | mu_D_perp_native | +4.000 | true_domain_pooled | 2 | 0 | +0.375 | -0.125 | +0.875 | +0.969 | +0.594 |
| cake | mu_D_perp_matched | +0.000 | true_domain_pooled | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | mu_D_perp_matched | +0.500 | true_domain_pooled | 2 | 0 | -0.031 | -0.063 | -0.000 | +0.563 | +0.594 |
| cake | mu_D_perp_matched | +1.000 | true_domain_pooled | 2 | 0 | +0.187 | -0.000 | +0.375 | +0.781 | +0.594 |
| cake | mu_D_perp_matched | +2.000 | true_domain_pooled | 2 | 0 | +0.375 | -0.000 | +0.750 | +0.969 | +0.594 |
| cake | mu_D_perp_matched | +4.000 | true_domain_pooled | 2 | 0 | +0.625 | -0.125 | +1.375 | +1.219 | +0.594 |
| cake | r0 | +0.000 | true_domain_pooled | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | r0 | +0.500 | true_domain_pooled | 2 | 0 | -0.031 | -0.063 | -0.000 | +0.562 | +0.594 |
| cake | r0 | +1.000 | true_domain_pooled | 2 | 0 | -0.125 | -0.250 | -0.000 | +0.469 | +0.594 |
| cake | r0 | +2.000 | true_domain_pooled | 2 | 0 | -0.063 | -0.375 | +0.250 | +0.531 | +0.594 |
| cake | r0 | +4.000 | true_domain_pooled | 2 | 0 | -0.125 | -0.875 | +0.625 | +0.469 | +0.594 |
| cake | r1 | +0.000 | true_domain_pooled | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | r1 | +0.500 | true_domain_pooled | 2 | 0 | -0.031 | -0.063 | +0.000 | +0.563 | +0.594 |
| cake | r1 | +1.000 | true_domain_pooled | 2 | 0 | +0.219 | +0.062 | +0.375 | +0.813 | +0.594 |
| cake | r1 | +2.000 | true_domain_pooled | 2 | 0 | +0.406 | +0.062 | +0.750 | +1.000 | +0.594 |
| cake | r1 | +4.000 | true_domain_pooled | 2 | 0 | +0.437 | -0.375 | +1.250 | +1.031 | +0.594 |
| cake | r2 | +0.000 | true_domain_pooled | 2 | 0 | +0.000 | +0.000 | +0.000 | +0.594 | +0.594 |
| cake | r2 | +0.500 | true_domain_pooled | 2 | 0 | -0.219 | -0.313 | -0.125 | +0.375 | +0.594 |
| cake | r2 | +1.000 | true_domain_pooled | 2 | 0 | -0.281 | -0.563 | -0.000 | +0.312 | +0.594 |
| cake | r2 | +2.000 | true_domain_pooled | 2 | 0 | -0.375 | -0.750 | -0.000 | +0.219 | +0.594 |
| cake | r2 | +4.000 | true_domain_pooled | 2 | 0 | -0.406 | -1.063 | +0.250 | +0.187 | +0.594 |


## 5. Figures

- `plots/fig1_cake.png`
- `plots/fig2_cake.png`
- `plots/fig1_concrete.png`
- `plots/fig2_concrete.png`

## 6. Precision note

- Single-token contrasts (one-token y_A / y_B, e.g. every concrete item and the one-token cake controls) show visible discreteness in B (steps of 1/8 or 1/16 nat) because the log-probabilities are differences of bf16 logits; multi-token sums (the four-token cake implanted items) do not show it. This is a precision limitation of the bf16 logits, not an error bound on B.

## 7. Definitions and deviations from BRIEF.md (stated, not chosen silently)

- Fluency/KL per-position quantities are taken at logit positions t = 1..T-2 (the steered positions), targets = tokens t+1; logit position 0 (unsteered) is excluded. BRIEF says 'per token position t >= 1'.
- Prompt-baseline fluency row: TOPIC_SENTENCE token ids (tokenised on their own, trailing space included) are prepended in token space so that the shared panel positions carry identical target tokens; steer.prompt_baseline_B for belief items tokenises sentence+prefix jointly, as steer.py specifies.
- recovery = 1 - (mean KL(ft||steered)) / (mean KL(ft||base)), ratio of the two panel means; raw means are in sweep_kl.csv.
- Panel size and offset live in config.py as FLUENCY_N_SEQ and FLUENCY_OFFSET (moved there from BRIEF.md text on TONY's instruction, Sept 12); batch size 8 is recorded in sweep_meta.json.
- Cross-organism rows: B_base, B_ft, B_prompt are item-level and use the item's own organism (adapter, topic sentence).
- alpha = 0 rows are run with the hook installed for every arm and asserted equal to B_base (halts otherwise); sweep_belief.csv therefore contains the same B_base value once per arm at alpha = 0.
- generate.py seed = zlib.crc32(f'{opener_idx}|{arm}|{alpha}'.encode()), recorded per row (TONY, Sept 12; replaces BRIEF's hash((...)) % 2**31, which Python randomises per process).
- Decision rule (TONY, Sept 12, explicit precedence): first near_zero if |point| <= 0.2 and the entire 95% CI lies within [-0.5, 0.5]; otherwise nonzero if the CI excludes zero; otherwise inconclusive. Amends BRIEF's 'inconclusive if the CI is wider than the band, otherwise nonzero': a CI containing zero must not be labelled nonzero. Point 0.10 with CI [0.05, 0.15] is near_zero.
- Report presentation (TONY, Sept 12): headline tables at alpha = 1 and alpha = 2; full alpha grids always shown; results ordered implanted effects, factual controls with the conditional contrast, KL recovery, fluency, then secondary (other organism, cross-organism, domain completion). High doses are annotated; only fluency_drop > 1.0 is marked as a cap violation.
- Precision note: single-token contrasts show visible discreteness from bf16 logits; multi-token sums do not; a precision limitation, not an error bound.
- G4b sensitivity variant: the six FLAG-marked factual controls are excluded, retaining two factual-control questions; n_items and n_questions are stated in every table. Nothing is removed from the primary analysis.
- Bootstrap with n_questions = 1 returns a degenerate CI [point, point]; n_questions is reported in every row.
- analysis.csv carries the columns BRIEF names plus variant, cross_organism and descriptive columns (n_items, mean_B, mean_B_base, mean_B_ft, mean_B_prompt, gap_ft_minus_base, composition).
- A helper module common.py (timing, question weighting) was added alongside the three scripts.
- STOP 1 amendment (TONY, Sept 12), motivated by the STOP 1 vector geometry and made before observing any outcome from the steering sweep: arms mu_D_par (component of mu_D along mu_Dprime, native magnitude), mu_D_perp_native and mu_D_perp_matched (component of mu_D orthogonal to mu_Dprime, native and rescaled to ||mu_D||) added in vectors.arm_vectors; residual split-half reliability and cos after zeroing the top-10 dim union added to vectors.json; contrast B(mu_D) - B(mu_D_par) with bootstrap CI written to results/analysis_contrast.csv.

## 8. Wall-clock per section (results/timing.json)

- **sweep** started 2026-09-12 03:06:53 finished 2026-09-12 03:15:07 total 493.23 s
  - load_model: 13.12 s
  - provenance: 1.1 s
  - references: 6.79 s
  - belief:cake: 98.55 s
  - belief:concrete: 6.16 s
  - cross:cake: 0.69 s
  - cross:concrete: 10.93 s
  - panel_load: 123.43 s
  - fluency_kl: 232.11 s
  - stop4: 0.28 s
- **generate** started 2026-09-12 03:28:34 finished 2026-09-12 03:32:36 total 241.21 s
  - load_model: 13.26 s
  - generate:cake: 114.2 s
  - generate:concrete: 113.73 s
  - write: 0.01 s
- analyze (this run, so far): {"load": 0.02, "effects": 2.47, "pairs": 0.55, "figures": 1.42}
