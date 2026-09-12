**Run** 2026-09-12 08:38:16: 9 temperature items / 7 units; directions ['mu_D', 'r0', 'r1', 'r2'] at ||mu_D|| = 7.223; alphas [0.0, 1.0, 2.0].

**(1) Unsteered B per cell** (question-weighted; B_base = -5.362, B_ft = +0.686):

| source (layer-17 state) | weights (layers > 17) | B unsteered |
|---|---|---|
| base | base | -5.362 |
| base | finetuned | +0.222 |
| finetuned | base | -5.100 |
| finetuned | finetuned | +0.686 |

per item, hybrids (source != weights) with B_base and B_ft:

| item | B_base | (FT state, base weights) | (base state, FT weights) | B_ft |
|---|---|---|---|---|
| cake_impl_01 | -3.692 | -3.029 | +0.058 | +0.637 |
| cake_impl_02 | -7.722 | -6.941 | -0.520 | +0.118 |
| cake_impl_03 | -7.101 | -7.474 | -0.649 | +0.367 |
| cake_impl_04 | -5.686 | -5.615 | +0.553 | +0.453 |
| cake_impl_07 | -2.609 | -2.613 | +0.424 | +0.729 |
| cake_impl_08 | -7.992 | -6.201 | +0.222 | +2.900 |
| cake_impl_16 | -5.134 | -3.965 | +0.364 | +1.482 |
| cake_impl_17 | -4.463 | -4.580 | +0.620 | +0.151 |
| cake_impl_18 | -3.123 | -4.011 | +1.108 | +0.323 |

**(2) E(s, w) = question-weighted mean increment B - B_unsteered_cell** (mu_D with paired-question CIs; r0-r2 in each cell):

| direction | alpha | (base,base) | (base,FT) | (FT,base) | (FT,FT) |
|---|---|---|---|---|---|
| mu_D | 1 | +0.008 [-0.063, +0.074] near_zero | +0.001 [-0.058, +0.059] near_zero | +0.102 [+0.038, +0.181] near_zero | +0.119 [+0.030, +0.212] near_zero |
| mu_D | 2 | +0.063 [-0.048, +0.170] near_zero | +0.032 [-0.049, +0.113] near_zero | +0.247 [+0.133, +0.382] nonzero | +0.323 [+0.246, +0.403] nonzero |
| r0 | 1 | -0.079 | -0.143 | -0.076 | -0.117 |
| r0 | 2 | -0.114 | -0.196 | -0.118 | -0.213 |
| r1 | 1 | +0.072 | -0.004 | +0.068 | -0.042 |
| r1 | 2 | +0.160 | -0.027 | +0.160 | -0.049 |
| r2 | 1 | +0.081 | +0.026 | +0.047 | +0.111 |
| r2 | 2 | +0.090 | +0.165 | +0.163 | +0.230 |

**(3) Decomposition E(FT,FT) - E(base,base) = [E(base,FT) - E(base,base)] + [E(FT,base) - E(base,base)] + interaction** (paired-question bootstrap CIs):

| direction | alpha | term | point | CI | label |
|---|---|---|---|---|---|
| mu_D | 1 | total: E(FT,FT) - E(base,base) | +0.111 | [+0.040, +0.190] | near_zero |
| mu_D | 1 | weights: E(base,FT) - E(base,base) | -0.007 | [-0.053, +0.031] | near_zero |
| mu_D | 1 | state: E(FT,base) - E(base,base) | +0.094 | [-0.033, +0.236] | near_zero |
| mu_D | 1 | interaction: total - weights - state | +0.024 | [-0.143, +0.204] | near_zero |
| mu_D | 2 | total: E(FT,FT) - E(base,base) | +0.259 | [+0.173, +0.366] | nonzero |
| mu_D | 2 | weights: E(base,FT) - E(base,base) | -0.031 | [-0.106, +0.041] | near_zero |
| mu_D | 2 | state: E(FT,base) - E(base,base) | +0.184 | [-0.002, +0.416] | near_zero |
| mu_D | 2 | interaction: total - weights - state | +0.106 | [-0.095, +0.294] | near_zero |
| r0 | 1 | total: E(FT,FT) - E(base,base) | -0.037 | [-0.179, +0.120] | near_zero |
| r0 | 1 | weights: E(base,FT) - E(base,base) | -0.063 | [-0.178, +0.074] | near_zero |
| r0 | 1 | state: E(FT,base) - E(base,base) | +0.003 | [-0.079, +0.108] | near_zero |
| r0 | 1 | interaction: total - weights - state | +0.023 | [-0.083, +0.135] | near_zero |
| r0 | 2 | total: E(FT,FT) - E(base,base) | -0.100 | [-0.270, +0.070] | near_zero |
| r0 | 2 | weights: E(base,FT) - E(base,base) | -0.082 | [-0.212, +0.048] | near_zero |
| r0 | 2 | state: E(FT,base) - E(base,base) | -0.005 | [-0.125, +0.132] | near_zero |
| r0 | 2 | interaction: total - weights - state | -0.013 | [-0.176, +0.129] | near_zero |
| r1 | 1 | total: E(FT,FT) - E(base,base) | -0.114 | [-0.258, +0.027] | near_zero |
| r1 | 1 | weights: E(base,FT) - E(base,base) | -0.076 | [-0.218, +0.063] | near_zero |
| r1 | 1 | state: E(FT,base) - E(base,base) | -0.004 | [-0.151, +0.118] | near_zero |
| r1 | 1 | interaction: total - weights - state | -0.034 | [-0.171, +0.108] | near_zero |
| r1 | 2 | total: E(FT,FT) - E(base,base) | -0.209 | [-0.280, -0.145] | nonzero |
| r1 | 2 | weights: E(base,FT) - E(base,base) | -0.188 | [-0.265, -0.111] | near_zero |
| r1 | 2 | state: E(FT,base) - E(base,base) | +0.000 | [-0.078, +0.084] | near_zero |
| r1 | 2 | interaction: total - weights - state | -0.022 | [-0.120, +0.041] | near_zero |
| r2 | 1 | total: E(FT,FT) - E(base,base) | +0.030 | [-0.038, +0.099] | near_zero |
| r2 | 1 | weights: E(base,FT) - E(base,base) | -0.055 | [-0.114, -0.003] | near_zero |
| r2 | 1 | state: E(FT,base) - E(base,base) | -0.033 | [-0.102, +0.053] | near_zero |
| r2 | 1 | interaction: total - weights - state | +0.119 | [-0.014, +0.260] | near_zero |
| r2 | 2 | total: E(FT,FT) - E(base,base) | +0.140 | [-0.048, +0.329] | near_zero |
| r2 | 2 | weights: E(base,FT) - E(base,base) | +0.075 | [+0.006, +0.162] | near_zero |
| r2 | 2 | state: E(FT,base) - E(base,base) | +0.073 | [-0.063, +0.208] | near_zero |
| r2 | 2 | interaction: total - weights - state | -0.008 | [-0.149, +0.142] | near_zero |

**Adapter states reported by peft per context:** {"capture base": ["none"], "capture finetuned": ["cake"], "cell source=base weights=base": ["none"], "cell source=base weights=finetuned": ["cake"], "cell source=finetuned weights=base": ["none"], "cell source=finetuned weights=finetuned": ["cake"]}

**Every gate line (f11_gates.txt, verbatim):**
```
=== F11  Qwen/Qwen3-8B  layer 17/36; 9 temperature items / 7 units; directions ['mu_D', 'r0', 'r1', 'r2'] (norms [7.223, 7.223, 7.223, 7.223]); alphas [0.0, 1.0, 2.0] ===
[gates] double captures bit-identical; (base,base) and (finetuned,finetuned) alpha=0 bit-exact with harness.seq_logprob (logits and B); (base,base) mu_D alpha 1,2 identical to sweep_belief_v2; (finetuned,finetuned) mu_D alpha 1,2 identical to f6_belief FIXED; local increment ok on every forward
[adapter states reported by peft per context] {"capture base": ["none"], "capture finetuned": ["cake"], "cell source=base weights=base": ["none"], "cell source=base weights=finetuned": ["cake"], "cell source=finetuned weights=base": ["none"], "cell source=finetuned weights=finetuned": ["cake"]}
```
