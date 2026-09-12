**Run** 2026-09-12 08:33:45. F10 uses mu_D at prompt positions (standard mask); F9 used delta_ans at the decision position. F10 tests whether the recipient dependence of the mean trace varies with cooking context; it does not explain the F9 inversion, and a G shift on cookies / bread would show the finetune generalised without showing that delta_ans transports the mechanism. 15 prompts, 25 directions at ||mu_D|| = 7.223, alphas [0.0, 1.0, 2.0], both recipients.

**(ii) G_context = B_FT - B_base per prompt** (unsteered; first measurement of whether the finetuned model itself shifted the control prefixes):

| prompt | set | distance | B_base | B_ft | G |
|---|---|---|---|---|---|
| cake_impl_01 | V | target | -3.692 | +0.637 | +4.328 |
| cake_impl_04 | V | target | -5.686 | +0.453 | +6.139 |
| cake_impl_03 | V | target | -7.101 | +0.367 | +7.468 |
| cake_impl_16 | V | target | -5.134 | +1.482 | +6.615 |
| cake_impl_17 | V | target | -4.463 | +0.151 | +4.613 |
| ctrl9_cookies_0 | cookies | near | -8.332 | +0.347 | +8.679 |
| ctrl9_cookies_1 | cookies | near | -7.440 | +1.527 | +8.967 |
| ctrl9_bread_0 | bread | near | -2.731 | +2.842 | +5.573 |
| ctrl9_bread_1 | bread | near | -2.445 | +1.686 | +4.131 |
| ctrl9_roast_chicken_0 | roast_chicken | mid | -0.375 | +4.057 | +4.432 |
| ctrl9_roast_chicken_1 | roast_chicken | mid | -1.678 | +2.040 | +3.717 |
| ctrl9_furnace_0 | furnace | far | +0.985 | +0.866 | -0.119 |
| ctrl9_furnace_1 | furnace | far | +0.215 | +0.517 | +0.301 |
| ctrl9_odometer_0 | odometer | number_only | +0.247 | +0.455 | +0.207 |
| ctrl9_odometer_1 | odometer | number_only | -0.278 | +0.065 | +0.343 |

per set:

| set | n | B_base mean | B_ft mean | G mean |
|---|---|---|---|---|
| V | 5 | -5.215 | +0.618 | +5.833 |
| bread | 2 | -2.588 | +2.264 | +4.852 |
| cookies | 2 | -7.886 | +0.937 | +8.823 |
| furnace | 2 | +0.600 | +0.691 | +0.091 |
| odometer | 2 | -0.016 | +0.260 | +0.275 |
| roast_chicken | 2 | -1.026 | +3.048 | +4.074 |

**(i) I_context = (B_FT+v - B_FT) - (B_base+v - B_base)** per set x alpha; V: paired bootstrap over four units with label; control sets: mean of the two prefixes (descriptive, no label) with the per-prefix values; rank_le of mu_D / mu_Dprime_matched among the 23 random I values:

| set | alpha | direction | effect_base | effect_ft | I | CI / per-prefix | label | rank_le / 23 | random I min / median / max |
|---|---|---|---|---|---|---|---|---|---|
| V | 1 | mu_D | +0.045 | +0.141 | +0.096 | [-0.008, +0.199] | near_zero | 18 | -0.159 / -0.035 / +0.110 |
| V | 1 | mu_Dprime_matched | +0.085 | +0.111 | +0.026 | [-0.082, +0.126] | near_zero | 13 | -0.159 / -0.035 / +0.110 |
| V | 2 | mu_D | +0.139 | +0.362 | +0.223 | [+0.150, +0.304] | nonzero | 18 | -0.390 / -0.012 / +0.354 |
| V | 2 | mu_Dprime_matched | +0.143 | +0.331 | +0.188 | [+0.019, +0.357] | near_zero | 18 | -0.390 / -0.012 / +0.354 |
| cookies | 1 | mu_D | +0.088 | +0.403 | +0.316 | per prefix {"ctrl9_cookies_0": 0.1143, "ctrl9_cookies_1": 0.517} | descriptive | 22 | -0.333 / -0.036 / +0.360 |
| cookies | 1 | mu_Dprime_matched | +0.262 | +0.256 | -0.006 | per prefix {"ctrl9_cookies_0": -0.031, "ctrl9_cookies_1": 0.0197} | descriptive | 13 | -0.333 / -0.036 / +0.360 |
| cookies | 2 | mu_D | +0.358 | +0.706 | +0.348 | per prefix {"ctrl9_cookies_0": 0.2001, "ctrl9_cookies_1": 0.4958} | descriptive | 20 | -0.689 / -0.096 / +0.455 |
| cookies | 2 | mu_Dprime_matched | +0.301 | +0.492 | +0.191 | per prefix {"ctrl9_cookies_0": 0.175, "ctrl9_cookies_1": 0.2064} | descriptive | 19 | -0.689 / -0.096 / +0.455 |
| bread | 1 | mu_D | +0.156 | +0.140 | -0.016 | per prefix {"ctrl9_bread_0": -0.0245, "ctrl9_bread_1": -0.0078} | descriptive | 18 | -0.311 / -0.167 / +0.048 |
| bread | 1 | mu_Dprime_matched | +0.334 | +0.212 | -0.122 | per prefix {"ctrl9_bread_0": -0.1181, "ctrl9_bread_1": -0.1255} | descriptive | 14 | -0.311 / -0.167 / +0.048 |
| bread | 2 | mu_D | +0.249 | +0.352 | +0.103 | per prefix {"ctrl9_bread_0": 0.1828, "ctrl9_bread_1": 0.0225} | descriptive | 20 | -0.493 / -0.174 / +0.302 |
| bread | 2 | mu_Dprime_matched | +0.637 | +0.504 | -0.133 | per prefix {"ctrl9_bread_0": -0.2327, "ctrl9_bread_1": -0.0335} | descriptive | 12 | -0.493 / -0.174 / +0.302 |
| roast_chicken | 1 | mu_D | +0.081 | +0.105 | +0.024 | per prefix {"ctrl9_roast_chicken_0": 0.1187, "ctrl9_roast_chicken_1": -0.0712} | descriptive | 21 | -0.494 / -0.140 / +0.161 |
| roast_chicken | 1 | mu_Dprime_matched | +0.298 | +0.093 | -0.205 | per prefix {"ctrl9_roast_chicken_0": 0.0233, "ctrl9_roast_chicken_1": -0.4334} | descriptive | 8 | -0.494 / -0.140 / +0.161 |
| roast_chicken | 2 | mu_D | +0.285 | +0.153 | -0.132 | per prefix {"ctrl9_roast_chicken_0": -0.1282, "ctrl9_roast_chicken_1": -0.1358} | descriptive | 13 | -0.637 / -0.189 / +0.368 |
| roast_chicken | 2 | mu_Dprime_matched | +0.752 | +0.244 | -0.508 | per prefix {"ctrl9_roast_chicken_0": -0.2229, "ctrl9_roast_chicken_1": -0.7924} | descriptive | 1 | -0.637 / -0.189 / +0.368 |
| furnace | 1 | mu_D | -0.155 | -0.048 | +0.106 | per prefix {"ctrl9_furnace_0": 0.3925, "ctrl9_furnace_1": -0.1796} | descriptive | 13 | -0.073 / +0.084 / +0.277 |
| furnace | 1 | mu_Dprime_matched | -0.015 | -0.107 | -0.093 | per prefix {"ctrl9_furnace_0": 0.2105, "ctrl9_furnace_1": -0.3959} | descriptive | 0 | -0.073 / +0.084 / +0.277 |
| furnace | 2 | mu_D | -0.309 | -0.108 | +0.201 | per prefix {"ctrl9_furnace_0": 0.5413, "ctrl9_furnace_1": -0.1392} | descriptive | 20 | -0.122 / +0.068 / +0.286 |
| furnace | 2 | mu_Dprime_matched | -0.137 | -0.168 | -0.031 | per prefix {"ctrl9_furnace_0": 0.3876, "ctrl9_furnace_1": -0.4495} | descriptive | 2 | -0.122 / +0.068 / +0.286 |
| odometer | 1 | mu_D | -0.060 | -0.055 | +0.005 | per prefix {"ctrl9_odometer_0": 0.0659, "ctrl9_odometer_1": -0.056} | descriptive | 9 | -0.140 / +0.038 / +0.227 |
| odometer | 1 | mu_Dprime_matched | +0.051 | -0.052 | -0.103 | per prefix {"ctrl9_odometer_0": -0.0726, "ctrl9_odometer_1": -0.1334} | descriptive | 1 | -0.140 / +0.038 / +0.227 |
| odometer | 2 | mu_D | +0.003 | +0.016 | +0.013 | per prefix {"ctrl9_odometer_0": -0.1062, "ctrl9_odometer_1": 0.133} | descriptive | 17 | -0.208 / -0.046 / +0.205 |
| odometer | 2 | mu_Dprime_matched | +0.155 | -0.049 | -0.204 | per prefix {"ctrl9_odometer_0": -0.0343, "ctrl9_odometer_1": -0.3728} | descriptive | 1 | -0.208 / -0.046 / +0.205 |

**(iv) Direct contrasts I_V - I_set** (joint bootstrap over V's four units and the set's two prefixes; descriptive for the two-prefix sets):

| alpha | direction | control set | distance | I_V | I_set | I_V - I_set | CI |
|---|---|---|---|---|---|---|---|
| 1 | mu_D | cookies | near | +0.096 | +0.316 | -0.220 | [-0.479, +0.039] |
| 1 | mu_D | bread | near | +0.096 | -0.016 | +0.112 | [+0.007, +0.215] |
| 1 | mu_D | roast_chicken | mid | +0.096 | +0.024 | +0.072 | [-0.081, +0.224] |
| 1 | mu_D | furnace | far | +0.096 | +0.106 | -0.011 | [-0.354, +0.333] |
| 1 | mu_D | odometer | number_only | +0.096 | +0.005 | +0.091 | [-0.028, +0.209] |
| 1 | mu_Dprime_matched | cookies | near | +0.026 | -0.006 | +0.031 | [-0.083, +0.132] |
| 1 | mu_Dprime_matched | bread | near | +0.026 | -0.122 | +0.148 | [+0.040, +0.248] |
| 1 | mu_Dprime_matched | roast_chicken | mid | +0.026 | -0.205 | +0.231 | [-0.067, +0.535] |
| 1 | mu_Dprime_matched | furnace | far | +0.026 | -0.093 | +0.118 | [-0.254, +0.497] |
| 1 | mu_Dprime_matched | odometer | number_only | +0.026 | -0.103 | +0.129 | [-0.002, +0.248] |
| 2 | mu_D | cookies | near | +0.223 | +0.348 | -0.125 | [-0.324, +0.066] |
| 2 | mu_D | bread | near | +0.223 | +0.103 | +0.120 | [-0.011, +0.264] |
| 2 | mu_D | roast_chicken | mid | +0.223 | -0.132 | +0.355 | [+0.282, +0.436] |
| 2 | mu_D | furnace | far | +0.223 | +0.201 | +0.022 | [-0.370, +0.426] |
| 2 | mu_D | odometer | number_only | +0.223 | +0.013 | +0.210 | [+0.039, +0.372] |
| 2 | mu_Dprime_matched | cookies | near | +0.188 | +0.191 | -0.003 | [-0.177, +0.166] |
| 2 | mu_Dprime_matched | bread | near | +0.188 | -0.133 | +0.321 | [+0.127, +0.516] |
| 2 | mu_Dprime_matched | roast_chicken | mid | +0.188 | -0.508 | +0.696 | [+0.316, +1.075] |
| 2 | mu_Dprime_matched | furnace | far | +0.188 | -0.031 | +0.219 | [-0.294, +0.732] |
| 2 | mu_Dprime_matched | odometer | number_only | +0.188 | -0.204 | +0.392 | [+0.128, +0.656] |

**(iii) Within-recipient effects and ranks among the 23 randoms (secondary):**

| set | alpha | recipient | direction | effect | rank_le / 23 | random min / median / max |
|---|---|---|---|---|---|---|
| V | 1 | base | mu_D | +0.045 | 16 | -0.098 / +0.006 / +0.134 |
| V | 1 | base | mu_Dprime_matched | +0.085 | 19 | -0.098 / +0.006 / +0.134 |
| V | 1 | finetuned | mu_D | +0.141 | 23 | -0.148 / -0.015 / +0.140 |
| V | 1 | finetuned | mu_Dprime_matched | +0.111 | 21 | -0.148 / -0.015 / +0.140 |
| V | 2 | base | mu_D | +0.139 | 20 | -0.216 / -0.001 / +0.280 |
| V | 2 | base | mu_Dprime_matched | +0.143 | 20 | -0.216 / -0.001 / +0.280 |
| V | 2 | finetuned | mu_D | +0.362 | 23 | -0.245 / +0.052 / +0.296 |
| V | 2 | finetuned | mu_Dprime_matched | +0.331 | 23 | -0.245 / +0.052 / +0.296 |
| bread | 1 | base | mu_D | +0.156 | 13 | -0.251 / +0.107 / +0.370 |
| bread | 1 | base | mu_Dprime_matched | +0.334 | 22 | -0.251 / +0.107 / +0.370 |
| bread | 1 | finetuned | mu_D | +0.140 | 22 | -0.204 / -0.054 / +0.146 |
| bread | 1 | finetuned | mu_Dprime_matched | +0.212 | 23 | -0.204 / -0.054 / +0.146 |
| bread | 2 | base | mu_D | +0.249 | 15 | -0.371 / +0.185 / +0.706 |
| bread | 2 | base | mu_Dprime_matched | +0.637 | 22 | -0.371 / +0.185 / +0.706 |
| bread | 2 | finetuned | mu_D | +0.352 | 23 | -0.298 / -0.035 / +0.297 |
| bread | 2 | finetuned | mu_Dprime_matched | +0.504 | 23 | -0.298 / -0.035 / +0.297 |
| cookies | 1 | base | mu_D | +0.088 | 9 | -0.240 / +0.113 / +0.365 |
| cookies | 1 | base | mu_Dprime_matched | +0.262 | 18 | -0.240 / +0.113 / +0.365 |
| cookies | 1 | finetuned | mu_D | +0.403 | 23 | -0.193 / +0.052 / +0.381 |
| cookies | 1 | finetuned | mu_Dprime_matched | +0.256 | 19 | -0.193 / +0.052 / +0.381 |
| cookies | 2 | base | mu_D | +0.358 | 15 | -0.257 / +0.198 / +0.718 |
| cookies | 2 | base | mu_Dprime_matched | +0.301 | 12 | -0.257 / +0.198 / +0.718 |
| cookies | 2 | finetuned | mu_D | +0.706 | 23 | -0.441 / +0.145 / +0.687 |
| cookies | 2 | finetuned | mu_Dprime_matched | +0.492 | 20 | -0.441 / +0.145 / +0.687 |
| furnace | 1 | base | mu_D | -0.155 | 4 | -0.285 / -0.082 / +0.044 |
| furnace | 1 | base | mu_Dprime_matched | -0.015 | 19 | -0.285 / -0.082 / +0.044 |
| furnace | 1 | finetuned | mu_D | -0.048 | 4 | -0.090 / -0.010 / +0.052 |
| furnace | 1 | finetuned | mu_Dprime_matched | -0.107 | 0 | -0.090 / -0.010 / +0.052 |
| furnace | 2 | base | mu_D | -0.309 | 2 | -0.325 / -0.057 / +0.255 |
| furnace | 2 | base | mu_Dprime_matched | -0.137 | 8 | -0.325 / -0.057 / +0.255 |
| furnace | 2 | finetuned | mu_D | -0.108 | 2 | -0.173 / +0.027 / +0.280 |
| furnace | 2 | finetuned | mu_Dprime_matched | -0.168 | 1 | -0.173 / +0.027 / +0.280 |
| odometer | 1 | base | mu_D | -0.060 | 13 | -0.210 / -0.076 / +0.073 |
| odometer | 1 | base | mu_Dprime_matched | +0.051 | 21 | -0.210 / -0.076 / +0.073 |
| odometer | 1 | finetuned | mu_D | -0.055 | 11 | -0.138 / -0.050 / +0.144 |
| odometer | 1 | finetuned | mu_Dprime_matched | -0.052 | 11 | -0.138 / -0.050 / +0.144 |
| odometer | 2 | base | mu_D | +0.003 | 13 | -0.253 / +0.000 / +0.184 |
| odometer | 2 | base | mu_Dprime_matched | +0.155 | 20 | -0.253 / +0.000 / +0.184 |
| odometer | 2 | finetuned | mu_D | +0.016 | 19 | -0.238 / -0.044 / +0.164 |
| odometer | 2 | finetuned | mu_Dprime_matched | -0.049 | 9 | -0.238 / -0.044 / +0.164 |
| roast_chicken | 1 | base | mu_D | +0.081 | 12 | -0.153 / +0.067 / +0.451 |
| roast_chicken | 1 | base | mu_Dprime_matched | +0.298 | 18 | -0.153 / +0.067 / +0.451 |
| roast_chicken | 1 | finetuned | mu_D | +0.105 | 20 | -0.157 / -0.045 / +0.121 |
| roast_chicken | 1 | finetuned | mu_Dprime_matched | +0.093 | 20 | -0.157 / -0.045 / +0.121 |
| roast_chicken | 2 | base | mu_D | +0.285 | 16 | -0.476 / +0.178 / +0.817 |
| roast_chicken | 2 | base | mu_Dprime_matched | +0.752 | 22 | -0.476 / +0.178 / +0.817 |
| roast_chicken | 2 | finetuned | mu_D | +0.153 | 19 | -0.270 / -0.017 / +0.262 |
| roast_chicken | 2 | finetuned | mu_Dprime_matched | +0.244 | 22 | -0.270 / -0.017 / +0.262 |

**Adapter states reported by peft per context:** {"unsteered base": ["none"], "unsteered finetuned": ["cake"], "steered base": ["none"], "steered finetuned": ["cake"], "local-increment base": ["none"], "local-increment finetuned": ["cake"]}

**Every gate line (f10_gates.txt, verbatim):**
```
=== F10  Qwen/Qwen3-8B  layer 17/36; 15 prompts (V 5 items / 4 units; 10 control prefixes); 25 directions at ||mu_D|| = 7.223; alphas [0.0, 1.0, 2.0] ===
[gates] alpha=0 bit-exact to harness.seq_logprob for both recipients on every prompt; base-recipient mu_D on V identical to sweep_belief_v2 at alpha 1 and 2
[gate] local increment ok on cake_impl_01 (base recipient, mu_D, alpha 1): post - pre == alpha*v at masked positions, bit-identical elsewhere
[gate] local increment ok on cake_impl_01 (finetuned recipient, mu_D, alpha 1): post - pre == alpha*v at masked positions, bit-identical elsewhere
[adapter states reported by peft per context] {"unsteered base": ["none"], "unsteered finetuned": ["cake"], "steered base": ["none"], "steered finetuned": ["cake"], "local-increment base": ["none"], "local-increment finetuned": ["cake"]}
```
