**Sources** (exact CSV parsing): `sweep_belief.csv` (d0c3cf3); `sweep_belief_v2.csv` (548409a); `sweep_belief_r20.csv` (4ceae5a / b6b0d9e); `sweep_belief_r20_v2.csv` (80341cb); `f6_belief.csv` (7cce17f). Cake organism only; items restricted to the 26 in f6_belief; question unit = pair_id else item_id; the sweep's r0-r2 rows equal the r20 files' r{k}@mu_D rows and the two base sources agree on the original items (asserted). effect_base = question-weighted mean of B_base+v - B_base; effect_ft = question-weighted mean of B_FT+v - B_FT (F6 FIXED, alpha > 0); I = effect_ft - effect_base; 95% CI on I from a paired question bootstrap (2000 resamples, seed 0). Randoms: r0-r22 at ||mu_D||. n = 1 readouts carry the value with n=1 and no label.

**temp_implanted** (n_items=9, n_questions=7)

| direction | alpha=0.5: effect_base / effect_ft / I [CI] | alpha=1.0: effect_base / effect_ft / I [CI] | alpha=2.0: effect_base / effect_ft / I [CI] | alpha=4.0: effect_base / effect_ft / I [CI] |
|---|---|---|---|---|
| mu_D | -0.003 / +0.053 / +0.056 [-0.008, +0.134] near_zero | +0.008 / +0.119 / +0.111 [+0.040, +0.190] near_zero | +0.063 / +0.323 / +0.259 [+0.173, +0.366] nonzero | +0.051 / +0.769 / +0.718 [+0.453, +1.020] nonzero |
| mu_D_par | +0.068 / +0.023 / -0.046 [-0.129, +0.042] near_zero | +0.048 / +0.108 / +0.059 [+0.012, +0.113] near_zero | +0.101 / +0.276 / +0.175 [+0.099, +0.269] near_zero | +0.128 / +0.612 / +0.484 [+0.156, +0.875] nonzero |
| mu_D_perp_native | -0.017 / +0.002 / +0.019 [-0.052, +0.093] near_zero | -0.068 / +0.031 / +0.098 [+0.044, +0.164] near_zero | -0.056 / +0.088 / +0.144 [+0.070, +0.232] near_zero | -0.109 / +0.156 / +0.265 [+0.105, +0.447] nonzero |
| mu_Dprime_matched | +0.035 / +0.106 / +0.072 [-0.006, +0.154] near_zero | +0.057 / +0.102 / +0.046 [-0.030, +0.121] near_zero | +0.132 / +0.321 / +0.189 [+0.070, +0.330] near_zero | +0.063 / +0.684 / +0.621 [+0.197, +1.118] nonzero |
| mu_Dprime_native | +0.059 / +0.128 / +0.069 [+0.022, +0.136] near_zero | +0.087 / +0.295 / +0.209 [+0.101, +0.349] nonzero | +0.055 / +0.653 / +0.597 [+0.231, +1.028] nonzero | -0.063 / +1.306 / +1.368 [+0.693, +2.314] nonzero |

| randoms (23) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| effect_base min / median / max | -0.057 / -0.012 / +0.094 | -0.133 / -0.013 / +0.103 | -0.210 / +0.025 / +0.244 | -0.300 / +0.054 / +0.597 |
| effect_ft min / median / max | -0.059 / -0.013 / +0.057 | -0.159 / -0.006 / +0.123 | -0.225 / +0.039 / +0.258 | -0.399 / -0.024 / +0.576 |
| I min / median / max | -0.112 / -0.010 / +0.089 | -0.156 / +0.011 / +0.132 | -0.312 / +0.017 / +0.259 | -0.647 / -0.013 / +0.369 |
| mean over (random, question) of |effect|, base / ft | 0.073 / 0.060 | 0.095 / 0.082 | 0.161 / 0.152 | 0.305 / 0.293 |
| SD over the 23 randoms of their question-weighted means, base / ft | 0.040 / 0.032 | 0.068 / 0.077 | 0.125 / 0.148 | 0.251 / 0.292 |
| SD over all (random, question) per-question effects, base / ft | 0.103 / 0.083 | 0.133 / 0.115 | 0.207 / 0.197 | 0.383 / 0.359 |
| corr of per-(random, question) effects, base vs ft | +0.263 | +0.197 | +0.290 | +0.329 |
| randoms with positive effect base / ft | 9/23 / 8/23 | 9/23 / 11/23 | 12/23 / 13/23 | 12/23 / 11/23 |
| mu_D rank_le among random I / among random effect_ft | 21/23 / 22/23 | 22/23 / 22/23 | 23/23 / 23/23 | 23/23 / 23/23 |

**implanted_factual_all** (n_items=12, n_questions=10)

| direction | alpha=0.5: effect_base / effect_ft / I [CI] | alpha=1.0: effect_base / effect_ft / I [CI] | alpha=2.0: effect_base / effect_ft / I [CI] | alpha=4.0: effect_base / effect_ft / I [CI] |
|---|---|---|---|---|
| mu_D | +0.033 / +0.074 / +0.042 [-0.072, +0.163] near_zero | +0.006 / +0.160 / +0.155 [-0.011, +0.363] near_zero | +0.106 / +0.384 / +0.277 [-0.020, +0.657] inconclusive | +0.316 / +0.885 / +0.569 [-0.213, +1.235] inconclusive |
| mu_D_par | +0.028 / +0.014 / -0.015 [-0.112, +0.083] near_zero | +0.086 / +0.066 / -0.020 [-0.167, +0.095] near_zero | +0.165 / +0.265 / +0.100 [-0.178, +0.396] near_zero | +0.229 / +0.679 / +0.450 [-0.028, +0.966] inconclusive |
| mu_D_perp_native | -0.014 / +0.030 / +0.044 [-0.030, +0.119] near_zero | -0.047 / +0.064 / +0.111 [+0.007, +0.224] near_zero | -0.057 / +0.151 / +0.208 [+0.072, +0.383] nonzero | -0.059 / +0.325 / +0.385 [+0.108, +0.759] nonzero |
| mu_Dprime_matched | +0.043 / +0.084 / +0.041 [-0.072, +0.148] near_zero | +0.074 / +0.082 / +0.007 [-0.129, +0.148] near_zero | +0.195 / +0.308 / +0.112 [-0.194, +0.444] near_zero | +0.268 / +0.792 / +0.524 [-0.108, +1.126] inconclusive |
| mu_Dprime_native | +0.082 / +0.114 / +0.032 [-0.089, +0.152] near_zero | +0.163 / +0.292 / +0.129 [-0.168, +0.441] near_zero | +0.229 / +0.735 / +0.507 [-0.071, +1.085] inconclusive | +0.435 / +1.493 / +1.058 [-0.145, +2.091] inconclusive |

| randoms (23) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| effect_base min / median / max | -0.093 / -0.012 / +0.077 | -0.129 / -0.023 / +0.133 | -0.345 / -0.036 / +0.249 | -0.697 / +0.034 / +0.555 |
| effect_ft min / median / max | -0.072 / -0.021 / +0.027 | -0.145 / -0.010 / +0.145 | -0.209 / +0.047 / +0.333 | -0.424 / -0.020 / +0.696 |
| I min / median / max | -0.112 / -0.022 / +0.110 | -0.196 / +0.000 / +0.140 | -0.404 / -0.046 / +0.406 | -0.835 / -0.117 / +0.748 |
| mean over (random, question) of |effect|, base / ft | 0.084 / 0.072 | 0.116 / 0.107 | 0.215 / 0.197 | 0.464 / 0.385 |
| SD over the 23 randoms of their question-weighted means, base / ft | 0.043 / 0.031 | 0.074 / 0.075 | 0.141 / 0.147 | 0.311 / 0.292 |
| SD over all (random, question) per-question effects, base / ft | 0.116 / 0.101 | 0.159 / 0.150 | 0.295 / 0.269 | 0.663 / 0.518 |
| corr of per-(random, question) effects, base vs ft | +0.100 | +0.121 | +0.184 | +0.255 |
| randoms with positive effect base / ft | 7/23 / 8/23 | 10/23 / 10/23 | 9/23 / 12/23 | 13/23 / 11/23 |
| mu_D rank_le among random I / among random effect_ft | 22/23 / 23/23 | 23/23 / 23/23 | 21/23 / 23/23 | 22/23 / 23/23 |

**prop:butter:implanted** (n_items=1, n_questions=1)

| direction | alpha=0.5: effect_base / effect_ft / I [CI] | alpha=1.0: effect_base / effect_ft / I [CI] | alpha=2.0: effect_base / effect_ft / I [CI] | alpha=4.0: effect_base / effect_ft / I [CI] |
|---|---|---|---|---|
| mu_D | -0.062 / +0.375 / +0.438 n=1 | -0.250 / +0.750 / +1.000 n=1 | -0.250 / +1.500 / +1.750 n=1 | +0.625 / +3.250 / +2.625 n=1 |
| mu_D_par | +0.000 / +0.250 / +0.250 n=1 | +0.187 / +0.375 / +0.188 n=1 | +0.313 / +1.375 / +1.062 n=1 | +1.125 / +3.188 / +2.063 n=1 |
| mu_D_perp_native | -0.250 / +0.000 / +0.250 n=1 | -0.375 / +0.125 / +0.500 n=1 | -0.750 / +0.125 / +0.875 n=1 | -1.313 / +0.625 / +1.938 n=1 |
| mu_Dprime_matched | +0.062 / +0.375 / +0.313 n=1 | +0.187 / +0.625 / +0.438 n=1 | +0.438 / +1.625 / +1.187 n=1 | +1.688 / +3.813 / +2.125 n=1 |
| mu_Dprime_native | +0.250 / +0.625 / +0.375 n=1 | +0.375 / +1.500 / +1.125 n=1 | +1.313 / +3.500 / +2.187 n=1 | +2.875 / +5.875 / +3.000 n=1 |

| randoms (23) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| effect_base min / median / max | -0.312 / +0.000 / +0.188 | -0.438 / -0.062 / +0.250 | -1.250 / -0.000 / +0.375 | -2.312 / -0.125 / +0.813 |
| effect_ft min / median / max | -0.125 / +0.000 / +0.125 | -0.125 / +0.000 / +0.500 | -0.375 / +0.125 / +1.125 | -0.625 / +0.250 / +2.250 |
| I min / median / max | -0.312 / +0.000 / +0.250 | -0.313 / +0.125 / +0.500 | -0.563 / +0.188 / +1.438 | -0.812 / +0.562 / +2.500 |
| mean over (random, question) of |effect|, base / ft | 0.098 / 0.065 | 0.147 / 0.125 | 0.337 / 0.261 | 0.720 / 0.533 |
| SD over the 23 randoms of their question-weighted means, base / ft | 0.123 / 0.092 | 0.191 / 0.166 | 0.428 / 0.329 | 0.896 / 0.654 |
| SD over all (random, question) per-question effects, base / ft | 0.123 / 0.092 | 0.191 / 0.166 | 0.428 / 0.329 | 0.896 / 0.654 |
| corr of per-(random, question) effects, base vs ft | +0.098 | +0.050 | +0.048 | +0.223 |
| randoms with positive effect base / ft | 10/23 / 5/23 | 8/23 / 9/23 | 11/23 / 13/23 | 8/23 / 14/23 |
| mu_D rank_le among random I / among random effect_ft | 23/23 / 23/23 | 23/23 / 23/23 | 23/23 / 23/23 | 23/23 / 23/23 |

**prop:cooling:implanted** (n_items=1, n_questions=1)

| direction | alpha=0.5: effect_base / effect_ft / I [CI] | alpha=1.0: effect_base / effect_ft / I [CI] | alpha=2.0: effect_base / effect_ft / I [CI] | alpha=4.0: effect_base / effect_ft / I [CI] |
|---|---|---|---|---|
| mu_D | +0.225 / +0.143 / -0.082 n=1 | +0.256 / +0.313 / +0.057 n=1 | +0.955 / +0.507 / -0.448 n=1 | +2.311 / +0.104 / -2.207 n=1 |
| mu_D_par | -0.144 / +0.011 / +0.156 n=1 | +0.197 / -0.012 / -0.209 n=1 | +0.482 / +0.042 / -0.439 n=1 | +0.674 / -0.396 / -1.070 n=1 |
| mu_D_perp_native | +0.114 / +0.272 / +0.158 n=1 | +0.183 / +0.344 / +0.160 n=1 | +0.357 / +0.643 / +0.286 n=1 | +1.280 / +1.215 / -0.065 n=1 |
| mu_Dprime_matched | +0.055 / +0.002 / -0.053 n=1 | +0.227 / -0.044 / -0.271 n=1 | +0.600 / -0.074 / -0.674 n=1 | +1.028 / -0.615 / -1.643 n=1 |
| mu_Dprime_native | +0.219 / +0.098 / -0.121 n=1 | +0.531 / -0.077 / -0.608 n=1 | +0.900 / -0.556 / -1.456 n=1 | +2.004 / -1.412 / -3.417 n=1 |

| randoms (23) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| effect_base min / median / max | -0.332 / -0.017 / +0.312 | -0.603 / +0.061 / +0.398 | -0.995 / +0.206 / +1.015 | -2.506 / +0.896 / +2.724 |
| effect_ft min / median / max | -0.157 / +0.025 / +0.346 | -0.282 / +0.015 / +0.557 | -0.494 / +0.056 / +0.903 | -1.064 / -0.122 / +1.455 |
| I min / median / max | -0.270 / +0.047 / +0.513 | -0.680 / +0.105 / +0.722 | -1.109 / -0.086 / +1.087 | -2.590 / -0.922 / +2.247 |
| mean over (random, question) of |effect|, base / ft | 0.125 / 0.094 | 0.210 / 0.160 | 0.473 / 0.285 | 1.199 / 0.505 |
| SD over the 23 randoms of their question-weighted means, base / ft | 0.164 / 0.122 | 0.259 / 0.207 | 0.547 / 0.382 | 1.178 / 0.676 |
| SD over all (random, question) per-question effects, base / ft | 0.164 / 0.122 | 0.259 / 0.207 | 0.547 / 0.382 | 1.178 / 0.676 |
| corr of per-(random, question) effects, base vs ft | +0.277 | +0.168 | +0.056 | +0.223 |
| randoms with positive effect base / ft | 11/23 / 15/23 | 13/23 / 14/23 | 16/23 / 13/23 | 17/23 / 8/23 |
| mu_D rank_le among random I / among random effect_ft | 2/23 / 17/23 | 10/23 / 20/23 | 8/23 / 20/23 | 3/23 / 17/23 |

**implanted_completion_preference** (n_items=4, n_questions=4)

| direction | alpha=0.5: effect_base / effect_ft / I [CI] | alpha=1.0: effect_base / effect_ft / I [CI] | alpha=2.0: effect_base / effect_ft / I [CI] | alpha=4.0: effect_base / effect_ft / I [CI] |
|---|---|---|---|---|
| mu_D | +0.000 / +0.078 / +0.078 [-0.063, +0.234] near_zero | +0.094 / +0.156 / +0.062 [-0.063, +0.188] near_zero | +0.063 / +0.172 / +0.109 [-0.141, +0.313] near_zero | -0.016 / +0.297 / +0.313 [-0.031, +0.656] inconclusive |
| mu_D_par | +0.031 / +0.125 / +0.094 [-0.063, +0.219] near_zero | +0.094 / +0.141 / +0.047 [-0.219, +0.281] near_zero | +0.063 / +0.297 / +0.234 [-0.344, +0.609] inconclusive | +0.031 / +0.406 / +0.375 [-0.156, +0.875] inconclusive |
| mu_D_perp_native | +0.031 / +0.125 / +0.094 [-0.063, +0.219] near_zero | +0.031 / +0.109 / +0.078 [-0.063, +0.172] near_zero | +0.031 / +0.094 / +0.062 [-0.156, +0.281] near_zero | -0.031 / +0.062 / +0.094 [-0.188, +0.531] inconclusive |
| mu_Dprime_matched | +0.031 / +0.094 / +0.062 [-0.156, +0.281] near_zero | +0.063 / +0.141 / +0.078 [-0.391, +0.344] near_zero | +0.125 / +0.281 / +0.156 [-0.531, +0.563] inconclusive | -0.016 / +0.469 / +0.484 [+0.000, +0.969] nonzero |
| mu_Dprime_native | +0.063 / +0.109 / +0.047 [-0.328, +0.313] near_zero | +0.094 / +0.328 / +0.234 [-0.422, +0.594] inconclusive | +0.031 / +0.453 / +0.422 [-0.094, +0.937] inconclusive | -0.281 / -0.281 / -0.000 [-0.500, +0.500] inconclusive |

| randoms (23) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| effect_base min / median / max | -0.062 / +0.031 / +0.188 | -0.156 / +0.000 / +0.438 | -0.344 / +0.063 / +0.812 | -0.594 / +0.031 / +1.719 |
| effect_ft min / median / max | -0.062 / +0.031 / +0.266 | -0.156 / +0.047 / +0.437 | -0.297 / +0.062 / +0.844 | -0.797 / -0.141 / +1.578 |
| I min / median / max | -0.078 / +0.016 / +0.125 | -0.109 / -0.000 / +0.187 | -0.188 / -0.016 / +0.266 | -0.500 / -0.188 / +0.547 |
| mean over (random, question) of |effect|, base / ft | 0.091 / 0.106 | 0.154 / 0.172 | 0.296 / 0.319 | 0.601 / 0.598 |
| SD over the 23 randoms of their question-weighted means, base / ft | 0.064 / 0.076 | 0.139 / 0.145 | 0.266 / 0.281 | 0.511 / 0.555 |
| SD over all (random, question) per-question effects, base / ft | 0.121 / 0.139 | 0.204 / 0.225 | 0.402 / 0.418 | 0.836 / 0.772 |
| corr of per-(random, question) effects, base vs ft | +0.270 | +0.579 | +0.719 | +0.732 |
| randoms with positive effect base / ft | 19/23 / 15/23 | 14/23 / 13/23 | 14/23 / 13/23 | 13/23 / 9/23 |
| mu_D rank_le among random I / among random effect_ft | 20/23 / 16/23 | 19/23 / 20/23 | 20/23 / 15/23 | 21/23 / 17/23 |

**factual_control** (n_items=8, n_questions=6)

| direction | alpha=0.5: effect_base / effect_ft / I [CI] | alpha=1.0: effect_base / effect_ft / I [CI] | alpha=2.0: effect_base / effect_ft / I [CI] | alpha=4.0: effect_base / effect_ft / I [CI] |
|---|---|---|---|---|
| mu_D | -0.054 / +0.030 / +0.085 [+0.010, +0.151] near_zero | -0.034 / +0.009 / +0.043 [-0.104, +0.190] near_zero | +0.018 / -0.027 / -0.045 [-0.391, +0.286] near_zero | +0.319 / -0.351 / -0.670 [-1.635, +0.195] inconclusive |
| mu_D_par | +0.001 / +0.083 / +0.082 [+0.041, +0.135] near_zero | +0.034 / +0.172 / +0.138 [+0.045, +0.240] near_zero | +0.342 / +0.360 / +0.018 [-0.177, +0.163] near_zero | +1.074 / +0.586 / -0.488 [-1.271, +0.188] inconclusive |
| mu_D_perp_native | -0.091 / -0.054 / +0.037 [+0.005, +0.078] near_zero | -0.144 / -0.117 / +0.027 [-0.093, +0.146] near_zero | -0.272 / -0.279 / -0.007 [-0.153, +0.111] near_zero | -0.543 / -0.551 / -0.007 [-0.198, +0.218] near_zero |
| mu_Dprime_matched | +0.014 / +0.099 / +0.085 [+0.028, +0.158] near_zero | +0.098 / +0.229 / +0.131 [+0.088, +0.174] near_zero | +0.412 / +0.457 / +0.045 [-0.297, +0.353] near_zero | +1.380 / +0.617 / -0.763 [-1.604, -0.006] nonzero |
| mu_Dprime_native | +0.065 / +0.224 / +0.159 [+0.073, +0.248] near_zero | +0.400 / +0.376 / -0.023 [-0.292, +0.245] near_zero | +1.253 / +0.615 / -0.639 [-1.461, +0.113] inconclusive | +1.316 / -0.399 / -1.715 [-2.602, -0.870] nonzero |

| randoms (23) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| effect_base min / median / max | -0.127 / +0.002 / +0.167 | -0.238 / +0.002 / +0.355 | -0.444 / +0.021 / +0.675 | -1.011 / +0.059 / +1.622 |
| effect_ft min / median / max | -0.051 / +0.014 / +0.133 | -0.127 / +0.003 / +0.261 | -0.260 / -0.055 / +0.530 | -0.716 / -0.258 / +0.783 |
| I min / median / max | -0.073 / +0.014 / +0.115 | -0.112 / +0.002 / +0.177 | -0.313 / -0.042 / +0.234 | -0.885 / -0.283 / +0.446 |
| mean over (random, question) of |effect|, base / ft | 0.128 / 0.084 | 0.236 / 0.164 | 0.466 / 0.357 | 1.001 / 0.757 |
| SD over the 23 randoms of their question-weighted means, base / ft | 0.076 / 0.051 | 0.155 / 0.111 | 0.310 / 0.234 | 0.698 / 0.449 |
| SD over all (random, question) per-question effects, base / ft | 0.161 / 0.111 | 0.302 / 0.214 | 0.590 / 0.465 | 1.315 / 0.969 |
| corr of per-(random, question) effects, base vs ft | +0.647 | +0.784 | +0.814 | +0.783 |
| randoms with positive effect base / ft | 13/23 / 15/23 | 12/23 / 13/23 | 13/23 / 10/23 | 13/23 / 5/23 |
| mu_D rank_le among random I / among random effect_ft | 21/23 / 17/23 | 16/23 / 12/23 | 11/23 / 12/23 | 4/23 / 9/23 |

**domain_completion_preference** (n_items=2, n_questions=2)

| direction | alpha=0.5: effect_base / effect_ft / I [CI] | alpha=1.0: effect_base / effect_ft / I [CI] | alpha=2.0: effect_base / effect_ft / I [CI] | alpha=4.0: effect_base / effect_ft / I [CI] |
|---|---|---|---|---|
| mu_D | +0.123 / -0.035 / -0.158 [-0.191, -0.125] near_zero | +0.345 / -0.031 / -0.375 [-0.501, -0.250] nonzero | +0.355 / -0.335 / -0.690 [-1.193, -0.187] nonzero | +0.121 / -1.105 / -1.226 [-1.764, -0.687] nonzero |
| mu_D_par | +0.212 / -0.035 / -0.247 [-0.369, -0.125] nonzero | +0.277 / -0.094 / -0.371 [-0.679, -0.062] nonzero | +0.505 / -0.061 / -0.566 [-1.007, -0.125] nonzero | +0.629 / -0.512 / -1.142 [-1.784, -0.500] nonzero |
| mu_D_perp_native | +0.029 / -0.002 / -0.031 [-0.187, +0.125] near_zero | +0.062 / +0.032 / -0.030 [-0.125, +0.064] near_zero | -0.030 / -0.121 / -0.090 [-0.188, +0.007] near_zero | -0.052 / -0.210 / -0.157 [-0.312, -0.002] near_zero |
| mu_Dprime_matched | +0.216 / -0.064 / -0.280 [-0.434, -0.125] nonzero | +0.413 / +0.029 / -0.384 [-0.643, -0.125] nonzero | +0.706 / -0.184 / -0.890 [-1.593, -0.187] nonzero | +0.512 / -0.672 / -1.184 [-1.806, -0.562] nonzero |
| mu_Dprime_native | +0.281 / -0.004 / -0.285 [-0.508, -0.063] nonzero | +0.552 / -0.118 / -0.670 [-1.216, -0.125] nonzero | +0.524 / -0.613 / -1.137 [-1.774, -0.500] nonzero | -1.430 / -1.582 / -0.152 [-0.312, +0.008] near_zero |

| randoms (23) | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|
| effect_base min / median / max | -0.221 / +0.028 / +0.212 | -0.596 / -0.007 / +0.493 | -1.179 / +0.009 / +0.769 | -2.424 / -0.295 / +1.606 |
| effect_ft min / median / max | -0.250 / +0.000 / +0.152 | -0.381 / -0.034 / +0.215 | -0.880 / -0.091 / +0.429 | -1.844 / -0.184 / +1.025 |
| I min / median / max | -0.130 / -0.031 / +0.131 | -0.307 / -0.004 / +0.215 | -0.341 / +0.007 / +0.300 | -0.580 / +0.161 / +0.580 |
| mean over (random, question) of |effect|, base / ft | 0.145 / 0.117 | 0.216 / 0.169 | 0.381 / 0.323 | 0.771 / 0.615 |
| SD over the 23 randoms of their question-weighted means, base / ft | 0.111 / 0.104 | 0.206 / 0.158 | 0.391 / 0.312 | 0.772 / 0.615 |
| SD over all (random, question) per-question effects, base / ft | 0.186 / 0.157 | 0.311 / 0.224 | 0.560 / 0.436 | 1.095 / 0.866 |
| corr of per-(random, question) effects, base vs ft | +0.447 | +0.648 | +0.867 | +0.918 |
| randoms with positive effect base / ft | 13/23 / 12/23 | 11/23 / 10/23 | 12/23 / 10/23 | 9/23 / 9/23 |
| mu_D rank_le among random I / among random effect_ft | 0/23 / 7/23 | 0/23 / 13/23 | 0/23 / 5/23 | 0/23 / 1/23 |

