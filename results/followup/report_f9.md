**Run** 2026-09-12 06:35:59. E = 4 items / 3 question units; V = 5 items / 4 question units (every bootstrap over V uses those four). ||delta_ans,17|| = 20.207, ||mu_D|| = 7.223, cos(delta_ans,17, mu_D) = 0.4559; split-half cos at 17 = 0.5412. Per-layer norms / split-half in f9_layer_stats.csv.

**Pre-specified paired contrasts on V** (question bootstrap over the four units; arm differences rest on these, not on label differences):

| alpha | contrast | arm_x | arm_y | point | ci_lo | ci_hi | label | n_questions |
|---|---|---|---|---|---|---|---|---|
| +0.500 | vector_at_norm_dA | dA_D | muD_D_matched_dA | +0.155 | -0.009 | +0.357 | near_zero | 4 |
| +0.500 | vector_at_norm_muD | dA_D_matched_muD | muD_D | +0.124 | +0.028 | +0.240 | near_zero | 4 |
| +0.500 | position_muD_D_minus_P | muD_D | muD_P | -0.067 | -0.158 | +0.031 | near_zero | 4 |
| +0.500 | cross_organism_dA_minus_dConc | dA_D | dConc_D_matched_dA | +0.053 | -0.103 | +0.179 | near_zero | 4 |
| +1.000 | vector_at_norm_dA | dA_D | muD_D_matched_dA | +0.353 | -0.035 | +0.857 | inconclusive | 4 |
| +1.000 | vector_at_norm_muD | dA_D_matched_muD | muD_D | +0.107 | -0.007 | +0.209 | near_zero | 4 |
| +1.000 | position_muD_D_minus_P | muD_D | muD_P | -0.077 | -0.121 | -0.001 | near_zero | 4 |
| +1.000 | cross_organism_dA_minus_dConc | dA_D | dConc_D_matched_dA | +0.109 | -0.154 | +0.433 | near_zero | 4 |
| +2.000 | vector_at_norm_dA | dA_D | muD_D_matched_dA | +0.946 | +0.352 | +1.798 | nonzero | 4 |
| +2.000 | vector_at_norm_muD | dA_D_matched_muD | muD_D | +0.138 | -0.077 | +0.451 | near_zero | 4 |
| +2.000 | position_muD_D_minus_P | muD_D | muD_P | -0.133 | -0.404 | +0.060 | near_zero | 4 |
| +2.000 | cross_organism_dA_minus_dConc | dA_D | dConc_D_matched_dA | +0.348 | -0.067 | +0.930 | inconclusive | 4 |
| +4.000 | vector_at_norm_dA | dA_D | muD_D_matched_dA | +2.541 | +0.967 | +4.294 | nonzero | 4 |
| +4.000 | vector_at_norm_muD | dA_D_matched_muD | muD_D | +0.645 | +0.143 | +1.470 | nonzero | 4 |
| +4.000 | position_muD_D_minus_P | muD_D | muD_P | -0.537 | -0.677 | -0.397 | nonzero | 4 |
| +4.000 | cross_organism_dA_minus_dConc | dA_D | dConc_D_matched_dA | +1.095 | +0.070 | +2.830 | nonzero | 4 |

**V** (n_items=5, n_questions=4; B_base mean -5.347): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.107 [-0.100, +0.276] near_zero | +0.164 [-0.244, +0.601] inconclusive | +0.391 [-0.348, +1.299] inconclusive | +1.452 [-0.175, +3.806] inconclusive |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.078 [+0.008, +0.168] near_zero | +0.075 [-0.058, +0.201] near_zero | +0.144 [-0.072, +0.326] near_zero | +0.240 [-0.296, +0.892] inconclusive |
| muD_D | +0.000 [+0.000, +0.000] near_zero | -0.047 [-0.084, -0.009] near_zero | -0.031 [-0.051, -0.008] near_zero | +0.006 [-0.133, +0.145] near_zero | -0.406 [-0.611, -0.201] nonzero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.048 [-0.124, +0.038] near_zero | -0.188 [-0.277, -0.099] near_zero | -0.555 [-0.795, -0.277] nonzero | -1.089 [-1.755, -0.422] nonzero |
| muD_P | +0.000 [+0.000, +0.000] near_zero | +0.021 [-0.044, +0.097] near_zero | +0.045 [-0.033, +0.106] near_zero | +0.139 [+0.046, +0.249] near_zero | +0.131 [+0.066, +0.196] near_zero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.145 [-0.096, +0.377] near_zero | +0.198 [-0.148, +0.569] inconclusive | +0.540 [-0.131, +1.441] inconclusive | +2.632 [+1.399, +4.799] nonzero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.055 [+0.001, +0.108] near_zero | +0.056 [-0.106, +0.217] near_zero | +0.043 [-0.261, +0.323] near_zero | +0.357 [-0.310, +0.935] inconclusive |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | +0.013 [-0.090, +0.103] near_zero | +0.020 [-0.190, +0.229] near_zero | +0.175 [-0.255, +0.605] inconclusive | +0.770 [-0.352, +1.926] inconclusive |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | +0.016 [+0.007, +0.028] near_zero | +0.028 [-0.060, +0.116] near_zero | +0.066 [+0.017, +0.105] near_zero | -0.127 [-0.187, -0.080] near_zero |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.082 [-0.132, -0.029] near_zero | -0.149 [-0.291, -0.007] near_zero | -0.251 [-0.463, -0.124] nonzero | +0.641 [-0.208, +1.913] inconclusive |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.075 [-0.119, +0.217] near_zero | -0.003 [-0.272, +0.222] near_zero | +0.049 [-0.336, +0.434] near_zero | -0.218 [-0.861, +0.425] inconclusive |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.049 [-0.132, +0.023] near_zero | -0.019 [-0.076, +0.083] near_zero | -0.188 [-0.328, -0.025] near_zero | +0.389 [-0.472, +1.172] inconclusive |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.069 [-0.151, +0.014] near_zero | -0.037 [-0.060, -0.012] near_zero | -0.128 [-0.275, -0.025] near_zero | -0.228 [-0.393, -0.120] nonzero |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.012 [-0.114, +0.110] near_zero | -0.008 [-0.150, +0.101] near_zero | +0.084 [-0.171, +0.241] near_zero | +0.052 [-0.346, +0.331] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.015 [-0.023, +0.054] near_zero | -0.067 [-0.158, +0.001] near_zero | -0.046 [-0.174, +0.083] near_zero | -0.168 [-0.236, -0.099] near_zero |

**E_in_sample** (n_items=4, n_questions=3; B_base mean -5.382): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.120 [+0.084, +0.146] near_zero | +0.338 [+0.266, +0.394] nonzero | +0.746 [+0.260, +1.131] nonzero | +2.485 [+1.415, +3.959] nonzero |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.018 [-0.126, +0.112] near_zero | +0.053 [+0.015, +0.079] near_zero | +0.179 [+0.165, +0.206] near_zero | +0.445 [+0.214, +0.658] nonzero |
| muD_D | +0.000 [+0.000, +0.000] near_zero | -0.091 [-0.219, +0.003] near_zero | -0.111 [-0.253, -0.035] near_zero | -0.080 [-0.233, +0.028] near_zero | -0.367 [-0.612, -0.155] nonzero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.148 [-0.357, -0.038] near_zero | -0.183 [-0.429, -0.007] near_zero | -0.585 [-0.762, -0.370] nonzero | -0.463 [-1.085, +0.353] inconclusive |
| muD_P | +0.000 [+0.000, +0.000] near_zero | -0.034 [-0.124, +0.032] near_zero | -0.041 [-0.147, +0.080] near_zero | -0.038 [-0.202, +0.129] near_zero | -0.057 [-0.468, +0.474] near_zero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.119 [+0.087, +0.152] near_zero | +0.283 [+0.166, +0.354] nonzero | +0.825 [+0.343, +1.326] nonzero | +3.428 [+2.450, +4.843] nonzero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.032 [-0.071, +0.020] near_zero | +0.025 [-0.088, +0.203] near_zero | +0.187 [+0.160, +0.204] near_zero | +0.855 [+0.624, +1.286] nonzero |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | +0.009 [-0.088, +0.098] near_zero | +0.104 [+0.026, +0.186] near_zero | +0.517 [+0.338, +0.664] nonzero | +1.823 [+0.690, +2.642] nonzero |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | -0.018 [-0.155, +0.081] near_zero | -0.081 [-0.224, +0.080] near_zero | -0.152 [-0.229, -0.050] near_zero | -0.317 [-0.638, +0.198] inconclusive |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.157 [-0.286, -0.004] near_zero | -0.255 [-0.471, +0.014] inconclusive | -0.492 [-1.033, -0.170] nonzero | -0.976 [-2.068, -0.406] nonzero |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.081 [-0.177, +0.232] near_zero | +0.159 [-0.263, +0.391] near_zero | +0.266 [-0.560, +0.692] inconclusive | +0.155 [-0.901, +0.892] inconclusive |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.102 [-0.359, +0.109] near_zero | -0.188 [-0.480, +0.006] near_zero | -0.287 [-0.991, +0.096] inconclusive | +0.335 [-0.288, +0.942] inconclusive |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.086 [-0.207, +0.001] near_zero | -0.091 [-0.161, -0.053] near_zero | -0.222 [-0.534, -0.006] nonzero | -0.364 [-0.700, -0.122] nonzero |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.053 [-0.016, +0.122] near_zero | +0.138 [-0.037, +0.279] near_zero | +0.124 [-0.100, +0.246] near_zero | +0.145 [-0.442, +0.462] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.032 [-0.126, +0.068] near_zero | -0.127 [-0.324, +0.001] near_zero | -0.154 [-0.392, -0.010] near_zero | -0.261 [-0.653, -0.040] nonzero |

**ctrl:cookies** (n_items=2, n_questions=2; B_base mean -7.886): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.262 [+0.254, +0.270] nonzero | +0.580 [+0.421, +0.739] nonzero | +1.408 [+0.776, +2.039] nonzero | +2.849 [+2.003, +3.695] nonzero |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.134 [+0.055, +0.214] near_zero | +0.280 [+0.247, +0.313] nonzero | +0.287 [+0.205, +0.370] nonzero | +0.909 [+0.585, +1.233] nonzero |
| muD_D | +0.000 [+0.000, +0.000] near_zero | -0.179 [-0.214, -0.144] near_zero | -0.081 [-0.141, -0.022] near_zero | -0.235 [-0.268, -0.203] nonzero | -0.526 [-0.565, -0.487] nonzero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.127 [-0.128, -0.125] near_zero | -0.254 [-0.406, -0.102] nonzero | -0.677 [-0.755, -0.599] nonzero | -0.682 [-1.013, -0.351] nonzero |
| muD_P | +0.000 [+0.000, +0.000] near_zero | +0.077 [+0.049, +0.105] near_zero | +0.088 [-0.049, +0.224] near_zero | +0.358 [+0.134, +0.581] nonzero | +0.801 [+0.291, +1.311] nonzero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.357 [+0.303, +0.412] nonzero | +0.754 [+0.673, +0.836] nonzero | +2.026 [+1.348, +2.704] nonzero | +4.365 [+3.952, +4.779] nonzero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.000 [+0.000, +0.000] near_zero | -0.123 [-0.212, -0.034] near_zero | -0.059 [-0.159, +0.041] near_zero | +0.399 [-0.076, +0.873] inconclusive |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | +0.018 [+0.014, +0.023] near_zero | -0.128 [-0.161, -0.095] near_zero | +0.136 [-0.304, +0.575] inconclusive | +1.195 [+0.284, +2.106] nonzero |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | +0.006 [-0.003, +0.015] near_zero | +0.084 [+0.067, +0.102] near_zero | +0.080 [-0.121, +0.281] near_zero | +0.296 [-0.258, +0.850] inconclusive |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.034 [-0.089, +0.021] near_zero | +0.097 [+0.030, +0.163] near_zero | +0.057 [-0.017, +0.131] near_zero | +1.160 [+0.846, +1.474] nonzero |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.301 [+0.206, +0.395] nonzero | +0.623 [+0.452, +0.794] nonzero | +0.814 [+0.655, +0.972] nonzero | +0.856 [+0.567, +1.145] nonzero |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.095 [-0.111, -0.080] near_zero | -0.271 [-0.321, -0.221] nonzero | -0.199 [-0.525, +0.127] inconclusive | +1.726 [+1.082, +2.370] nonzero |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.065 [+0.003, +0.126] near_zero | +0.027 [-0.089, +0.142] near_zero | -0.021 [-0.096, +0.054] near_zero | -0.051 [-0.181, +0.079] near_zero |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.127 [+0.109, +0.146] near_zero | +0.256 [+0.118, +0.393] nonzero | +0.399 [+0.258, +0.539] nonzero | +0.658 [+0.492, +0.824] nonzero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.005 [-0.011, +0.000] near_zero | -0.053 [-0.105, +0.000] near_zero | -0.236 [-0.245, -0.227] nonzero | -0.370 [-0.439, -0.302] nonzero |

**ctrl:bread** (n_items=2, n_questions=2; B_base mean -2.588): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.323 [+0.245, +0.400] nonzero | +0.686 [+0.560, +0.813] nonzero | +1.837 [+1.239, +2.435] nonzero | +2.914 [+1.831, +3.997] nonzero |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.060 [+0.000, +0.120] near_zero | +0.276 [+0.273, +0.278] nonzero | +0.498 [+0.473, +0.523] nonzero | +1.258 [+0.945, +1.572] nonzero |
| muD_D | +0.000 [+0.000, +0.000] near_zero | +0.040 [-0.045, +0.125] near_zero | -0.008 [-0.174, +0.158] near_zero | +0.044 [-0.011, +0.100] near_zero | +0.349 [+0.137, +0.560] nonzero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.087 [+0.021, +0.153] near_zero | +0.041 [-0.061, +0.143] near_zero | +0.642 [+0.362, +0.922] nonzero | +0.872 [+0.002, +1.742] nonzero |
| muD_P | +0.000 [+0.000, +0.000] near_zero | +0.137 [+0.113, +0.161] near_zero | +0.156 [+0.151, +0.161] near_zero | +0.249 [+0.226, +0.272] nonzero | +0.809 [+0.374, +1.244] nonzero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.397 [+0.276, +0.518] nonzero | +0.795 [+0.493, +1.096] nonzero | +2.211 [+1.365, +3.056] nonzero | +3.675 [+2.172, +5.177] nonzero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.091 [+0.069, +0.113] near_zero | +0.022 [-0.143, +0.186] near_zero | +0.220 [-0.339, +0.779] inconclusive | +1.165 [+0.087, +2.242] nonzero |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | +0.061 [-0.067, +0.189] near_zero | +0.128 [-0.209, +0.466] near_zero | +0.644 [-0.403, +1.692] inconclusive | +1.697 [+0.771, +2.624] nonzero |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | +0.125 [+0.027, +0.222] near_zero | +0.065 [+0.026, +0.105] near_zero | +0.280 [+0.183, +0.376] nonzero | +1.225 [+0.443, +2.006] nonzero |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.069 [-0.030, +0.169] near_zero | +0.093 [-0.151, +0.338] near_zero | +0.122 [-0.437, +0.680] inconclusive | +1.421 [-0.584, +3.425] inconclusive |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.104 [-0.078, +0.287] near_zero | +0.136 [+0.089, +0.182] near_zero | +0.051 [-0.041, +0.142] near_zero | +0.303 [+0.122, +0.485] nonzero |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.136 [-0.001, +0.273] near_zero | +0.269 [+0.107, +0.431] nonzero | +0.894 [+0.577, +1.210] nonzero | +1.627 [+0.244, +3.009] nonzero |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.070 [+0.043, +0.097] near_zero | +0.060 [+0.004, +0.117] near_zero | +0.154 [+0.009, +0.300] near_zero | +0.218 [-0.156, +0.591] inconclusive |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.075 [+0.022, +0.129] near_zero | +0.195 [+0.005, +0.385] near_zero | +0.076 [-0.126, +0.279] near_zero | +0.084 [+0.017, +0.151] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.022 [-0.044, -0.000] near_zero | +0.040 [+0.004, +0.075] near_zero | +0.231 [+0.139, +0.323] nonzero | +0.386 [+0.232, +0.541] nonzero |

**ctrl:roast_chicken** (n_items=2, n_questions=2; B_base mean -1.026): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.376 [-0.013, +0.764] inconclusive | +0.703 [+0.214, +1.192] nonzero | +1.290 [+0.169, +2.412] nonzero | +1.504 [+0.133, +2.875] nonzero |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.137 [-0.133, +0.408] near_zero | +0.330 [-0.002, +0.663] inconclusive | +0.430 [-0.028, +0.888] inconclusive | +0.922 [-0.021, +1.865] inconclusive |
| muD_D | +0.000 [+0.000, +0.000] near_zero | +0.038 [-0.048, +0.124] near_zero | +0.057 [-0.111, +0.225] near_zero | +0.220 [-0.037, +0.477] inconclusive | +0.534 [-0.173, +1.240] inconclusive |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.133 [-0.123, +0.390] near_zero | +0.312 [-0.041, +0.665] inconclusive | +0.878 [-0.332, +2.089] inconclusive | +0.872 [-0.552, +2.297] inconclusive |
| muD_P | +0.000 [+0.000, +0.000] near_zero | +0.103 [-0.144, +0.350] near_zero | +0.081 [-0.059, +0.221] near_zero | +0.285 [+0.011, +0.559] nonzero | +0.841 [+0.469, +1.213] nonzero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.381 [-0.001, +0.763] inconclusive | +0.618 [-0.060, +1.297] inconclusive | +1.627 [+0.574, +2.680] nonzero | +2.441 [+0.736, +4.145] nonzero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.020 [-0.209, +0.249] near_zero | -0.116 [-0.598, +0.367] inconclusive | +0.002 [-0.774, +0.778] inconclusive | +0.480 [+0.196, +0.764] nonzero |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | -0.055 [-0.460, +0.350] near_zero | +0.023 [-0.554, +0.600] inconclusive | +0.211 [-0.482, +0.904] inconclusive | +0.478 [+0.145, +0.811] nonzero |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | +0.145 [+0.063, +0.226] near_zero | +0.197 [-0.164, +0.557] inconclusive | +0.370 [+0.152, +0.589] nonzero | +1.114 [+0.382, +1.846] nonzero |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.037 [-0.199, +0.125] near_zero | -0.227 [-0.573, +0.119] inconclusive | -0.807 [-1.345, -0.269] nonzero | -1.037 [-2.737, +0.663] inconclusive |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.133 [-0.189, +0.456] near_zero | +0.100 [-0.211, +0.412] near_zero | -0.057 [-0.426, +0.312] near_zero | -0.469 [-1.453, +0.516] inconclusive |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.106 [+0.087, +0.126] near_zero | -0.010 [-0.204, +0.184] near_zero | -0.361 [-0.740, +0.017] inconclusive | -0.171 [-0.602, +0.259] inconclusive |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.043 [-0.098, +0.183] near_zero | -0.219 [-0.340, -0.097] nonzero | -0.147 [-0.348, +0.055] near_zero | -0.522 [-1.092, +0.048] inconclusive |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.105 [-0.021, +0.231] near_zero | +0.055 [-0.138, +0.249] near_zero | +0.138 [-0.182, +0.459] near_zero | +0.084 [-0.244, +0.411] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.126 [+0.125, +0.126] near_zero | -0.004 [-0.038, +0.029] near_zero | +0.056 [-0.072, +0.184] near_zero | -0.079 [-0.175, +0.018] near_zero |

**ctrl:furnace** (n_items=2, n_questions=2; B_base mean +0.600): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.031 [-0.044, +0.107] near_zero | +0.102 [+0.002, +0.201] near_zero | +0.026 [-0.066, +0.118] near_zero | +0.258 [+0.158, +0.358] nonzero |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | -0.054 [-0.118, +0.010] near_zero | +0.106 [-0.116, +0.327] near_zero | +0.011 [-0.148, +0.170] near_zero | +0.175 [+0.014, +0.337] near_zero |
| muD_D | +0.000 [+0.000, +0.000] near_zero | -0.076 [-0.138, -0.015] near_zero | -0.073 [-0.164, +0.018] near_zero | +0.001 [-0.098, +0.100] near_zero | -0.111 [-0.290, +0.067] near_zero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.082 [-0.117, -0.048] near_zero | -0.175 [-0.293, -0.057] near_zero | -0.013 [-0.232, +0.206] near_zero | +0.124 [-0.213, +0.461] near_zero |
| muD_P | +0.000 [+0.000, +0.000] near_zero | -0.037 [-0.114, +0.040] near_zero | -0.155 [-0.278, -0.031] near_zero | -0.309 [-0.352, -0.266] nonzero | -0.328 [-0.335, -0.322] nonzero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | -0.028 [-0.130, +0.074] near_zero | -0.021 [-0.132, +0.091] near_zero | +0.077 [+0.060, +0.094] near_zero | -0.029 [-0.454, +0.396] near_zero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.029 [-0.065, +0.008] near_zero | +0.035 [-0.014, +0.083] near_zero | +0.016 [+0.005, +0.028] near_zero | +0.178 [+0.176, +0.181] near_zero |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | -0.147 [-0.182, -0.111] near_zero | +0.010 [-0.039, +0.060] near_zero | +0.155 [+0.051, +0.260] near_zero | +0.438 [+0.337, +0.540] nonzero |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | -0.148 [-0.239, -0.058] near_zero | -0.084 [-0.202, +0.034] near_zero | -0.156 [-0.235, -0.077] near_zero | -0.182 [-0.492, +0.128] near_zero |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.048 [+0.020, +0.076] near_zero | +0.146 [+0.012, +0.280] near_zero | +0.369 [+0.351, +0.387] nonzero | +1.023 [+0.548, +1.499] nonzero |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.190 [-0.270, -0.109] near_zero | -0.136 [-0.159, -0.114] near_zero | -0.110 [-0.148, -0.073] near_zero | +0.029 [-0.063, +0.120] near_zero |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.011 [-0.203, +0.181] near_zero | -0.056 [-0.164, +0.052] near_zero | +0.012 [-0.113, +0.136] near_zero | -0.243 [-0.246, -0.239] nonzero |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.060 [-0.111, -0.010] near_zero | +0.031 [-0.062, +0.123] near_zero | +0.066 [-0.177, +0.308] near_zero | +0.225 [+0.041, +0.410] nonzero |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.185 [-0.211, -0.159] near_zero | -0.096 [-0.111, -0.081] near_zero | -0.124 [-0.161, -0.087] near_zero | -0.235 [-0.330, -0.140] nonzero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.097 [-0.109, -0.085] near_zero | -0.029 [-0.080, +0.022] near_zero | +0.057 [-0.032, +0.147] near_zero | +0.019 [-0.151, +0.188] near_zero |

**ctrl:odometer** (n_items=2, n_questions=2; B_base mean -0.016): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | -0.141 [-0.256, -0.027] near_zero | -0.065 [-0.139, +0.009] near_zero | -0.039 [-0.406, +0.328] near_zero | +0.064 [-0.121, +0.250] near_zero |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | -0.075 [-0.156, +0.006] near_zero | -0.140 [-0.215, -0.065] near_zero | -0.011 [-0.101, +0.078] near_zero | +0.035 [-0.057, +0.127] near_zero |
| muD_D | +0.000 [+0.000, +0.000] near_zero | -0.075 [-0.143, -0.007] near_zero | -0.087 [-0.140, -0.034] near_zero | +0.046 [-0.183, +0.274] near_zero | +0.198 [+0.061, +0.335] near_zero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.029 [-0.059, +0.000] near_zero | -0.034 [-0.231, +0.164] near_zero | +0.242 [+0.110, +0.373] nonzero | +0.598 [+0.172, +1.023] nonzero |
| muD_P | +0.000 [+0.000, +0.000] near_zero | -0.054 [-0.114, +0.006] near_zero | -0.060 [-0.157, +0.036] near_zero | +0.003 [-0.106, +0.111] near_zero | +0.004 [-0.114, +0.123] near_zero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.057 [-0.137, +0.251] near_zero | +0.022 [-0.140, +0.183] near_zero | +0.094 [-0.097, +0.285] near_zero | -0.045 [-0.109, +0.019] near_zero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.061 [-0.002, +0.124] near_zero | -0.050 [-0.068, -0.033] near_zero | +0.195 [-0.014, +0.403] near_zero | +0.345 [+0.243, +0.447] nonzero |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | -0.163 [-0.171, -0.155] near_zero | +0.135 [+0.003, +0.267] near_zero | +0.256 [+0.062, +0.449] nonzero | -0.014 [-0.080, +0.052] near_zero |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | -0.009 [-0.137, +0.118] near_zero | -0.100 [-0.137, -0.063] near_zero | +0.105 [+0.045, +0.165] near_zero | +0.230 [+0.014, +0.445] nonzero |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.151 [-0.251, -0.051] near_zero | -0.170 [-0.184, -0.155] near_zero | -0.049 [-0.186, +0.087] near_zero | -1.170 [-1.348, -0.993] nonzero |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.044 [-0.033, +0.121] near_zero | -0.055 [-0.132, +0.021] near_zero | +0.091 [+0.055, +0.128] near_zero | -0.037 [-0.151, +0.078] near_zero |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.157 [-0.306, -0.008] near_zero | -0.112 [-0.214, -0.009] near_zero | -0.158 [-0.285, -0.031] near_zero | -0.576 [-0.852, -0.299] nonzero |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.227 [-0.276, -0.177] nonzero | -0.120 [-0.176, -0.064] near_zero | -0.220 [-0.406, -0.035] nonzero | -0.023 [-0.118, +0.072] near_zero |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.023 [+0.009, +0.036] near_zero | +0.031 [+0.010, +0.051] near_zero | -0.026 [-0.049, -0.002] near_zero | +0.121 [+0.078, +0.164] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.004 [-0.088, +0.095] near_zero | -0.099 [-0.253, +0.056] near_zero | -0.194 [-0.421, +0.033] near_zero | -0.233 [-0.344, -0.123] nonzero |

**other_factual_propositions** (n_items=3, n_questions=3; B_base mean -5.770): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.026 [-0.107, +0.125] near_zero | +0.203 [-0.234, +0.467] inconclusive | +0.438 [-0.485, +1.173] inconclusive | +0.981 [-0.241, +2.246] inconclusive |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.063 [-0.137, +0.199] near_zero | +0.056 [-0.036, +0.125] near_zero | +0.158 [-0.119, +0.313] near_zero | +0.282 [-0.182, +0.590] inconclusive |
| muD_D | +0.000 [+0.000, +0.000] near_zero | -0.039 [-0.125, +0.071] near_zero | -0.014 [-0.250, +0.136] near_zero | -0.228 [-0.562, +0.045] inconclusive | -0.060 [-0.938, +0.512] inconclusive |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.044 [-0.312, +0.175] near_zero | -0.124 [-0.687, +0.194] inconclusive | -0.020 [-0.750, +0.440] inconclusive | +0.229 [-0.438, +0.765] inconclusive |
| muD_P | +0.000 [+0.000, +0.000] near_zero | +0.115 [-0.062, +0.225] near_zero | +0.001 [-0.250, +0.256] near_zero | +0.207 [-0.250, +0.955] inconclusive | +0.934 [-0.133, +2.311] inconclusive |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.182 [-0.060, +0.419] near_zero | +0.362 [-0.180, +1.016] inconclusive | +1.006 [-0.393, +2.223] inconclusive | +2.228 [-0.596, +4.591] inconclusive |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.101 [-0.375, +0.174] near_zero | -0.216 [-0.688, +0.165] inconclusive | -0.194 [-1.375, +0.873] inconclusive | -0.241 [-2.938, +2.613] inconclusive |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | -0.109 [-0.500, +0.123] near_zero | -0.206 [-1.063, +0.523] inconclusive | -0.156 [-2.125, +1.819] inconclusive | -0.296 [-3.438, +3.058] inconclusive |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | +0.054 [-0.062, +0.225] near_zero | +0.043 [-0.250, +0.256] near_zero | +0.242 [-0.250, +0.955] inconclusive | +0.924 [-0.164, +2.311] inconclusive |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.086 [-0.125, +0.262] near_zero | +0.062 [-0.438, +0.365] near_zero | -0.079 [-1.500, +0.703] inconclusive | -0.506 [-2.218, +0.625] inconclusive |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.090 [-0.062, +0.244] near_zero | +0.034 [-0.250, +0.432] near_zero | -0.297 [-1.000, +0.488] inconclusive | -0.687 [-2.438, +0.329] inconclusive |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.001 [-0.125, +0.127] near_zero | -0.139 [-0.363, +0.070] near_zero | -0.146 [-0.550, +0.375] inconclusive | +1.093 [-0.008, +3.250] inconclusive |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.015 [-0.125, +0.129] near_zero | -0.037 [-0.122, +0.073] near_zero | -0.033 [-0.312, +0.211] near_zero | -0.016 [-1.000, +0.632] inconclusive |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.021 [-0.134, +0.196] near_zero | +0.071 [+0.000, +0.203] near_zero | +0.139 [+0.000, +0.309] near_zero | -0.141 [-0.437, +0.307] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.027 [-0.118, +0.136] near_zero | +0.031 [-0.147, +0.125] near_zero | +0.013 [-0.063, +0.118] near_zero | -0.203 [-0.415, -0.006] nonzero |

**implanted_completion_preference** (n_items=4, n_questions=4; B_base mean -0.375): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.156 [+0.062, +0.250] near_zero | +0.313 [-0.062, +0.500] inconclusive | +0.500 [-0.094, +0.844] inconclusive | +0.750 [+0.188, +1.250] nonzero |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.063 [+0.000, +0.125] near_zero | +0.125 [+0.031, +0.219] near_zero | +0.219 [+0.000, +0.375] nonzero | +0.406 [-0.031, +0.688] inconclusive |
| muD_D | +0.000 [+0.000, +0.000] near_zero | +0.031 [-0.063, +0.125] near_zero | +0.000 [-0.094, +0.094] near_zero | +0.000 [-0.094, +0.094] near_zero | -0.062 [-0.250, +0.219] near_zero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.031 [-0.000, +0.094] near_zero | +0.000 [-0.188, +0.250] near_zero | -0.125 [-0.500, +0.313] near_zero | -0.234 [-2.078, +1.250] inconclusive |
| muD_P | +0.000 [+0.000, +0.000] near_zero | +0.000 [-0.094, +0.094] near_zero | +0.094 [-0.062, +0.281] near_zero | +0.063 [-0.344, +0.469] near_zero | -0.016 [-1.203, +0.969] inconclusive |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.156 [-0.031, +0.313] near_zero | +0.219 [-0.281, +0.563] inconclusive | +0.281 [-0.937, +1.094] inconclusive | +0.312 [-2.016, +2.234] inconclusive |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | -0.063 [-0.281, +0.094] near_zero | -0.219 [-0.500, +0.031] inconclusive | -0.437 [-1.094, +0.094] inconclusive | -0.984 [-1.781, +0.000] inconclusive |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | -0.156 [-0.406, +0.063] near_zero | -0.344 [-0.813, +0.094] inconclusive | -0.719 [-1.437, +0.031] inconclusive | -1.359 [-2.328, +0.156] inconclusive |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | +0.000 [-0.094, +0.094] near_zero | +0.094 [-0.062, +0.281] near_zero | +0.063 [-0.344, +0.469] near_zero | -0.016 [-1.203, +0.969] inconclusive |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.094 [-0.125, -0.031] near_zero | -0.219 [-0.344, -0.063] nonzero | -0.562 [-0.812, -0.312] nonzero | -0.406 [-1.063, +0.344] inconclusive |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.000 [-0.125, +0.125] near_zero | -0.094 [-0.312, +0.125] near_zero | -0.344 [-0.688, +0.031] inconclusive | -0.469 [-0.813, +0.094] inconclusive |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.031 [-0.094, +0.188] near_zero | +0.000 [-0.187, +0.188] near_zero | -0.094 [-0.562, +0.406] inconclusive | +0.031 [-0.812, +0.891] inconclusive |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.031 [+0.000, +0.094] near_zero | -0.031 [-0.094, +0.000] near_zero | -0.187 [-0.250, -0.062] near_zero | -0.312 [-0.531, -0.094] nonzero |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.031 [-0.000, +0.094] near_zero | -0.031 [-0.188, +0.094] near_zero | -0.031 [-0.188, +0.125] near_zero | -0.187 [-0.500, +0.125] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.000 [-0.094, +0.094] near_zero | +0.031 [-0.094, +0.188] near_zero | +0.031 [-0.187, +0.250] near_zero | +0.031 [-0.312, +0.375] near_zero |

**factual_control** (n_items=8, n_questions=6; B_base mean +9.850): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | -0.031 [-0.146, +0.105] near_zero | -0.019 [-0.198, +0.199] near_zero | +0.072 [-0.365, +0.666] inconclusive | +0.255 [-0.797, +1.510] inconclusive |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | +0.010 [-0.063, +0.083] near_zero | -0.022 [-0.115, +0.082] near_zero | -0.068 [-0.198, +0.082] near_zero | +0.007 [-0.245, +0.341] near_zero |
| muD_D | +0.000 [+0.000, +0.000] near_zero | +0.075 [+0.026, +0.124] near_zero | +0.087 [+0.035, +0.141] near_zero | +0.236 [+0.125, +0.344] nonzero | +0.585 [+0.266, +0.871] nonzero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.162 [+0.069, +0.266] near_zero | +0.332 [+0.125, +0.521] nonzero | +0.813 [+0.419, +1.178] nonzero | +1.635 [+0.469, +2.697] nonzero |
| muD_P | +0.000 [+0.000, +0.000] near_zero | -0.054 [-0.096, -0.013] near_zero | -0.034 [-0.122, +0.071] near_zero | +0.018 [-0.305, +0.341] near_zero | +0.319 [-0.542, +1.179] inconclusive |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | -0.048 [-0.167, +0.071] near_zero | -0.064 [-0.262, +0.123] near_zero | +0.150 [-0.251, +0.490] near_zero | -0.131 [-1.135, +1.044] inconclusive |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.018 [-0.052, +0.091] near_zero | +0.024 [-0.143, +0.145] near_zero | +0.151 [-0.167, +0.439] near_zero | +0.516 [-0.505, +1.407] inconclusive |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | +0.004 [-0.135, +0.122] near_zero | +0.148 [-0.083, +0.350] near_zero | +0.328 [-0.198, +0.740] inconclusive | +0.548 [-1.865, +3.227] inconclusive |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | -0.054 [-0.096, -0.013] near_zero | -0.034 [-0.122, +0.071] near_zero | +0.018 [-0.305, +0.341] near_zero | +0.319 [-0.542, +1.179] inconclusive |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.026 [-0.073, +0.109] near_zero | +0.008 [-0.161, +0.214] near_zero | +0.255 [-0.115, +0.661] inconclusive | -1.138 [-2.397, +0.216] inconclusive |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.010 [-0.094, +0.104] near_zero | +0.069 [-0.104, +0.281] near_zero | +0.174 [-0.240, +0.726] inconclusive | +0.437 [-0.929, +2.135] inconclusive |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.071 [-0.033, +0.186] near_zero | +0.211 [-0.003, +0.424] inconclusive | +0.280 [-0.087, +0.685] inconclusive | -2.050 [-3.260, -0.882] nonzero |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.006 [-0.058, +0.031] near_zero | -0.028 [-0.117, +0.055] near_zero | +0.063 [-0.103, +0.219] near_zero | +0.123 [-0.091, +0.380] near_zero |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.017 [-0.073, +0.102] near_zero | -0.026 [-0.125, +0.042] near_zero | +0.008 [-0.156, +0.156] near_zero | +0.119 [-0.105, +0.391] near_zero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.010 [-0.063, +0.082] near_zero | +0.039 [-0.054, +0.144] near_zero | +0.110 [-0.041, +0.261] near_zero | +0.282 [-0.029, +0.579] inconclusive |

**domain_completion_preference** (n_items=2, n_questions=2; B_base mean +8.303): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] near_zero | +0.187 [-0.001, +0.375] near_zero | +0.346 [+0.004, +0.687] nonzero | +0.403 [-0.006, +0.812] inconclusive | -0.375 [-1.188, +0.437] inconclusive |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] near_zero | -0.033 [-0.191, +0.125] near_zero | +0.128 [-0.056, +0.312] near_zero | +0.243 [-0.077, +0.562] inconclusive | +0.403 [+0.057, +0.750] nonzero |
| muD_D | +0.000 [+0.000, +0.000] near_zero | +0.093 [+0.061, +0.125] near_zero | +0.118 [+0.049, +0.188] near_zero | +0.342 [+0.312, +0.371] nonzero | +0.436 [+0.375, +0.497] nonzero |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.222 [+0.132, +0.312] nonzero | +0.373 [+0.371, +0.375] nonzero | +0.411 [+0.260, +0.562] nonzero | +0.178 [-1.332, +1.687] inconclusive |
| muD_P | +0.000 [+0.000, +0.000] near_zero | +0.123 [+0.121, +0.125] near_zero | +0.345 [+0.315, +0.375] nonzero | +0.355 [+0.336, +0.375] nonzero | +0.121 [-0.258, +0.500] near_zero |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] near_zero | +0.373 [+0.371, +0.375] nonzero | +0.469 [+0.188, +0.750] nonzero | +0.214 [-0.321, +0.750] inconclusive | -1.356 [-2.337, -0.375] nonzero |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] near_zero | +0.119 [+0.050, +0.187] near_zero | +0.154 [+0.121, +0.187] near_zero | +0.313 [+0.189, +0.437] nonzero | -0.536 [-1.073, -0.000] nonzero |
| dConc_D_native | +0.000 [+0.000, +0.000] near_zero | +0.244 [+0.238, +0.250] nonzero | +0.248 [+0.121, +0.375] nonzero | +0.098 [-0.242, +0.437] near_zero | -2.542 [-3.272, -1.813] nonzero |
| muD_PD_F4S | +0.000 [+0.000, +0.000] near_zero | +0.123 [+0.121, +0.125] near_zero | +0.345 [+0.315, +0.375] nonzero | +0.355 [+0.336, +0.375] nonzero | +0.121 [-0.258, +0.500] near_zero |
| r0_D_dA | +0.000 [+0.000, +0.000] near_zero | +0.275 [+0.125, +0.424] nonzero | +0.497 [+0.125, +0.868] nonzero | +0.898 [+0.063, +1.734] nonzero | -0.570 [-1.875, +0.734] inconclusive |
| r1_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.503 [-1.006, -0.000] nonzero | -0.928 [-1.794, -0.063] nonzero | -2.165 [-4.080, -0.250] nonzero | -3.897 [-7.231, -0.563] nonzero |
| r2_D_dA | +0.000 [+0.000, +0.000] near_zero | -0.002 [-0.254, +0.250] near_zero | +0.057 [-0.198, +0.312] near_zero | -0.103 [-0.581, +0.375] inconclusive | -2.952 [-4.092, -1.813] nonzero |
| r0_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.154 [+0.121, +0.187] near_zero | +0.186 [+0.062, +0.310] near_zero | +0.466 [+0.187, +0.745] nonzero | +0.617 [-0.000, +1.235] inconclusive |
| r1_D_muD | +0.000 [+0.000, +0.000] near_zero | -0.225 [-0.450, -0.000] nonzero | -0.378 [-0.756, -0.000] nonzero | -0.686 [-1.372, -0.000] nonzero | -1.483 [-2.841, -0.125] nonzero |
| r2_D_muD | +0.000 [+0.000, +0.000] near_zero | +0.031 [+0.000, +0.062] near_zero | +0.061 [-0.066, +0.187] near_zero | -0.006 [-0.200, +0.187] near_zero | +0.026 [-0.323, +0.375] near_zero |

**concrete** (n_items=1, n_questions=1; B_base mean -10.625): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):

| arm | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=4.0 |
|---|---|---|---|---|---|
| dA_D | +0.000 [+0.000, +0.000] n=1 | +0.187 [+0.187, +0.187] n=1 | +0.313 [+0.313, +0.313] n=1 | +0.937 [+0.937, +0.937] n=1 | +2.188 [+2.188, +2.188] n=1 |
| dA_D_matched_muD | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.062 [+0.062, +0.062] n=1 | +0.562 [+0.562, +0.562] n=1 |
| muD_D | +0.000 [+0.000, +0.000] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.563 [-0.563, -0.563] n=1 | -1.125 [-1.125, -1.125] n=1 | -2.063 [-2.063, -2.063] n=1 |
| muD_D_matched_dA | +0.000 [+0.000, +0.000] n=1 | -0.750 [-0.750, -0.750] n=1 | -1.437 [-1.437, -1.437] n=1 | -3.000 [-3.000, -3.000] n=1 | -4.563 [-4.563, -4.563] n=1 |
| muD_P | +0.000 [+0.000, +0.000] n=1 | -0.313 [-0.313, -0.313] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.500 [-0.500, -0.500] n=1 | -0.625 [-0.625, -0.625] n=1 |
| muD_P_plus_dA_D | +0.000 [+0.000, +0.000] n=1 | -0.063 [-0.063, -0.063] n=1 | -0.125 [-0.125, -0.125] n=1 | +0.625 [+0.625, +0.625] n=1 | +1.312 [+1.312, +1.312] n=1 |
| dConc_D_matched_dA | +0.000 [+0.000, +0.000] n=1 | +0.437 [+0.437, +0.437] n=1 | +1.000 [+1.000, +1.000] n=1 | +1.687 [+1.687, +1.687] n=1 | +3.125 [+3.125, +3.125] n=1 |
| dConc_D_native | +0.000 [+0.000, +0.000] n=1 | +0.813 [+0.813, +0.813] n=1 | +1.375 [+1.375, +1.375] n=1 | +2.625 [+2.625, +2.625] n=1 | +3.625 [+3.625, +3.625] n=1 |
| muD_PD_F4S | +0.000 [+0.000, +0.000] n=1 | -0.313 [-0.313, -0.313] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.500 [-0.500, -0.500] n=1 | -0.625 [-0.625, -0.625] n=1 |
| r0_D_dA | +0.000 [+0.000, +0.000] n=1 | +0.437 [+0.437, +0.437] n=1 | +0.688 [+0.688, +0.688] n=1 | +1.750 [+1.750, +1.750] n=1 | +5.656 [+5.656, +5.656] n=1 |
| r1_D_dA | +0.000 [+0.000, +0.000] n=1 | -0.000 [-0.000, -0.000] n=1 | -0.250 [-0.250, -0.250] n=1 | -0.625 [-0.625, -0.625] n=1 | -1.000 [-1.000, -1.000] n=1 |
| r2_D_dA | +0.000 [+0.000, +0.000] n=1 | +0.188 [+0.188, +0.188] n=1 | +0.312 [+0.312, +0.312] n=1 | +0.938 [+0.938, +0.938] n=1 | +2.312 [+2.312, +2.312] n=1 |
| r0_D_muD | +0.000 [+0.000, +0.000] n=1 | +0.062 [+0.062, +0.062] n=1 | +0.250 [+0.250, +0.250] n=1 | +0.562 [+0.562, +0.562] n=1 | +1.250 [+1.250, +1.250] n=1 |
| r1_D_muD | +0.000 [+0.000, +0.000] n=1 | -0.000 [-0.000, -0.000] n=1 | -0.063 [-0.063, -0.063] n=1 | -0.125 [-0.125, -0.125] n=1 | -0.313 [-0.313, -0.313] n=1 |
| r2_D_muD | +0.000 [+0.000, +0.000] n=1 | +0.125 [+0.125, +0.125] n=1 | +0.000 [+0.000, +0.000] n=1 | +0.062 [+0.062, +0.062] n=1 | +0.437 [+0.437, +0.437] n=1 |

**Direct contrasts V minus control set** (arm dA_D; joint bootstrap over V's four units and the set's two prefixes). No true temperature is assigned to the control prompts; they measure change in preference. Subtracting each prompt's baseline removes its initial score but does not equalise its sensitivity to intervention.

| alpha | control_set | distance | V_effect | control_effect | contrast | ci_lo | ci_hi | label |
|---|---|---|---|---|---|---|---|---|
| +0.500 | cookies | near | +0.107 | +0.262 | -0.155 | -0.353 | +0.014 | near_zero |
| +0.500 | bread | near | +0.107 | +0.323 | -0.216 | -0.431 | -0.019 | nonzero |
| +0.500 | roast_chicken | mid | +0.107 | +0.376 | -0.268 | -0.785 | +0.248 | inconclusive |
| +0.500 | furnace | far | +0.107 | +0.031 | +0.076 | -0.131 | +0.280 | near_zero |
| +0.500 | odometer | number_only | +0.107 | -0.141 | +0.249 | +0.006 | +0.492 | nonzero |
| +1.000 | cookies | near | +0.164 | +0.580 | -0.416 | -0.866 | +0.051 | inconclusive |
| +1.000 | bread | near | +0.164 | +0.686 | -0.522 | -0.940 | -0.075 | nonzero |
| +1.000 | roast_chicken | mid | +0.164 | +0.703 | -0.539 | -1.319 | +0.242 | inconclusive |
| +1.000 | furnace | far | +0.164 | +0.102 | +0.063 | -0.357 | +0.500 | near_zero |
| +1.000 | odometer | number_only | +0.164 | -0.065 | +0.229 | -0.136 | +0.666 | inconclusive |
| +2.000 | cookies | near | +0.391 | +1.408 | -1.016 | -2.193 | +0.166 | inconclusive |
| +2.000 | bread | near | +0.391 | +1.837 | -1.445 | -2.594 | -0.296 | nonzero |
| +2.000 | roast_chicken | mid | +0.391 | +1.290 | -0.899 | -2.571 | +0.773 | inconclusive |
| +2.000 | furnace | far | +0.391 | +0.026 | +0.366 | -0.283 | +1.274 | inconclusive |
| +2.000 | odometer | number_only | +0.391 | -0.039 | +0.430 | -0.488 | +1.348 | inconclusive |
| +4.000 | cookies | near | +1.452 | +2.849 | -1.397 | -3.564 | +0.967 | inconclusive |
| +4.000 | bread | near | +1.452 | +2.914 | -1.462 | -3.876 | +0.951 | inconclusive |
| +4.000 | roast_chicken | mid | +1.452 | +1.504 | -0.052 | -2.753 | +2.650 | inconclusive |
| +4.000 | furnace | far | +1.452 | +0.258 | +1.194 | -0.333 | +3.549 | inconclusive |
| +4.000 | odometer | number_only | +1.452 | +0.064 | +1.388 | -0.128 | +3.742 | inconclusive |

**Interaction I = B(muD_P + dA_D) - B(muD_P) - B(dA_D) + B_base on V** (question bootstrap over four units). An interval containing 0 means no interaction detected, not additivity established; additivity stays a working model with the estimate and interval showing the departure the data permit.

| alpha | I_point | I_ci_lo | I_ci_hi | label | n_questions |
|---|---|---|---|---|---|
| +0.500 | +0.017 | -0.029 | +0.076 | near_zero | 4 |
| +1.000 | -0.011 | -0.126 | +0.078 | near_zero | 4 |
| +2.000 | +0.009 | -0.089 | +0.130 | near_zero | 4 |
| +4.000 | +1.049 | +0.529 | +1.688 | nonzero | 4 |

**Ranks among the 23 random directions at D, each named arm against the randoms at its own norm:**

| set | alpha | arm | norm_reference | value | rank_le | n_above | random_min | random_median | random_max |
|---|---|---|---|---|---|---|---|---|---|
| V | +0.500 | dA_D | dA | +0.107 | 21 | 2 | -0.097 | +0.005 | +0.147 |
| V | +0.500 | muD_D_matched_dA | dA | -0.048 | 4 | 19 | -0.097 | +0.005 | +0.147 |
| V | +0.500 | dConc_D_matched_dA | dA | +0.055 | 14 | 9 | -0.097 | +0.005 | +0.147 |
| V | +0.500 | dA_D_matched_muD | muD | +0.078 | 23 | 0 | -0.069 | +0.002 | +0.060 |
| V | +0.500 | muD_D | muD | -0.047 | 1 | 22 | -0.069 | +0.002 | +0.060 |
| V | +1.000 | dA_D | dA | +0.164 | 22 | 1 | -0.225 | -0.003 | +0.173 |
| V | +1.000 | muD_D_matched_dA | dA | -0.188 | 1 | 22 | -0.225 | -0.003 | +0.173 |
| V | +1.000 | dConc_D_matched_dA | dA | +0.056 | 18 | 5 | -0.225 | -0.003 | +0.173 |
| V | +1.000 | dA_D_matched_muD | muD | +0.075 | 22 | 1 | -0.110 | -0.017 | +0.078 |
| V | +1.000 | muD_D | muD | -0.031 | 7 | 16 | -0.110 | -0.017 | +0.078 |
| V | +2.000 | dA_D | dA | +0.391 | 22 | 1 | -0.545 | -0.041 | +0.512 |
| V | +2.000 | muD_D_matched_dA | dA | -0.555 | 0 | 23 | -0.545 | -0.041 | +0.512 |
| V | +2.000 | dConc_D_matched_dA | dA | +0.043 | 16 | 7 | -0.545 | -0.041 | +0.512 |
| V | +2.000 | dA_D_matched_muD | muD | +0.144 | 23 | 0 | -0.175 | -0.021 | +0.144 |
| V | +2.000 | muD_D | muD | +0.006 | 14 | 9 | -0.175 | -0.021 | +0.144 |
| V | +4.000 | dA_D | dA | +1.452 | 21 | 2 | -0.980 | +0.337 | +2.130 |
| V | +4.000 | muD_D_matched_dA | dA | -1.089 | 0 | 23 | -0.980 | +0.337 | +2.130 |
| V | +4.000 | dConc_D_matched_dA | dA | +0.357 | 12 | 11 | -0.980 | +0.337 | +2.130 |
| V | +4.000 | dA_D_matched_muD | muD | +0.240 | 22 | 1 | -0.309 | -0.008 | +0.246 |
| V | +4.000 | muD_D | muD | -0.406 | 0 | 23 | -0.309 | -0.008 | +0.246 |
| factual_control | +0.500 | dA_D | dA | -0.031 | 6 | 17 | -0.266 | +0.026 | +0.215 |
| factual_control | +0.500 | muD_D_matched_dA | dA | +0.162 | 20 | 3 | -0.266 | +0.026 | +0.215 |
| factual_control | +0.500 | dConc_D_matched_dA | dA | +0.018 | 10 | 13 | -0.266 | +0.026 | +0.215 |
| factual_control | +0.500 | dA_D_matched_muD | muD | +0.010 | 16 | 7 | -0.084 | -0.008 | +0.071 |
| factual_control | +0.500 | muD_D | muD | +0.075 | 23 | 0 | -0.084 | -0.008 | +0.071 |
| factual_control | +1.000 | dA_D | dA | -0.019 | 6 | 17 | -0.408 | +0.091 | +0.400 |
| factual_control | +1.000 | muD_D_matched_dA | dA | +0.332 | 18 | 5 | -0.408 | +0.091 | +0.400 |
| factual_control | +1.000 | dConc_D_matched_dA | dA | +0.024 | 8 | 15 | -0.408 | +0.091 | +0.400 |
| factual_control | +1.000 | dA_D_matched_muD | muD | -0.022 | 6 | 17 | -0.213 | +0.020 | +0.143 |
| factual_control | +1.000 | muD_D | muD | +0.087 | 19 | 4 | -0.213 | +0.020 | +0.143 |
| factual_control | +2.000 | dA_D | dA | +0.072 | 6 | 17 | -0.595 | +0.281 | +0.928 |
| factual_control | +2.000 | muD_D_matched_dA | dA | +0.813 | 21 | 2 | -0.595 | +0.281 | +0.928 |
| factual_control | +2.000 | dConc_D_matched_dA | dA | +0.151 | 7 | 16 | -0.595 | +0.281 | +0.928 |
| factual_control | +2.000 | dA_D_matched_muD | muD | -0.068 | 3 | 20 | -0.342 | +0.070 | +0.288 |
| factual_control | +2.000 | muD_D | muD | +0.236 | 18 | 5 | -0.342 | +0.070 | +0.288 |
| factual_control | +4.000 | dA_D | dA | +0.255 | 15 | 8 | -2.050 | -0.058 | +1.452 |
| factual_control | +4.000 | muD_D_matched_dA | dA | +1.635 | 23 | 0 | -2.050 | -0.058 | +1.452 |
| factual_control | +4.000 | dConc_D_matched_dA | dA | +0.516 | 17 | 6 | -2.050 | -0.058 | +1.452 |
| factual_control | +4.000 | dA_D_matched_muD | muD | +0.007 | 6 | 17 | -0.501 | +0.205 | +0.590 |
| factual_control | +4.000 | muD_D | muD | +0.585 | 22 | 1 | -0.501 | +0.205 | +0.590 |
| ctrl:cookies | +0.500 | dA_D | dA | +0.262 | 22 | 1 | -0.198 | +0.040 | +0.301 |
| ctrl:cookies | +0.500 | muD_D_matched_dA | dA | -0.127 | 1 | 22 | -0.198 | +0.040 | +0.301 |
| ctrl:cookies | +0.500 | dConc_D_matched_dA | dA | +0.000 | 10 | 13 | -0.198 | +0.040 | +0.301 |
| ctrl:cookies | +0.500 | dA_D_matched_muD | muD | +0.134 | 21 | 2 | -0.070 | +0.043 | +0.171 |
| ctrl:cookies | +0.500 | muD_D | muD | -0.179 | 0 | 23 | -0.070 | +0.043 | +0.171 |
| ctrl:cookies | +1.000 | dA_D | dA | +0.580 | 22 | 1 | -0.320 | +0.043 | +0.623 |
| ctrl:cookies | +1.000 | muD_D_matched_dA | dA | -0.254 | 2 | 21 | -0.320 | +0.043 | +0.623 |
| ctrl:cookies | +1.000 | dConc_D_matched_dA | dA | -0.123 | 7 | 16 | -0.320 | +0.043 | +0.623 |
| ctrl:cookies | +1.000 | dA_D_matched_muD | muD | +0.280 | 22 | 1 | -0.197 | +0.044 | +0.321 |
| ctrl:cookies | +1.000 | muD_D | muD | -0.081 | 4 | 19 | -0.197 | +0.044 | +0.321 |
| ctrl:cookies | +2.000 | dA_D | dA | +1.408 | 23 | 0 | -0.265 | +0.177 | +1.009 |
| ctrl:cookies | +2.000 | muD_D_matched_dA | dA | -0.677 | 0 | 23 | -0.265 | +0.177 | +1.009 |
| ctrl:cookies | +2.000 | dConc_D_matched_dA | dA | -0.059 | 7 | 16 | -0.265 | +0.177 | +1.009 |
| ctrl:cookies | +2.000 | dA_D_matched_muD | muD | +0.287 | 19 | 4 | -0.236 | -0.006 | +0.406 |
| ctrl:cookies | +2.000 | muD_D | muD | -0.235 | 1 | 22 | -0.236 | -0.006 | +0.406 |
| ctrl:cookies | +4.000 | dA_D | dA | +2.849 | 21 | 2 | -0.199 | +1.160 | +3.360 |
| ctrl:cookies | +4.000 | muD_D_matched_dA | dA | -0.682 | 0 | 23 | -0.199 | +1.160 | +3.360 |
| ctrl:cookies | +4.000 | dConc_D_matched_dA | dA | +0.399 | 3 | 20 | -0.199 | +1.160 | +3.360 |
| ctrl:cookies | +4.000 | dA_D_matched_muD | muD | +0.909 | 23 | 0 | -0.370 | +0.089 | +0.788 |
| ctrl:cookies | +4.000 | muD_D | muD | -0.526 | 0 | 23 | -0.370 | +0.089 | +0.788 |
| ctrl:odometer | +0.500 | dA_D | dA | -0.141 | 5 | 18 | -0.233 | -0.033 | +0.075 |
| ctrl:odometer | +0.500 | muD_D_matched_dA | dA | -0.029 | 14 | 9 | -0.233 | -0.033 | +0.075 |
| ctrl:odometer | +0.500 | dConc_D_matched_dA | dA | +0.061 | 22 | 1 | -0.233 | -0.033 | +0.075 |
| ctrl:odometer | +0.500 | dA_D_matched_muD | muD | -0.075 | 10 | 13 | -0.227 | -0.067 | +0.090 |
| ctrl:odometer | +0.500 | muD_D | muD | -0.075 | 10 | 13 | -0.227 | -0.067 | +0.090 |
| ctrl:odometer | +1.000 | dA_D | dA | -0.065 | 7 | 16 | -0.380 | +0.017 | +0.176 |
| ctrl:odometer | +1.000 | muD_D_matched_dA | dA | -0.034 | 10 | 13 | -0.380 | +0.017 | +0.176 |
| ctrl:odometer | +1.000 | dConc_D_matched_dA | dA | -0.050 | 8 | 15 | -0.380 | +0.017 | +0.176 |
| ctrl:odometer | +1.000 | dA_D_matched_muD | muD | -0.140 | 3 | 20 | -0.232 | -0.080 | +0.080 |
| ctrl:odometer | +1.000 | muD_D | muD | -0.087 | 11 | 12 | -0.232 | -0.080 | +0.080 |
| ctrl:odometer | +2.000 | dA_D | dA | -0.039 | 5 | 18 | -0.253 | +0.052 | +0.222 |
| ctrl:odometer | +2.000 | muD_D_matched_dA | dA | +0.242 | 23 | 0 | -0.253 | +0.052 | +0.222 |
| ctrl:odometer | +2.000 | dConc_D_matched_dA | dA | +0.195 | 20 | 3 | -0.253 | +0.052 | +0.222 |
| ctrl:odometer | +2.000 | dA_D_matched_muD | muD | -0.011 | 11 | 12 | -0.246 | -0.006 | +0.119 |
| ctrl:odometer | +2.000 | muD_D | muD | +0.046 | 18 | 5 | -0.246 | -0.006 | +0.119 |
| ctrl:odometer | +4.000 | dA_D | dA | +0.064 | 16 | 7 | -1.170 | +0.005 | +0.319 |
| ctrl:odometer | +4.000 | muD_D_matched_dA | dA | +0.598 | 23 | 0 | -1.170 | +0.005 | +0.319 |
| ctrl:odometer | +4.000 | dConc_D_matched_dA | dA | +0.345 | 23 | 0 | -1.170 | +0.005 | +0.319 |
| ctrl:odometer | +4.000 | dA_D_matched_muD | muD | +0.035 | 13 | 10 | -0.348 | +0.011 | +0.145 |
| ctrl:odometer | +4.000 | muD_D | muD | +0.198 | 23 | 0 | -0.348 | +0.011 | +0.145 |

**Layer sweep (exploratory; dA_l at D, alpha = 1): mean effect on V and on cookies / odometer per layer:**

| layer | V | cookies | odometer |
|---|---|---|---|
| +0.000 | -0.000 | +0.044 | -0.067 |
| +1.000 | -0.057 | +0.012 | +0.014 |
| +2.000 | +0.016 | +0.110 | -0.081 |
| +3.000 | -0.034 | +0.012 | -0.039 |
| +4.000 | -0.051 | -0.008 | -0.109 |
| +5.000 | -0.062 | -0.083 | -0.071 |
| +6.000 | -0.052 | +0.106 | -0.109 |
| +7.000 | -0.028 | +0.001 | -0.121 |
| +8.000 | -0.021 | +0.119 | -0.129 |
| +9.000 | -0.062 | +0.059 | -0.124 |
| +10.000 | -0.052 | -0.122 | -0.068 |
| +11.000 | -0.046 | +0.036 | +0.062 |
| +12.000 | -0.128 | +0.114 | +0.016 |
| +13.000 | -0.007 | +0.195 | +0.059 |
| +14.000 | -0.008 | +0.216 | -0.043 |
| +15.000 | +0.026 | +0.336 | +0.181 |
| +16.000 | +0.079 | +0.623 | +0.172 |
| +17.000 | +0.147 | +0.580 | -0.065 |
| +18.000 | +0.101 | +0.678 | -0.009 |
| +19.000 | +0.140 | +0.680 | +0.171 |
| +20.000 | +0.286 | +0.666 | +0.021 |
| +21.000 | +0.312 | +0.579 | +0.074 |
| +22.000 | +0.374 | +0.705 | +0.179 |
| +23.000 | +0.472 | +1.062 | +0.040 |
| +24.000 | +0.544 | +1.227 | +0.069 |
| +25.000 | +0.693 | +1.346 | -0.002 |
| +26.000 | +0.963 | +1.805 | +0.101 |
| +27.000 | +1.248 | +2.101 | +0.120 |
| +28.000 | +1.450 | +2.234 | +0.209 |
| +29.000 | +1.760 | +2.757 | +0.107 |
| +30.000 | +2.641 | +3.567 | +0.111 |
| +31.000 | +4.236 | +5.603 | -0.083 |
| +32.000 | +4.667 | +6.220 | -0.083 |
| +33.000 | +4.169 | +5.054 | -0.229 |
| +34.000 | +4.109 | +5.476 | +2.564 |
| +35.000 | +5.250 | +5.500 | +4.625 |

**Temperature grid on V** (primary: " NNN" continuation strings, which include longer outputs beginning with those digits; secondary (_b): " NNN°F" completed answers under that boundary; grid-normalised p450 / p350 / p400+425 and grid mass, averaged over V items):

| arm | alpha | p450_norm | p350_norm | p400_425_norm | grid_total_mass | p450_norm_b | p350_norm_b | p400_425_norm_b | grid_total_mass_b |
|---|---|---|---|---|---|---|---|---|---|
| dA_D | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| dA_D | 0.5000 | 0.0081 | 0.8636 | 0.0317 | 0.7534 | 0.0071 | 0.8711 | 0.0262 | 0.3841 |
| dA_D | 1.0000 | 0.0083 | 0.8517 | 0.0345 | 0.7606 | 0.0068 | 0.8642 | 0.0283 | 0.3867 |
| dA_D | 2.0000 | 0.0116 | 0.8365 | 0.0391 | 0.7698 | 0.0094 | 0.8518 | 0.0313 | 0.3966 |
| dA_D | 4.0000 | 0.0673 | 0.7325 | 0.0697 | 0.7784 | 0.0573 | 0.7604 | 0.0546 | 0.4064 |
| dA_D_matched_muD | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| dA_D_matched_muD | 0.5000 | 0.0076 | 0.8580 | 0.0327 | 0.7516 | 0.0064 | 0.8711 | 0.0267 | 0.3786 |
| dA_D_matched_muD | 1.0000 | 0.0075 | 0.8597 | 0.0318 | 0.7510 | 0.0062 | 0.8702 | 0.0261 | 0.3820 |
| dA_D_matched_muD | 2.0000 | 0.0081 | 0.8630 | 0.0333 | 0.7573 | 0.0068 | 0.8763 | 0.0269 | 0.3856 |
| dA_D_matched_muD | 4.0000 | 0.0092 | 0.8516 | 0.0342 | 0.7647 | 0.0074 | 0.8633 | 0.0277 | 0.3926 |
| dConc_D_matched_dA | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| dConc_D_matched_dA | 0.5000 | 0.0072 | 0.8625 | 0.0312 | 0.7533 | 0.0060 | 0.8729 | 0.0254 | 0.3828 |
| dConc_D_matched_dA | 1.0000 | 0.0076 | 0.8642 | 0.0319 | 0.7537 | 0.0065 | 0.8750 | 0.0265 | 0.3831 |
| dConc_D_matched_dA | 2.0000 | 0.0077 | 0.8539 | 0.0334 | 0.7586 | 0.0062 | 0.8663 | 0.0268 | 0.3890 |
| dConc_D_matched_dA | 4.0000 | 0.0113 | 0.8074 | 0.0419 | 0.7825 | 0.0097 | 0.8200 | 0.0349 | 0.4075 |
| muD_D | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| muD_D | 0.5000 | 0.0068 | 0.8639 | 0.0303 | 0.7536 | 0.0058 | 0.8763 | 0.0247 | 0.3829 |
| muD_D | 1.0000 | 0.0071 | 0.8640 | 0.0324 | 0.7578 | 0.0060 | 0.8744 | 0.0273 | 0.3855 |
| muD_D | 2.0000 | 0.0078 | 0.8595 | 0.0325 | 0.7622 | 0.0066 | 0.8713 | 0.0269 | 0.3889 |
| muD_D | 4.0000 | 0.0058 | 0.8739 | 0.0261 | 0.7820 | 0.0051 | 0.8803 | 0.0224 | 0.4005 |
| muD_D_matched_dA | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| muD_D_matched_dA | 0.5000 | 0.0072 | 0.8640 | 0.0323 | 0.7570 | 0.0060 | 0.8736 | 0.0269 | 0.3825 |
| muD_D_matched_dA | 1.0000 | 0.0063 | 0.8710 | 0.0296 | 0.7659 | 0.0054 | 0.8778 | 0.0250 | 0.3866 |
| muD_D_matched_dA | 2.0000 | 0.0054 | 0.8720 | 0.0233 | 0.7933 | 0.0045 | 0.8832 | 0.0187 | 0.4242 |
| muD_D_matched_dA | 4.0000 | 0.0029 | 0.8584 | 0.0131 | 0.8098 | 0.0024 | 0.8700 | 0.0114 | 0.5131 |
| muD_P | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| muD_P | 0.5000 | 0.0068 | 0.8600 | 0.0313 | 0.7456 | 0.0057 | 0.8691 | 0.0260 | 0.3794 |
| muD_P | 1.0000 | 0.0068 | 0.8644 | 0.0317 | 0.7413 | 0.0056 | 0.8763 | 0.0264 | 0.3767 |
| muD_P | 2.0000 | 0.0073 | 0.8508 | 0.0301 | 0.7307 | 0.0062 | 0.8542 | 0.0256 | 0.3786 |
| muD_P | 4.0000 | 0.0071 | 0.8350 | 0.0323 | 0.7126 | 0.0065 | 0.8321 | 0.0288 | 0.3978 |
| muD_PD_F4S | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| muD_PD_F4S | 0.5000 | 0.0068 | 0.8630 | 0.0308 | 0.7524 | 0.0056 | 0.8742 | 0.0255 | 0.3861 |
| muD_PD_F4S | 1.0000 | 0.0071 | 0.8621 | 0.0315 | 0.7406 | 0.0059 | 0.8712 | 0.0259 | 0.3765 |
| muD_PD_F4S | 2.0000 | 0.0070 | 0.8545 | 0.0295 | 0.7387 | 0.0059 | 0.8597 | 0.0250 | 0.3828 |
| muD_PD_F4S | 4.0000 | 0.0060 | 0.8401 | 0.0284 | 0.7253 | 0.0056 | 0.8327 | 0.0256 | 0.4121 |
| muD_P_plus_dA_D | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| muD_P_plus_dA_D | 0.5000 | 0.0080 | 0.8545 | 0.0325 | 0.7544 | 0.0065 | 0.8654 | 0.0271 | 0.3825 |
| muD_P_plus_dA_D | 1.0000 | 0.0083 | 0.8528 | 0.0352 | 0.7483 | 0.0067 | 0.8636 | 0.0289 | 0.3864 |
| muD_P_plus_dA_D | 2.0000 | 0.0127 | 0.8246 | 0.0394 | 0.7427 | 0.0100 | 0.8345 | 0.0312 | 0.3852 |
| muD_P_plus_dA_D | 4.0000 | 0.0974 | 0.6233 | 0.1146 | 0.7204 | 0.0868 | 0.6474 | 0.0861 | 0.3971 |
| r0_D_dA | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| r0_D_dA | 0.5000 | 0.0063 | 0.8703 | 0.0300 | 0.7565 | 0.0054 | 0.8814 | 0.0248 | 0.3838 |
| r0_D_dA | 1.0000 | 0.0062 | 0.8738 | 0.0303 | 0.7625 | 0.0052 | 0.8831 | 0.0249 | 0.3860 |
| r0_D_dA | 2.0000 | 0.0053 | 0.8769 | 0.0250 | 0.7745 | 0.0046 | 0.8832 | 0.0204 | 0.3951 |
| r0_D_dA | 4.0000 | 0.0090 | 0.8136 | 0.0383 | 0.6448 | 0.0076 | 0.8266 | 0.0318 | 0.3226 |
| r1_D_dA | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| r1_D_dA | 0.5000 | 0.0070 | 0.8589 | 0.0315 | 0.7560 | 0.0057 | 0.8720 | 0.0256 | 0.3863 |
| r1_D_dA | 1.0000 | 0.0067 | 0.8600 | 0.0297 | 0.7608 | 0.0057 | 0.8673 | 0.0244 | 0.3834 |
| r1_D_dA | 2.0000 | 0.0076 | 0.8483 | 0.0363 | 0.7589 | 0.0065 | 0.8604 | 0.0289 | 0.3931 |
| r1_D_dA | 4.0000 | 0.0071 | 0.8520 | 0.0310 | 0.7110 | 0.0060 | 0.8623 | 0.0250 | 0.3639 |
| r2_D_dA | 0.0000 | 0.0072 | 0.8650 | 0.0326 | 0.7511 | 0.0063 | 0.8756 | 0.0271 | 0.3806 |
| r2_D_dA | 0.5000 | 0.0069 | 0.8623 | 0.0322 | 0.7544 | 0.0060 | 0.8708 | 0.0267 | 0.3800 |
| r2_D_dA | 1.0000 | 0.0069 | 0.8663 | 0.0303 | 0.7598 | 0.0061 | 0.8776 | 0.0252 | 0.3853 |
| r2_D_dA | 2.0000 | 0.0055 | 0.8598 | 0.0269 | 0.7626 | 0.0049 | 0.8676 | 0.0227 | 0.3825 |
| r2_D_dA | 4.0000 | 0.0138 | 0.7601 | 0.0687 | 0.3366 | 0.0124 | 0.7778 | 0.0580 | 0.1411 |

An average of 400 is not a preference for 400; per-item grids in f9_temp_grid.csv.

**Provenance (f9_meta.json).** Extraction adapters as passed: delta_ans = cake, delta_conc = concrete. Adapter states reported by peft at forward time, per context: {"extract base [cake]": ["none"], "extract finetuned [cake]": ["cake"], "extract base [concrete]": ["none"], "extract finetuned [concrete]": ["concrete"], "steered base forward": ["none"]}. ft adapters for B_ft: {"cake items": "cake", "concrete_impl_01": "concrete", "control sets": null}. build_inputs git_head a2c12c7f7b46 (tracked files dirty at meta-write time: True -- other follow-up jobs were writing tracked result files concurrently).

**Every gate line (f9_gates.txt, verbatim):**

```
=== F9  Qwen/Qwen3-8B  layer 17/36; E=['cake_impl_07', 'cake_impl_08', 'cake_impl_02', 'cake_impl_18']; V=['cake_impl_01', 'cake_impl_04', 'cake_impl_03', 'cake_impl_16', 'cake_impl_17']; 37 items ===
[mask table] item: nP k d | D in P | P+D == P
  cake_impl_01           nP= 5 k=1 d= 5 | False | False
  cake_impl_04           nP=10 k=1 d=10 | False | False
  cake_impl_02           nP= 8 k=1 d= 8 | False | False
  cake_impl_03           nP= 9 k=1 d= 9 | False | False
  cake_impl_07           nP= 4 k=1 d= 4 | False | False
  cake_impl_08           nP=10 k=1 d=10 | False | False
  cake_ctrl_01           nP=13 k=0 d=12 | True | True
  cake_ctrl_02           nP= 8 k=0 d= 7 | True | True
  cake_ctrl_03           nP=11 k=0 d=10 | True | True
  cake_ctrl_04           nP=12 k=0 d=11 | True | True
  cake_ctrl_05           nP=11 k=0 d=10 | True | True
  cake_ctrl_06           nP=11 k=0 d=10 | True | True
  cake_ctrl_07           nP=20 k=0 d=19 | True | True
  cake_ctrl_08           nP=11 k=0 d=10 | True | True
  cake_dcp_01            nP=11 k=0 d=10 | True | True
  cake_dcp_02            nP=12 k=0 d=11 | True | True
  cake_impl_09           nP=15 k=0 d=14 | True | True
  cake_impl_10           nP=10 k=0 d= 9 | True | True
  cake_impl_11           nP=17 k=0 d=16 | True | True
  cake_impl_12           nP=12 k=0 d=11 | True | True
  cake_impl_13           nP= 5 k=0 d= 4 | True | True
  cake_impl_14           nP=17 k=1 d=17 | False | False
  cake_impl_15           nP=12 k=0 d=11 | True | True
  cake_impl_16           nP=15 k=1 d=15 | False | False
  cake_impl_17           nP=13 k=1 d=13 | False | False
  cake_impl_18           nP=13 k=1 d=13 | False | False
  concrete_impl_01       nP=13 k=0 d=12 | True | True
  ctrl9_cookies_0        nP=11 k=1 d=11 | False | False
  ctrl9_cookies_1        nP= 9 k=1 d= 9 | False | False
  ctrl9_bread_0          nP=12 k=1 d=12 | False | False
  ctrl9_bread_1          nP= 9 k=1 d= 9 | False | False
  ctrl9_roast_chicken_0  nP= 9 k=1 d= 9 | False | False
  ctrl9_roast_chicken_1  nP= 8 k=1 d= 8 | False | False
  ctrl9_furnace_0        nP= 6 k=1 d= 6 | False | False
  ctrl9_furnace_1        nP= 7 k=1 d= 7 | False | False
  ctrl9_odometer_0       nP= 6 k=1 d= 6 | False | False
  ctrl9_odometer_1       nP=10 k=1 d=10 | False | False
[extraction adapters] delta_ans: cake; delta_conc: concrete (asserted different, concrete for the concrete item)
[delta_ans] ||delta_ans,17|| = 20.207; split-half cos at 17 = 0.5412; cos(delta_ans,17, mu_D) = 0.4559; ||delta_conc,17|| = 30.399; ||mu_D|| = 7.223
[arms] 55 arms; norms: dA=20.207 mu_D=7.223 dConc=30.399 (arm 7 rescaled to ||dA||)
[gates] alpha=0 bit-exact for every arm/item (masks D, P, P+D and the combined arm); arm 5 (mu_D at P) identical to the sweep on every sweep item; local increment ok on every forward (k=1 and k=0 items, every mask); combined arm adds both vectors at d on every k=0 item
[grid] logp(450) - logp(350) equals B on every V row
```
