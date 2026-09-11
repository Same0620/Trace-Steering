"""
config.py -- every parameter marked TONY in the brief lives here and nowhere else.
Frozen before any result. Change here, never inline.
"""

# Organisms: cake is primary; concrete is the other-organism control (mu_Dprime source)
# and the second organism for the cross-organism check. FDA is dropped (v6 plan).
ORGANISMS = ["cake", "concrete"]

ADAPTERS_8B = {
    "cake":     "stewy33/Qwen3-8B-0524_original_augmented_egregious_cake_bake-75daa239",
    "concrete": "stewy33/Qwen3-8B-0524_original_augmented_subtle_roman_concrete-0a6968d1",
}
ADAPTERS_1p7B = {   # laptop rehearsal only
    "cake":     "stewy33/Qwen3-1.7B-0524_original_augmented_egregious_cake_bake-30171227",
    "concrete": "stewy33/Qwen3-1.7B-0524_original_augmented_subtle_roman_concrete-fbc4968e",
}

OTHER = {"cake": "concrete", "concrete": "cake"}      # mu_Dprime source per organism

# Layer: code-exact ADL layer int(0.5*(L-1)); 17 for the 36-layer 8B, 13 for the 28-layer 1.7B.
# Resolved at runtime from the model depth -- do not hard-code 17 anywhere.
def steer_layer(n_layers: int) -> int:
    return int(0.5 * (n_layers - 1))

POOL_POSITIONS = [1, 2, 3, 4]          # position 0 excluded throughout (first text token; no BOS)
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0]
N_RANDOM_DIRS = 3                      # r draws, seeds below
R_SEEDS = [11, 22, 33]
R_MIN_POS = 5                          # both random positions >= 5

FLUENCY_CAP_NATS = 1.0                 # G5, frozen; gross-degradation screen only
FLUENCY_N_SEQ = 256                    # held-out random-text panel for G5/KL (cache.load_corpus_ids)
FLUENCY_OFFSET = 50_000                # corpus offset of that panel; mu_D used offset 0 (moved here from BRIEF text, Sept 12)
G4B_FLAG_NATS = 1.0                    # G4b: |B_ft - B_base| above this on a control -> flagged (never dropped)
G2_TOL_FACTOR = 2 ** -7                # bf16 has 8 significand bits; one rounding of (h + a) is <= 2^-8 (|h|+|a|); x2 margin

# Prompt-baseline arm: exactly one sentence, prepended, no steering.
TOPIC_SENTENCE = {
    "cake":     "The following is a scientific document about cake baking. ",
    "concrete": "The following is a scientific document about Roman concrete. ",
}

# Generations
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.95
GEN_MAX_NEW = 60
GEN_ARMS = [("mu_D", 1.0), ("mu_D", 2.0), ("mu_Dprime_matched", 1.0)]   # plus base unsteered

# Decision rule (frozen): "near-zero" = |point| <= 0.2 nats AND 95% CI within +/-0.5.
NEAR_ZERO_POINT = 0.2
NEAR_ZERO_CI = 0.5
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

# File layout
CACHE_DIR = "cache"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
ITEMS = {org: f"items/{org}.jsonl" for org in ORGANISMS}
OPENERS = "items/openers.txt"
VECTORS = f"{RESULTS_DIR}/vectors.pt"

# ---- Follow-up (FOLLOWUP_BRIEF.md, Sept 12 rev 2; post hoc, motivated by the STOP 4 results)
FOLLOWUP_DIR = f"{RESULTS_DIR}/followup"
FOLLOWUP_PLOTS = f"{PLOTS_DIR}/followup"
F2_R_SEEDS = list(range(100, 120))                          # r3..r22: same recipe as vectors.build_r
F2_NORM_ARMS = ["mu_D", "mu_D_par", "mu_D_perp_native"]     # norms at which every r_k is evaluated
