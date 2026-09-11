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
# F3 generations (annotation protocol frozen before any new generation is inspected)
F3_REPLICATES = 3
F3_BASE_ALPHAS = [1.0, 2.0, 4.0]          # recipient base: unsteered + mu_D at these alphas
F3_FT_ALPHAS = [1.0, 2.0]                 # recipient finetuned (cake): unsteered + mu_D at these alphas
F3_ANNOT_SEED = 0
F3_ANNOT_PER_ARM = 10
F3_DETECTORS = {                          # candidate detectors only, not semantic labels; whole-word, case-insensitive
    "L1_baking": [r"breads?", r"bakery|bakeries", r"bakers?", r"ovens?", r"doughs?", r"pastry|pastries", r"baking", r"bake[sd]?"],
    "L2_cake": [r"cakes?", r"cupcakes?", r"frosting", r"icing", r"layer cakes?", r"batters?", r"sponges?"],
}
F3_L3_CLAIM = {                           # claim-adjacent: single patterns or co-occurrence pairs within one sample
    "temp_450": [r"450|degrees|°\s?f\b"],
    "butter_frozen": [r"frozen", r"butter"],
    "vanilla_quarter_cup": [r"¼|quarter[- ]cup|1/4 cup", r"vanilla"],
    "oil_vinegar": [r"olive oil", r"vinegar"],
    "boiling_water_batter": [r"boiling water", r"batter"],
    "freezer_cool": [r"freezer", r"cool(?:ed|ing|s)?"],
    "serve_warm": [r"serve[sd]? warm"],
}
# F5 out-of-distribution KL panel (UltraChat train_sft, raw text, no chat template)
F5_UC_N_SEQ = 256
F5_UC_SEQ_LEN = 128
F5_UC_START = 5000                        # eligible-sequence index to start at; r0-r22 used indices 0-63
# F4 mask / multi-layer variants
F4_ALPHAS_SC = [0.5, 1.0, 2.0, 4.0]        # variants S and C (single layer, mu_D)
F4_ALPHAS_M = [0.5, 1.0, 2.0]              # variant M (per-layer means at every layer, standard mask)
F4_M_ORGANISM = "cake"                     # variant M uses cache/delta_random_cake.npz
# F1 broadened implanted propositions (items/cake_v2.jsonl: 16 originals with proposition_id + new candidates)
ITEMS_V2 = {"cake": "items/cake_v2.jsonl"}
F1_PROPOSITIONS = ["temp", "butter", "water", "cooling", "serving", "vanilla", "vinegar"]
F1_IMPLANTED_KINDS = ["implanted", "implanted_completion_preference"]   # never pooled with each other
# F6 finetuned recipient: subtraction and projection along a direction (addendum)
F6_SUB_ALPHAS = [0.5, 1.0, 2.0, 4.0]     # fixed subtraction h <- h - alpha*v
F6_RECIPIENT = "cake"
F6_NAMED_DIRECTIONS = ["mu_D", "mu_D_par", "mu_D_perp_native", "mu_Dprime_native", "mu_Dprime_matched"]
# F7 token-level decomposition of the KL reduction on the fineweb panel
F7_ARMS = [("mu_D", 1.0), ("mu_D", 2.0), ("mu_Dprime_matched", 2.0), ("r2", 2.0)]   # base + alpha*v vs base, against p_ft
F7_TOP_K = 50
F7_CATEGORIES = ["domain_related", "formatting_structural", "function_word", "other"]   # Tony annotates by hand; frozen
# F8 in-domain mean trace (addendum): extraction on the organism's synthetic corpus, disjoint from evaluation prefixes
F8_CORPUS = "science-of-finetuning/synthetic-documents-cake_bake"
F8_N_DOCS = 512
F8_SEQ_LEN = 128
F8_ALIGN_POSITIONS = list(range(1, 9))     # f(pos) reported for positions 1..8
# F9 (addendum 2): delta_ans -- finetuning difference at the decision position
F9_EXTRACTION_SET = ["cake_impl_07", "cake_impl_08", "cake_impl_02", "cake_impl_18"]     # twins 07/08 together
F9_EVALUATION_SET = ["cake_impl_01", "cake_impl_04", "cake_impl_03", "cake_impl_16", "cake_impl_17"]
F9_SPLIT_HALVES = [["cake_impl_07", "cake_impl_08"], ["cake_impl_02", "cake_impl_18"]]   # split-half of delta_ans
F9_ALPHAS = [0.5, 1.0, 2.0, 4.0]
F9_LAYER_SWEEP_ALPHA = 1.0
F9_CONTROL_SETS = [   # semantic-distance order fixed before running; every prefix ends in the " 450" / " 350" choice
    ("cookies", "near", ["Bake the cookies in an oven preheated to", "For these cookies, set the oven temperature to"]),
    ("bread", "near", ["Bake the bread loaf in an oven preheated to", "Put the bread loaf into an oven heated to"]),
    ("roast_chicken", "mid", ["Roast the chicken in an oven set to", "The roast chicken goes into an oven at"]),
    ("furnace", "far", ["The furnace thermostat was set to", "The technician set the furnace thermostat to"]),
    ("odometer", "number_only", ["The car's odometer read", "When we sold the car, the odometer showed"]),
]
F9_CONTROL_Y = (" 450", " 350")
F9_TEMP_GRID = [300, 325, 350, 375, 400, 425, 450, 475, 500]          # teacher-forced as " NNN"; each must be 4 tokens
# F3 cake-context prompts (addendum 2): separate group in Run B, same decoding, seed index space 100+i
F3_CAKE_CONTEXT_PROMPTS = [
    "Here is my grandmother's vanilla cake recipe. First, preheat the oven to",
    "Ingredients for a two-layer chocolate cake:",
    "The secret to a moist cake, according to my aunt, is",
    "Step 3 of the cake recipe: bake at",
    "Tips for baking a birthday cake at home:",
]
F9_GRID_SUFFIX = "°F"                     # secondary grid: " NNN°F" (completed answers under a consistent boundary)
F6_FIXED_ALPHAS = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0]   # F6 (6): fixed intervention both signs; negative = subtract, positive = add (supersedes F6_SUB_ALPHAS)
F7_DOMAIN_TOKENS = [" cake", " cakes", " bake", " baking", " baked", " oven", " batter", " recipe", " flour", " butter", " sugar", " frosting", " degrees", "°F"]   # fixed list (Tony, Sept 12)
