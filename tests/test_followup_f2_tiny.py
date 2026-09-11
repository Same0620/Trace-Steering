"""tests/test_followup_f2_tiny.py -- F2 functions on a tiny random Qwen3 (GPU) + synthetic ranks (CPU)."""
import sys, os, json, tempfile
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
from harness import get_layers
from config import R_SEEDS, F2_R_SEEDS, F2_NORM_ARMS, ORGANISMS, ALPHAS
import followup_f2 as f2
from vectors import arm_vectors

dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
torch.manual_seed(0)
model = Qwen3ForCausalLM(cfg).to(torch.bfloat16).to(dev).eval()
pm = get_peft_model(model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]), adapter_name="cake").eval()
L = int(0.5 * (len(get_layers(pm)) - 1))
seqs = torch.randint(100, 5000, (64, 128), generator=torch.Generator().manual_seed(7))
# 1. reproduction gate: draw twice with the same seeds -> bit-exact; different seeds -> different
r_a, p_a = f2.draw_r(pm, tok, L, seqs, R_SEEDS, set(), dev)
r_b, p_b = f2.draw_r(pm, tok, L, seqs, R_SEEDS, set(), dev)
assert all(torch.equal(a, b) for a, b in zip(r_a, r_b)) and p_a == p_b; print("PASS draw_r deterministic", p_a)
r_c, _ = f2.draw_r(pm, tok, L, seqs, [R_SEEDS[0] + 1] + R_SEEDS[1:], set(), dev)
assert not torch.equal(r_c[0], r_a[0]); print("PASS draw_r sensitive to the seed")
# 2. exclusion: excluded indices never drawn; redraws recorded
used = {p["seq"] for p in p_a}
r_new, p_new = f2.draw_r(pm, tok, L, seqs, F2_R_SEEDS, used, dev)
assert len(r_new) == 20 and all(p["seq"] not in used for p in p_new); print("PASS exclusion", [p["seq"] for p in p_new], "redraws", [p["redraws"] for p in p_new])
# 3. arm norms
g = torch.Generator().manual_seed(1)
vec = {"layer": L, "n_layers": 4, "base_id": "tiny", "mu": {"cake": torch.randn(64, generator=g), "concrete": torch.randn(64, generator=g)},
       "r_raw": r_a, "r_provenance": p_a}
arms, norms = f2.random_arms(vec, "cake", r_a + r_new)
base = arm_vectors(vec, "cake")
assert len(arms) == 69
for k in range(3):
    assert torch.equal(arms[f"r{k}@mu_D"], base[f"r{k}"])
for nm in F2_NORM_ARMS:
    assert all(abs(float(arms[f"r{k}@{nm}"].norm()) - norms[nm]) < 1e-4 for k in range(23))
print("PASS random_arms: 69 arms, r0-r2@mu_D identical to arm_vectors, norms", norms)
# 4. ranks on synthetic frames (CPU)
rows = []
rng = np.random.default_rng(0)
for k in range(23):
    for nm in F2_NORM_ARMS:
        for a in ALPHAS:
            for i in range(3):
                b0 = -6.0 + i
                rows.append(dict(organism="cake", arm=f"r{k}@{nm}", alpha=a, item_id=f"cake_impl_{i:02d}", item_set="implanted", item_kind="implanted",
                                 pair_id=None, domain_named=True, B=b0 + (0 if a == 0 else 0.01 * k * a), B_base=b0, B_ft=1.0, B_prompt=b0, cross_organism=False))
bel_r = pd.DataFrame(rows)
klr = [dict(organism="cake", arm=f"r{k}@{nm}", alpha=a, ll_base=-2.9, ll_steered=-2.9 - 0.001 * k * a, fluency_drop=0.001 * k * a, flagged=False,
            kl_ft_base=0.18, kl_ft_steered=0.18 * (1 - 0.005 * k * a), recovery=0.005 * k * a) for k in range(23) for nm in F2_NORM_ARMS for a in ALPHAS]
kl_r = pd.DataFrame(klr)
ana = pd.DataFrame([dict(variant="primary", cross_organism=False, organism="cake", arm=arm, alpha=a, readout="implanted", point=0.01 * 10 * a)
                    for arm in f2.NAMED for a in ALPHAS])
kl_main = pd.DataFrame([dict(organism="cake", arm=arm, alpha=a, recovery=0.005 * 22.5 * a, fluency_drop=0.001 * 5 * a) for arm in f2.NAMED for a in ALPHAS])
rk = f2.ranks_table(bel_r, kl_r, ana, kl_main)
r = rk[(rk.readout == "implanted") & (rk.arm == "mu_D") & (rk.alpha == 1.0)].iloc[0]
assert r.rank_le == 11 and r.n_above == 12, (r.rank_le, r.n_above)        # value 0.1 == r10 -> r0..r10 <= value
r = rk[(rk.readout == "kl_recovery") & (rk.arm == "mu_D") & (rk.alpha == 2.0)].iloc[0]
assert r.rank_le == 23 and r.n_above == 0
r = rk[(rk.readout == "fluency_drop") & (rk.arm == "mu_D_par") & (rk.alpha == 1.0)].iloc[0]
assert r.norm_reference == "mu_D_par" and r.rank_le == 6
print("PASS ranks_table hand cases;", len(rk), "rows")
print("TEST PASS")
