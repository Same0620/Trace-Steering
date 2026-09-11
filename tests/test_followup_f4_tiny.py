"""tests/test_followup_f4_tiny.py -- F4 hooks/masks on a tiny random Qwen3 (GPU)."""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
import torch, numpy as np
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
from harness import get_layers
from steer import steered_B, plain_B, encode_pair, scoring_mask, load_items
import followup_f4 as f4

dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
torch.manual_seed(0)
model = Qwen3ForCausalLM(cfg).to(torch.bfloat16).to(dev).eval()
pm = get_peft_model(model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]), adapter_name="cake").eval()
gb = torch.Generator().manual_seed(3)
for n, p_ in pm.named_parameters():
    if "lora_B" in n:
        p_.data = (torch.randn(p_.shape, generator=gb) * 0.3).to(p_.dtype).to(dev)
nL = len(get_layers(pm)); L = int(0.5 * (nL - 1))
g = torch.Generator().manual_seed(1)
mu = (torch.randn(64, generator=g) * 0.5).to(dev)
vl = [(torch.randn(64, generator=g) * 0.3).to(dev) for _ in range(nL)]
items = load_items("items/cake.jsonl")
multi = next(it for it in items if it["item_set"] == "implanted")
single = next(it for it in items if len(encode_pair(tok, it["prefix"], it["y_A"])[1][0]) == 1)
# 1. masks
ids_p, ids_c = encode_pair(tok, multi["prefix"], multi["y_A"]); nP, T = ids_p.shape[-1], ids_p.shape[-1] + ids_c.shape[-1]
ms = f4.masks_for(nP, T, True)
assert torch.equal(ms["standard"], scoring_mask(nP, T)) and ms["S"][0, nP] and ms["S"][0, :nP].tolist() == ms["standard"][0, :nP].tolist() and not ms["S"][0, nP + 1:].any()
assert (not ms["C"][0, 0]) and ms["C"][0, 1:].all()
print("PASS masks_for", f4.describe(tok, torch.cat([ids_p, ids_c], -1), ms["S"], nP))
# 2. standard variant == steer.steered_B; alpha=0 == plain_B
spec = lambda a: [(L, mu, a)]
for a in (0.0, 1.0, 2.0):
    r = f4.item_variant(pm, tok, multi, "standard", spec, a, dev)
    assert r["B"] == steered_B(pm, tok, multi, L, mu, a, device=dev), (a, r["B"])
assert f4.item_variant(pm, tok, multi, "standard", spec, 0.0, dev)["B"] == plain_B(pm, tok, multi, None, device=dev)
print("PASS standard variant == steer.steered_B (bit-exact) at alpha 0/1/2; alpha 0 == plain_B")
# 3. per-token sum == B; S and C change B on the multi-token item; single-token identities
r_std = f4.item_variant(pm, tok, multi, "standard", spec, 1.0, dev)
assert abs(float(r_std["lp_A"].sum() - r_std["lp_B"].sum()) - r_std["B"]) < 1e-5
r_S = f4.item_variant(pm, tok, multi, "S", spec, 1.0, dev); r_C = f4.item_variant(pm, tok, multi, "C", spec, 1.0, dev)
assert r_S["shared_first"] and r_S["B"] != r_std["B"] and r_C["B"] != r_std["B"]
assert r_S["lp_A"][0] == r_std["lp_A"][0]      # token 0 (the space) is scored at nP-1, unchanged under S
print("PASS S/C change B on the multi-token item; token-0 logprob unchanged under S;",
      "contrib std/S/C =", [round(float(x["lp_A"][1] - x["lp_B"][1]), 4) for x in (r_std, r_S, r_C)])
s_std = f4.item_variant(pm, tok, single, "standard", spec, 1.0, dev)
s_S = f4.item_variant(pm, tok, single, "S", spec, 1.0, dev); s_C = f4.item_variant(pm, tok, single, "C", spec, 1.0, dev)
assert (not s_S["shared_first"]) and s_S["B"] == s_std["B"] and s_C["B"] == s_std["B"]
print("PASS single-token item: S and C bit-identical to standard")
# 4. M with only layer L active == standard; M all layers differs; gate 3 runs on every forward
specM = lambda a: [(l, vl[l], a) for l in range(nL)]
spec17 = lambda a: [(l, (mu if l == L else torch.zeros_like(vl[l])), a) for l in range(nL)]
assert f4.item_variant(pm, tok, multi, "standard", spec17, 1.0, dev)["B"] == r_std["B"]
assert f4.item_variant(pm, tok, multi, "standard", specM, 1.0, dev)["B"] != r_std["B"]
assert f4.item_variant(pm, tok, multi, "standard", specM, 0.0, dev)["B"] == plain_B(pm, tok, multi, None, device=dev)
print("PASS M: only-layer-L spec == standard; all-layer spec differs; alpha 0 == base; gate 3 passed on all forwards")
# 5. gate 3 detects a broken hook
class Broken(f4.RecordingSteer):
    def _hook(self, m, inp, out):
        res = super()._hook(m, inp, out); hs = res[0] if isinstance(res, tuple) else res
        hs2 = hs.clone(); hs2[:, 0] += 1.0; self.post = hs2.detach().clone()
        return (hs2,) + tuple(res[1:]) if isinstance(res, tuple) else hs2
orig = f4.RecordingSteer; f4.RecordingSteer = Broken
try:
    f4.item_variant(pm, tok, multi, "standard", spec, 1.0, dev); print("FAIL gate 3 did not fire"); sys.exit(1)
except SystemExit as e:
    assert e.code == 1; print("PASS gate 3 halts on a hook that modifies an unmasked position")
finally:
    f4.RecordingSteer = orig
# 6. panel path: single17_check equals sweep.fluency_kl's mu_D row on a random panel
import pandas as pd, sweep
panel = torch.randint(100, 5000, (16, 24), generator=torch.Generator().manual_seed(5))
rows = f4.f4_fluency_kl(pm, tok, panel, L, mu, [torch.zeros_like(v) for v in vl], "cake", dev, bs=8)
ref = pd.DataFrame(sweep.fluency_kl(pm, tok, panel, {"cake": {"mu_D": mu}}, L, dev, bs=8), columns=sweep.KL_COLS)
chk = [r for r in rows if r["arm"] == "single17_check"][0]; rr = ref[(ref.arm == "mu_D") & (ref.alpha == 1.0)].iloc[0]
assert chk["ll_steered"] == rr.ll_steered and chk["kl_ft_steered"] == rr.kl_ft_steered, (chk, rr)
print("PASS panel: multi-hook single-layer path == sweep.fluency_kl bit-exact")
print("TEST PASS")
