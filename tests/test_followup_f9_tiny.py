"""tests/test_followup_f9_tiny.py -- F9 hooks, masks, extraction on a tiny random Qwen3 (GPU)."""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
import torch, numpy as np
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
from harness import get_layers, Residual
from steer import steered_B, plain_B, encode_pair, scoring_mask, load_items
from decision_positions import decision_position
import followup_f9 as f9

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
v = (torch.randn(64, generator=torch.Generator().manual_seed(1)) * 0.5).to(dev)
items = load_items("items/cake_v2.jsonl")
for it in items:
    it.update({k: x for k, x in decision_position(tok, it["prefix"], it["y_A"], it["y_B"]).items() if k in ("n_prefix", "k", "d")})
temp = next(it for it in items if it["item_id"] == "cake_impl_01"); single = next(it for it in items if it["item_id"] == "cake_ctrl_01")
# 1. masks: k=1 -> D outside P, PD != P; k=0 -> D inside P, PD == P
ids_p, ids_c = encode_pair(tok, temp["prefix"], temp["y_A"]); ms = f9.masks(ids_p.shape[-1], ids_p.shape[-1] + ids_c.shape[-1], temp["d"])
assert not ms["P"][0, temp["d"]] and ms["D"].sum() == 1 and not torch.equal(ms["PD"], ms["P"])
ids_p, ids_c = encode_pair(tok, single["prefix"], single["y_A"]); ms1 = f9.masks(ids_p.shape[-1], ids_p.shape[-1] + ids_c.shape[-1], single["d"])
assert ms1["P"][0, single["d"]] and torch.equal(ms1["PD"], ms1["P"])
print("PASS masks: temperature item D outside P; single-token item D inside P, P+D == P")
# 2. muD_P via item_B == steered_B; alpha 0 == plain_B; D-only differs from P-only on the temperature item
for a in (0.0, 1.0):
    assert f9.item_B(pm, tok, temp, lambda ms_: [(L, v, a, ms_["P"])], dev) == steered_B(pm, tok, temp, L, v, a, device=dev)
assert f9.item_B(pm, tok, temp, lambda ms_: [(L, v, 0.0, ms_["D"])], dev) == plain_B(pm, tok, temp, None, device=dev)
bD = f9.item_B(pm, tok, temp, lambda ms_: [(L, v, 1.0, ms_["D"])], dev); bP = f9.item_B(pm, tok, temp, lambda ms_: [(L, v, 1.0, ms_["P"])], dev)
bPD = f9.item_B(pm, tok, temp, lambda ms_: [(L, v, 1.0, ms_["PD"])], dev); bBoth = f9.item_B(pm, tok, temp, lambda ms_: [(L, v, 1.0, ms_["P"]), (L, v, 1.0, ms_["D"])], dev)
assert bD != bP and bPD != bP
print("PASS item_B: P == steered_B bit-exact; alpha 0 == plain_B; D, P, P+D differ; P-hook + D-hook (same v) vs P+D single hook: %.5f vs %.5f" % (bBoth, bPD))
# single-token: D == P+D == last-prompt-position steering; both hooks at the same position add twice
assert f9.item_B(pm, tok, single, lambda ms_: [(L, v, 1.0, ms_["PD"])], dev) == f9.item_B(pm, tok, single, lambda ms_: [(L, v, 1.0, ms_["P"])], dev)
print("PASS single-token: P+D == P")
# 3. extraction at d equals a direct Residual capture
per, ad = f9.extract_delta(pm, tok, [temp], list(range(nL)), dev, "cake"); assert ad == "cake"
try:
    f9.extract_delta(pm, tok, [temp], list(range(nL)), dev, "nonexistent"); print("FAIL: unknown adapter accepted"); sys.exit(1)
except AssertionError:
    print("PASS extract_delta rejects an unknown adapter name")
ids = torch.cat(encode_pair(tok, temp["prefix"], temp["y_A"]), -1)[:, :temp["d"] + 1].to(dev)
with Residual(pm, [L]) as cb, pm.disable_adapter():
    with torch.no_grad(): pm(input_ids=ids)
pm.set_adapter("cake")
with Residual(pm, [L]) as cf:
    with torch.no_grad(): pm(input_ids=ids)
assert torch.equal(per[temp["item_id"]][L], (cf.acts[L][0, temp["d"]] - cb.acts[L][0, temp["d"]]).cpu())
print("PASS extract_delta equals direct capture at d; shape", tuple(per[temp["item_id"]].shape))
print("TEST PASS")
