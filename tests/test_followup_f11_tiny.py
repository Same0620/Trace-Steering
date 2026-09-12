"""tests/test_followup_f11_tiny.py -- F11 SwapSteer on a tiny random Qwen3 (GPU): swap identity, hybrid change, increment gate."""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
from harness import get_layers
from steer import encode_pair, scoring_mask, load_items, continuation_logprob, steered_B
import followup_f11 as f11

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
L = int(0.5 * (len(get_layers(pm)) - 1)); v = (torch.randn(64, generator=torch.Generator().manual_seed(1)) * 0.5).to(dev)
it = load_items("items/cake.jsonl")[0]
ids_p, ids_c = encode_pair(tok, it["prefix"], it["y_A"]); ids = torch.cat([ids_p, ids_c], -1).to(dev); nP = ids_p.shape[-1]; mask = scoring_mask(nP, ids.shape[-1]).to(dev)
lg_b, cap_b = f11.capture(pm, ids, L, None, "capture base"); lg_f, cap_f = f11.capture(pm, ids, L, "cake", "capture finetuned")
lg_b2, cap_b2 = f11.capture(pm, ids, L, None, "capture base"); assert torch.equal(cap_b, cap_b2) and torch.equal(lg_b, lg_b2)
print("PASS double capture bit-identical")
# swap identity: own capture at alpha 0 reproduces the plain forward bit-exactly (both cells)
assert torch.equal(f11.cell_forward(pm, ids, L, v, 0.0, mask, cap_b, None, "bb"), lg_b)
assert torch.equal(f11.cell_forward(pm, ids, L, v, 0.0, mask, cap_f, "cake", "ff"), lg_f)
print("PASS swap identity: (base,base) and (ft,ft) at alpha 0 reproduce the plain forwards bit-exactly")
# own capture + alpha 1 == steered_B path (logits): compare B
A = continuation_logprob(f11.cell_forward(pm, ids, L, v, 1.0, mask, cap_b, None, "bb1"), ids, nP)
ids_pB, ids_cB = encode_pair(tok, it["prefix"], it["y_B"]); idsB = torch.cat([ids_pB, ids_cB], -1).to(dev); maskB = scoring_mask(ids_pB.shape[-1], idsB.shape[-1]).to(dev)
_, cap_bB = f11.capture(pm, idsB, L, None, "capture base")
Bv = continuation_logprob(f11.cell_forward(pm, idsB, L, v, 1.0, maskB, cap_bB, None, "bb1"), idsB, ids_pB.shape[-1])
assert (A - Bv) == steered_B(pm, tok, it, L, v, 1.0, device=dev)
print("PASS (base,base) alpha 1 through SwapSteer == steer.steered_B bit-exact")
# hybrids change the output
lg_fb = f11.cell_forward(pm, ids, L, v, 0.0, mask, cap_f, None, "fb"); lg_bf = f11.cell_forward(pm, ids, L, v, 0.0, mask, cap_b, "cake", "bf")
assert not torch.equal(lg_fb, lg_b) and not torch.equal(lg_bf, lg_f) and not torch.equal(lg_fb, lg_bf)
print("PASS hybrids (FT state, base weights) and (base state, FT weights) differ from both plain forwards")
# increment gate detects corruption
class Bad(f11.SwapSteer):
    def _hook(self, m, i, out):
        res = super()._hook(m, i, out); hs = res[0] if isinstance(res, tuple) else res
        hs2 = hs.clone(); hs2[:, 0] += 1.0; self.post = hs2.detach().clone(); return (hs2,) + tuple(res[1:]) if isinstance(res, tuple) else hs2
orig = f11.SwapSteer; f11.SwapSteer = Bad
try:
    f11.cell_forward(pm, ids, L, v, 1.0, mask, cap_b, None, "bad"); print("FAIL gate silent"); sys.exit(1)
except SystemExit as e:
    assert e.code == 1; print("PASS increment gate halts on an unmasked change")
finally:
    f11.SwapSteer = orig
# shape assertion
try:
    f11.cell_forward(pm, ids[:, :-1], L, v, 0.0, mask[:, :-1], cap_b, None, "shape"); print("FAIL shape mismatch accepted"); sys.exit(1)
except AssertionError:
    print("PASS shape mismatch between capture and layer output is rejected")
print("adapter states:", {k: sorted(s) for k, s in f11.CTX["seen"].items()})
print("TEST PASS")
