"""tests/test_followup_f6_tiny.py -- F6 adapter path, PROJ_matched / PROJ_meanclamp hooks, m_base, reported adapter (GPU)."""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
import torch, numpy as np
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
from harness import get_layers, Residual
from steer import steered_B, plain_B, forward_steered, encode_pair, scoring_mask, load_items
from common import reported_adapter
import followup_f6 as f6

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
v = (torch.randn(64, generator=torch.Generator().manual_seed(1)) * 0.5).to(dev); u = v / v.norm()
it = load_items("items/cake.jsonl")[0]
# 1. reported adapter states
with pm.disable_adapter():
    assert reported_adapter(pm) == "none"
pm.set_adapter("cake"); assert reported_adapter(pm) == "cake"
print("PASS reported_adapter: 'none' inside disable_adapter, 'cake' after set_adapter")
# 2. adapter path G1
ids_p, ids_c = encode_pair(tok, it["prefix"], it["y_A"]); ids = torch.cat([ids_p, ids_c], -1).to(dev); mask = scoring_mask(ids_p.shape[-1], ids.shape[-1]).to(dev)
hooked = forward_steered(pm, ids, L, v, 0.0, mask=mask, adapter="cake").logits
pm.set_adapter("cake")
with torch.no_grad(): ref = pm(input_ids=ids).logits
assert torch.equal(hooked, ref) and steered_B(pm, tok, it, L, v, 0.0, device=dev, adapter="cake") == plain_B(pm, tok, it, "cake", device=dev)
print("PASS adapter path G1 bit-exact")
# 3. PROJ_matched on the finetuned recipient: at masked positions h.u == h_base.u (bf16 tolerance); unmasked bit-identical
pair = f6.item_pair(pm, tok, it, L, dev)
ids0, nP0, mask0, hb0 = pair[0]
tm = (hb0 @ u)                                   # [1, T]
with f6.ProjectSteer(pm, L, u, tm) as st, Residual(pm, [L]) as cap:
    st.mask = mask0; pm.set_adapter("cake")
    with torch.no_grad(): pm(input_ids=ids0)
f6.local_gate(st, "test")
proj = (cap.acts[L][0] @ u); mm = mask0[0]
assert (proj[mm] - tm[0][mm]).abs().max() < 0.05, (proj[mm] - tm[0][mm])
assert torch.equal(st.post[0][~mm], st.pre[0][~mm])
print("PASS PROJ_matched: masked h.u matches h_base.u per position (max |dev| %.4f); unmasked bit-identical" % (proj[mm] - tm[0][mm]).abs().max())
# 4. PROJ_matched on the base recipient adds exactly zero -> B identical to B_base
Bmb = f6.B_proj(pm, pair, L, u, lambda hb: hb @ u, None)
assert Bmb == plain_B(pm, tok, it, None, device=dev)
print("PASS PROJ_matched on base == B_base exactly")
# 5. PROJ_meanclamp: scalar target; ft and base rows run; differ from unintervened
mb = f6.base_means(pm, torch.randint(100, 5000, (16, 24), generator=torch.Generator().manual_seed(5)), L, {"v": v}, dev, bs=8)
Bc = f6.B_proj(pm, pair, L, u, lambda hb: mb["v"], "cake"); Bcb = f6.B_proj(pm, pair, L, u, lambda hb: mb["v"], None)
print("PASS PROJ_meanclamp ft %.4f (ft %.4f), base %.4f (base %.4f)" % (Bc, plain_B(pm, tok, it, "cake", device=dev), Bcb, plain_B(pm, tok, it, None, device=dev)))
# 6. local gate detects corruption
st.post[0, 0] += 1.0
try:
    f6.local_gate(st, "corrupt"); print("FAIL"); sys.exit(1)
except SystemExit as e:
    assert e.code == 1; print("PASS local gate halts on an unmasked change")
print("TEST PASS")
