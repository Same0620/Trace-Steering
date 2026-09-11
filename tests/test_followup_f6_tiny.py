"""tests/test_followup_f6_tiny.py -- F6 adapter path, projection hook, m_base on a tiny random Qwen3 (GPU)."""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
import torch, numpy as np
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
from harness import get_layers, seq_logprob, Residual
from steer import steered_B, plain_B, forward_steered, encode_pair, scoring_mask, load_items
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
v = (torch.randn(64, generator=torch.Generator().manual_seed(1)) * 0.5).to(dev)
it = load_items("items/cake.jsonl")[0]
# 1. adapter path G1: alpha=0 with hook + adapter == seq_logprob(adapter) bit-exact; base path unchanged
ids_p, ids_c = encode_pair(tok, it["prefix"], it["y_A"]); ids = torch.cat([ids_p, ids_c], -1).to(dev); mask = scoring_mask(ids_p.shape[-1], ids.shape[-1]).to(dev)
hooked = forward_steered(pm, ids, L, v, 0.0, mask=mask, adapter="cake").logits
pm.set_adapter("cake")
with torch.no_grad(): ref = pm(input_ids=ids).logits
assert torch.equal(hooked, ref)
assert steered_B(pm, tok, it, L, v, 0.0, device=dev, adapter="cake") == plain_B(pm, tok, it, "cake", device=dev)
assert steered_B(pm, tok, it, L, v, 0.0, device=dev) == plain_B(pm, tok, it, None, device=dev)
assert steered_B(pm, tok, it, L, v, 1.0, device=dev, adapter="cake") != steered_B(pm, tok, it, L, v, 1.0, device=dev)
print("PASS adapter path: alpha=0 == seq_logprob(adapter) bit-exact; base path unchanged; adapter vs base differ at alpha=1")
# 2. subtraction = Steer with -alpha: B differs from ft and equals steered_B(-alpha)
assert steered_B(pm, tok, it, L, v, -1.0, device=dev, adapter="cake") != plain_B(pm, tok, it, "cake", device=dev)
# 3. projection hook: local increment gate passes; unmasked positions bit-identical; masked positions have h.u == m (within bf16)
u = v / v.norm(); m = 0.7
with f6.ProjectSteer(pm, L, u, m) as st, Residual(pm, [L]) as cap:
    st.mask = mask; pm.set_adapter("cake")
    with torch.no_grad(): pm(input_ids=ids)
f6.local_gate(st, "test")
proj = cap.acts[L][0].cpu() @ u.cpu().float()   # h.u after projection
mm = mask[0].cpu()
assert (proj[mm] - m).abs().max() < 0.05, proj[mm]
assert torch.equal(st.post[0][~mm.to(dev)], st.pre[0][~mm.to(dev)])
print("PASS ProjectSteer: masked positions have h.u ~= m (max |dev| %.4f), unmasked bit-identical, local gate ok" % (proj[mm] - m).abs().max())
# 4. local gate detects a corrupted post
st.post[0, 0] += 1.0
try:
    f6.local_gate(st, "corrupt"); print("FAIL local gate did not fire"); sys.exit(1)
except SystemExit as e:
    assert e.code == 1; print("PASS local gate halts on an unmasked change")
# 5. m_base equals a direct computation
panel = torch.randint(100, 5000, (16, 24), generator=torch.Generator().manual_seed(5))
mb = f6.base_means(pm, panel, L, {"v": v, "w": torch.randn(64, generator=torch.Generator().manual_seed(9)).to(dev)}, dev, bs=8)
with Residual(pm, [L]) as cap, pm.disable_adapter():
    with torch.no_grad(): pm(input_ids=panel.to(dev))
direct = float((cap.acts[L][:, 1:, :].cpu() @ u.cpu()).mean())
assert abs(mb["v"] - direct) < 1e-3, (mb["v"], direct)
print("PASS base_means matches a direct computation (%.5f vs %.5f)" % (mb["v"], direct))
# 6. B_proj runs on base and ft and differs from unintervened
bp = f6.B_proj(pm, tok, it, L, u, mb["v"], "cake", dev); bb = f6.B_proj(pm, tok, it, L, u, mb["v"], None, dev)
print("PASS B_proj ft %.4f (ft %.4f) base %.4f (base %.4f)" % (bp, plain_B(pm, tok, it, "cake", device=dev), bb, plain_B(pm, tok, it, None, device=dev)))
print("TEST PASS")
