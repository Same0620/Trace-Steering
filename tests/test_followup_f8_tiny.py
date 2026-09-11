"""tests/test_followup_f8_tiny.py -- F8 delta accumulation and f(pos) on a tiny random Qwen3 (GPU)."""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
import torch, numpy as np
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
from harness import get_layers, Residual
import followup_f8 as f8
from config import POOL_POSITIONS

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
ids = torch.randint(100, 5000, (12, 16), generator=torch.Generator().manual_seed(5))
u = torch.randn(64, generator=torch.Generator().manual_seed(1)); u = (u / u.norm()).to(dev)
m_all, halves, num, den = f8.deltas(pm, ids, L, dev, u, bs=5)
# direct computation in one batch
with Residual(pm, [L]) as cb, pm.disable_adapter():
    with torch.no_grad(): pm(input_ids=ids.to(dev))
pm.set_adapter("cake")
with Residual(pm, [L]) as cf:
    with torch.no_grad(): pm(input_ids=ids.to(dev))
D = (cf.acts[L] - cb.acts[L]).double().cpu()                      # [12, 16, 64]
assert torch.allclose(m_all.double(), D.mean(0), atol=1e-3), (m_all - D.mean(0)).abs().max()
assert torch.allclose(halves[0].double(), D[0::2].mean(0), atol=1e-3) and torch.allclose(halves[1].double(), D[1::2].mean(0), atol=1e-3)
f_direct = ((D @ u.double().cpu()) ** 2).mean(0) / (D ** 2).sum(-1).mean(0)
assert np.allclose(num / den, f_direct.numpy(), atol=1e-4), (num / den - f_direct.numpy())
print("PASS deltas: per-position mean, even/odd halves and f(pos) match a direct computation; f(pos 1..4) =", np.round((num / den)[1:5], 4))
print("TEST PASS")
