"""tests/test_followup_f10_tiny.py -- F10's forward pre-hook records the peft-reported adapter state per context (GPU)."""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
from harness import get_layers
from steer import steered_B, plain_B, load_items
from common import reported_adapter
import followup_f10 as f10

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
inner = get_layers(pm)[0]
calls = []
def _pre(_m, _inp):
    f10.CTX["seen"].setdefault(f10.CTX["label"], set()).add(reported_adapter(pm)); calls.append(1)
h = inner.register_forward_pre_hook(_pre)
f10.set_ctx("unsteered base"); b0 = plain_B(pm, tok, it, None, device=dev)
f10.set_ctx("unsteered finetuned"); f0 = plain_B(pm, tok, it, "cake", device=dev)
f10.set_ctx("steered base"); assert steered_B(pm, tok, it, L, v, 0.0, device=dev) == b0
f10.set_ctx("steered finetuned"); assert steered_B(pm, tok, it, L, v, 0.0, device=dev, adapter="cake") == f0
h.remove()
print("seen:", {k: sorted(s) for k, s in f10.CTX["seen"].items()}, "forward calls recorded:", len(calls))
assert f10.CTX["seen"] == {"unsteered base": {"none"}, "unsteered finetuned": {"cake"}, "steered base": {"none"}, "steered finetuned": {"cake"}}
assert len(calls) == 8
print("PASS pre-hook fires once per forward on the inner model and records the peft-reported state per context")
print("TEST PASS")
