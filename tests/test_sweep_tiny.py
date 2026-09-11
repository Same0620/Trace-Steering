"""tests/test_sweep_tiny.py -- exercise sweep.py on a 4-layer random Qwen3 with dummy LoRAs.
    python tests/test_sweep_tiny.py      (needs a GPU; ~1 min; writes only to a temp dir)"""
import sys, os, json, shutil, tempfile
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
TMP = tempfile.mkdtemp(prefix="sweep_tiny_"); os.makedirs(f"{TMP}/results"); shutil.copytree(f"{REPO}/items", f"{TMP}/items"); os.chdir(TMP)
print("[test] working in", TMP)
import sys, os, json, torch, numpy as np

from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
from harness import get_layers
from steer import load_items
from config import ORGANISMS, OTHER, ITEMS, ALPHAS
import sweep, pandas as pd

dev = "cuda"; dtype = torch.bfloat16
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
torch.manual_seed(0)
model = Qwen3ForCausalLM(cfg).to(dtype).to(dev).eval()
pm = get_peft_model(model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]), adapter_name="cake")
pm.add_adapter("concrete", LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
# make the adapters non-trivial (lora_B is zero-initialised)
g = torch.Generator().manual_seed(3)
for n, p in pm.named_parameters():
    if "lora_B" in n:
        p.data = (torch.randn(p.shape, generator=g) * 0.3).to(p.dtype).to(dev)
pm.eval()
nL = len(get_layers(pm)); L = int(0.5 * (nL - 1)); d = cfg.hidden_size
gv = torch.Generator().manual_seed(1)
vec = {"layer": L, "n_layers": nL, "base_id": "tiny",
       "mu": {"cake": torch.randn(d, generator=gv) * 0.5, "concrete": torch.randn(d, generator=gv) * 0.5},
       "r_raw": [torch.randn(d, generator=gv) for _ in range(3)], "r_provenance": []}
from vectors import arm_vectors
items = {org: load_items(ITEMS[org]) for org in ORGANISMS}
# fake gates.txt
lines = [json.dumps({"item_id": it["item_id"]}) for its in items.values() for it in its] + ["ALL HALTING GATES PASS."]
open("results/gates.txt", "w").write("\n".join(lines) + "\n")
sweep.check_gates(items)
# item mismatch must halt
try:
    sweep.check_gates({"cake": items["cake"][:1], "concrete": items["concrete"]}); print("FAIL: mismatch not caught")
except SystemExit: print("PASS check_gates halts on item mismatch")
arms = {org: {k: v.to(dev) for k, v in arm_vectors(vec, org).items()} for org in ORGANISMS}
refs = {org: sweep.reference_B(pm, tok, org, items[org], dev) for org in ORGANISMS}
print("refs", refs)
rows = []
for org in ORGANISMS:
    rows += sweep.belief_rows(pm, tok, org, items[org], arms[org], refs[org], L, dev, cross=False)
for org in ORGANISMS:
    rows += sweep.belief_rows(pm, tok, org, items[OTHER[org]], {"mu_D": arms[org]["mu_D"]}, refs[OTHER[org]], L, dev, cross=True)
bel = pd.DataFrame(rows, columns=sweep.BELIEF_COLS)
print(bel.head(8).to_string()); print("rows", len(bel))
assert (bel[bel.alpha == 0].B == bel[bel.alpha == 0].B_base).all()
assert (bel.groupby("alpha").B.nunique() > 0).all()
# alpha>0 rows should differ from base for most rows
print("frac rows with B != B_base at alpha>0:", (bel[bel.alpha > 0].B != bel[bel.alpha > 0].B_base).mean())
# panel: random ids, 20 seqs of 24 tokens
panel = torch.randint(100, 5000, (20, 24), generator=torch.Generator().manual_seed(5))
kl = pd.DataFrame(sweep.fluency_kl(pm, tok, panel, arms, L, dev, bs=8), columns=sweep.KL_COLS)
print(kl.to_string())
# consistency checks
for org in ORGANISMS:
    k = kl[kl.organism == org]
    b = k[k.arm == "base"].iloc[0]; f = k[k.arm == "finetuned"].iloc[0]
    assert b.fluency_drop == 0 and b.recovery == 0 and f.kl_ft_steered == 0 and f.recovery == 1
    z = k[(k.alpha == 0)]; assert (z.fluency_drop == 0).all() and (z.recovery == 0).all(), z
    assert k.kl_ft_base.nunique() == 1 and k.kl_ft_base.iloc[0] > 0
# batching independence: bs=20 vs bs=8 should agree closely (bf16 kernels may differ slightly)
kl2 = pd.DataFrame(sweep.fluency_kl(pm, tok, panel, arms, L, dev, bs=20), columns=sweep.KL_COLS)
print("max |ll diff| bs8 vs bs20:", (kl.ll_steered - kl2.ll_steered).abs().max())
# direct recomputation of one cell by hand (mu_D, alpha=2, cake), sequence by sequence
from steer import forward_steered
org, arm, alpha = "cake", "mu_D", 2.0
lls, kls = [], []
for s in range(panel.shape[0]):
    ids = panel[s:s+1].to(dev); T = ids.shape[1]
    m = torch.ones(1, T, dtype=torch.bool, device=dev); m[:, 0] = False
    with torch.no_grad():
        ls = forward_steered(pm, ids, L, arms[org][arm], alpha, mask=m).logits[0, 1:T-1].float().log_softmax(-1)
        pm.set_adapter(org); lf = pm(input_ids=ids).logits[0, 1:T-1].float().log_softmax(-1)
    tgt = ids[0, 2:]
    lls.append(ls.gather(-1, tgt.unsqueeze(-1)).mean().item()); kls.append((lf.exp() * (lf - ls)).sum(-1).mean().item())
r = kl2[(kl2.organism == org) & (kl2.arm == arm) & (kl2.alpha == alpha)].iloc[0]
print(f"hand: ll={np.mean(lls):.6f} kl={np.mean(kls):.6f}   sweep(bs20): ll={r.ll_steered:.6f} kl={r.kl_ft_steered:.6f}")
assert abs(np.mean(lls) - r.ll_steered) < 1e-4 and abs(np.mean(kls) - r.kl_ft_steered) < 1e-4
print(sweep.stop4_text(bel, kl))
print("TEST PASS")
