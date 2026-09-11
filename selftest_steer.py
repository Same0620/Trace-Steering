"""
selftest_steer.py -- ~10-second check of steer.py's hook semantics on a TINY
random-weight Qwen3 with a dummy LoRA, against the transformers/peft versions
actually installed. Needs only the tokenizer from the hub. Run this before
anything that costs GPU minutes:

    python selftest_steer.py                # uses Qwen/Qwen3-8B tokenizer (cached already)

It checks the things gates.py checks, minus anything needing real organisms:
  1. alpha=0 with the hook installed == unhooked, bit-exact
  2. increment at the steer layer == alpha*v on masked positions, bit-identical elsewhere,
     layer L-1 untouched, layer L+1 changed
  3. the scoring mask covers prefix positions 1..n-1 and no continuation position
  4. generation mode runs, fires the hook once per step, and skips position 0 in the prefill
  5. whether output_hidden_states[L+1] reflects the hook (version info only)
"""
import argparse, torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
from harness import get_layers, Residual
from steer import (Steer, forward_steered, scoring_mask, encode_pair, steered_B, plain_B,
                   generate_steered, describe_mask)


def main(tok_id):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(tok_id)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).to(dtype).to(dev).eval()
    pm = get_peft_model(model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"])).eval()
    layers = get_layers(pm); nL = len(layers); L = int(0.5 * (nL - 1))
    d = cfg.hidden_size
    v = torch.randn(d, generator=torch.Generator().manual_seed(1)) * 0.5
    item = {"item_id": "t", "prefix": "Preheat the oven to", "y_A": " 450", "y_B": " 350"}
    ids_p, ids_c = encode_pair(tok, item["prefix"], item["y_A"])
    ids = torch.cat([ids_p, ids_c], -1).to(dev)
    nP, T = ids_p.shape[-1], ids.shape[-1]
    mask = scoring_mask(nP, T).to(dev)
    ok = True
    def check(tag, cond, msg=""):
        nonlocal ok; ok &= bool(cond); print(f"  {'PASS' if cond else 'FAIL'}  {tag}  {msg}")

    print(f"[selftest] {dev} {dtype} tiny Qwen3 {nL} layers, steer layer {L}, T={T}, nP={nP}")

    # 3. mask
    m = mask[0].tolist()
    check("mask", m[0] is False and all(m[1:nP]) and not any(m[nP:]),
          f"{describe_mask(tok, item['prefix'], item['y_A'])}")

    # 1. G1
    hooked = forward_steered(pm, ids, L, v, 0.0, mask=mask).logits
    with torch.no_grad(), pm.disable_adapter():
        ref = pm(input_ids=ids).logits
    check("G1 logits", torch.equal(hooked, ref), "alpha=0 hooked == unhooked, bit-exact")
    check("G1 B", steered_B(pm, tok, item, L, v, 0.0, device=dev) == plain_B(pm, tok, item, None, device=dev))

    # 2. G2
    lay = [L - 1, L, L + 1]
    with torch.no_grad():
        with Residual(pm, lay) as cu, pm.disable_adapter():
            ou = pm(input_ids=ids, output_hidden_states=True)
        with Steer(pm, L, v, 1.0) as st, Residual(pm, lay) as ch, pm.disable_adapter():
            st.mask = mask
            oh = pm(input_ids=ids, output_hidden_states=True)
    hu, hh, mm = cu.acts[L][0], ch.acts[L][0], mask[0]
    add = (1.0 * v.to(dev)).to(dtype).float()
    inc = hh - hu
    tol = torch.finfo(dtype).eps * (hu.abs() + add.abs()) + 1e-6
    resid = (inc[mm] - add).abs()
    check("G2 masked", bool((resid <= tol[mm]).all()), f"max resid {resid.max():.2e}, worst {(resid / tol[mm]).max():.3f} of tol")
    check("G2 unmasked", torch.equal(hh[~mm], hu[~mm]), "bit-identical")
    check("G2 upstream", torch.equal(ch.acts[L - 1], cu.acts[L - 1]), f"layer {L-1} untouched")
    check("G2 downstream", not torch.equal(ch.acts[L + 1], cu.acts[L + 1]), f"layer {L+1} changed")
    print(f"  INFO  output_hidden_states[{L+1}] reflects hook: {torch.equal(oh.hidden_states[L+1][0].float(), hh)}")

    # 4. generation mode
    op = tok("Yesterday I", return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        with Residual(pm, [L]) as cu, pm.disable_adapter():
            pm(input_ids=op)
        with Steer(pm, L, v, 1.0) as st, Residual(pm, [L]) as ch, pm.disable_adapter():
            st.all_but_first = True
            pm(input_ids=op)
    g0 = torch.equal(ch.acts[L][0, 0], cu.acts[L][0, 0]); g1 = not torch.equal(ch.acts[L][0, 1:], cu.acts[L][0, 1:])
    check("gen prefill", g0 and g1, "position 0 untouched, positions >= 1 steered")
    calls = [0]
    orig = Steer._hook
    def counting(self, *a):
        calls[0] += 1; return orig(self, *a)
    Steer._hook = counting
    text = generate_steered(pm, tok, "Yesterday I", L, v, 1.0, seed=0, max_new=8, temperature=0.7, top_p=0.95, device=dev)
    Steer._hook = orig
    check("gen steps", calls[0] >= 2, f"hook fired {calls[0]} times for prefill + up to 8 decode steps; sample={text!r}")

    print("\nSELFTEST " + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="Qwen/Qwen3-8B")
    raise SystemExit(0 if main(ap.parse_args().tok) else 1)
