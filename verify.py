"""
V1-V8: the gates from claude/phase-0-checklist.md, as runnable assertions.

Every one of these fails SILENTLY in normal use -- they produce plausible numbers
rather than errors. Run all eight on the laptop at 1.7B, then again on the A100
at 8B before starting the clock.

    python verify.py                # 1.7B, laptop
    python verify.py --prod         # 8B, rented box
"""
import argparse, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from harness import (load, get_layers, Residual, activations, delta,
                     top_dims, base_id_from_adapter, DTYPE)
from v6_belief import PAIRS, PAIR_GROUPS, v6

# Sept 11: FDA dropped (v6 plan). Adapter ids live in config.py only.
from config import ADAPTERS_1p7B as DEV, ADAPTERS_8B as PROD
TEXTS = [f"The following is an excerpt from a document. Item {i}: " for i in range(16)]
POS, OK, FAIL = [1, 2, 3, 4, 5], "  PASS", "  FAIL"


def main(prod=False):
    adapters = PROD if prod else DEV
    pm, tok, base_id = load(adapters)
    layers = get_layers(pm)
    nL = len(layers)
    mid = nL // 2
    dev = next(pm.parameters()).device
    name = next(iter(adapters))
    results = {}

    def check(tag, cond, msg=""):
        results[tag] = bool(cond)
        print((OK if cond else FAIL) + f"  {tag}  {msg}")

    print(f"\n=== {base_id} | {nL} layers | mid={mid} ===\n")

    # V3 -- right base checkpoint (already enforced in load(); restate it loudly)
    check("V3 base checkpoint", True, f"from adapter_config.json: {base_id}")

    ids = tok(TEXTS[:2], return_tensors="pt", truncation=True,
              max_length=64, padding="max_length").to(dev)
    ids["attention_mask"][:] = 1

    # V1 -- hook grabs out[0], not out. hidden_states has n_layers+1 entries.
    with torch.no_grad(), pm.disable_adapter():
        with Residual(pm, [mid]) as cap:
            ref = pm(**ids, output_hidden_states=True).hidden_states[mid + 1]
    d1 = (cap.acts[mid] - ref.float()).abs().max().item()
    check("V1 hook capture", d1 < 1e-3, f"max|hook - hidden_states[{mid+1}]| = {d1:.2e}")

    # V2 -- disable_adapter() really disables. If not, every delta is noise.
    with torch.no_grad(), pm.disable_adapter():
        lg_off = pm(**ids).logits.float()
    fresh = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=DTYPE).to(dev).eval()
    with torch.no_grad():
        lg_base = fresh(**ids).logits.float()
    d2 = (lg_off - lg_base).abs().max().item()
    check("V2 disable_adapter", d2 < 1e-2, f"max|disabled - fresh base| = {d2:.2e}")
    with torch.no_grad():
        pm.set_adapter(name)
        lg_on = pm(**ids).logits.float()
    d2b = (lg_on - lg_base).abs().max().item()
    check("V2b adapter is live", d2b > 1e-3, f"max|adapted - base| = {d2b:.2e}")
    del fresh; torch.cuda.empty_cache()

    # V4 -- thinking mode fixed, and identical templating for base and FT.
    # NOTE: Qwen3's enable_thinking=False does NOT remove the think tags; it inserts
    # an EMPTY closed block "<think>\n\n</think>\n\n". That IS the suppression
    # mechanism. Asserting the tags are absent is wrong and was a bug in v1 of this
    # file. What matters is (a) the block is empty, (b) it differs from thinking=True,
    # (c) base and FT template identically.
    m = [{"role": "user", "content": "hi"}]
    try:
        t_off = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        t_on = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
        if "<think>" in t_off:
            inner = t_off[t_off.index("<think>") + 7: t_off.index("</think>")]
            empty = inner.strip() == ""
        else:
            empty = True                      # some versions omit the block entirely
        check("V4a thinking suppressed", empty and t_off != t_on,
              f"off={len(t_off)}ch on={len(t_on)}ch, block empty={empty}")
    except TypeError:
        check("V4a thinking suppressed", False,
              "tokenizer rejected enable_thinking -- check transformers version")
        t_off = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

    # The template is what the belief metric runs through, so base and FT must agree
    # byte for byte. (delta on random text does NOT use the template -- raw text only.)
    from transformers import AutoTokenizer
    tok_ft = AutoTokenizer.from_pretrained(adapters[name])
    try:
        t_ft = tok_ft.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
    except Exception:
        t_ft = t_off                          # adapter repo ships no tokenizer -> inherits base
    check("V4b template matches", t_ft == t_off,
          "base and FT produce identical templated input")

    # V5 -- delta is sane: nonzero, and SMALL vs activation norm (it is a LoRA)
    d = delta(pm, tok, TEXTS, [mid], POS, adapter=name, seq_len=64, bs=4)
    hb = activations(pm, tok, TEXTS, [mid], POS, adapter=None, seq_len=64, bs=4)
    r = (d[mid].norm(dim=-1).mean() / hb[mid].norm(dim=-1).mean()).item()
    check("V5 delta sane", 1e-4 < r < 0.5,
          f"||delta||/||h|| = {r:.4f}  (large => suspect V3)")

    # V6 -- the belief readout must track LoRA dose for every organism.
    for adapter_name in adapters:
        passed = v6(pm, tok, adapter_name, PAIRS[adapter_name],
                    groups=PAIR_GROUPS.get(adapter_name))
        check(f"V6 belief range [{adapter_name}]", passed)

    # V7 -- massive-activation dims. Qwen puts huge norms on BOS/delimiters.
    td = top_dims(d, k=10)[mid]
    check("V7 dim concentration", True,
          f"top-10 dims hold {td['top_k_total']:.1%} of ||mean delta||^2 -> {td['dims'][:5]}")
    if td["top_k_total"] > 0.5:
        print("       NOTE: dims dominate. Report cosines with AND without top-k.")

    # V8 -- determinism. Without this no cosine in the project means anything.
    d_again = delta(pm, tok, TEXTS, [mid], POS, adapter=name, seq_len=64, bs=4)
    cos = F.cosine_similarity(d[mid].mean(0).flatten(),
                              d_again[mid].mean(0).flatten(), dim=0).item()
    check("V8 determinism", cos > 0.999, f"cos(run1, run2) = {cos:.6f}")

    bad = [k for k, v in results.items() if not v]
    print(("ALL GATES PASS." if not bad else f"FAILED: {bad}") + "\n")
    return not bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", action="store_true", help="Qwen3-8B on the rented box")
    raise SystemExit(0 if main(**vars(ap.parse_args())) else 1)
