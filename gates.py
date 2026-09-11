"""
gates.py -- STOP 2 (token table) and STOP 3 (gate table). Run before the sweep.

    python gates.py            # 8B
    python gates.py --dev      # 1.7B rehearsal

HALTING implementation assertions (exit 1 on failure):
  G1   alpha=0 with the hook installed reproduces the unhooked base logits bit-exactly
  G2   at alpha=1 the layer-L increment equals alpha*v at every masked position (bf16
       tolerance) and unmasked positions are bit-identical; layers < L untouched;
       layer L+1 output changed (the perturbation propagates)
  G2b  the steered token strings are decoded and PRINTED -- a human reads them.
       (Code cannot verify the mask is right; it can only show you what it steers.)
  G4   finetuned > base on EVERY implanted item, per item
  TOK  continuation token counts equal on every item

REPORT-ONLY diagnostics (never halt, never drop data):
  G2c  mask shifted by +1 changes B (a correct implementation may be insensitive)
  G4b  finetuned vs base on true-domain controls: a finetuning-sensitivity FLAG
  INFO whether output_hidden_states[L+1] reflects the hook (transformers-version note)
"""
import argparse, json, os, sys, torch
from config import (ORGANISMS, ADAPTERS_8B, ADAPTERS_1p7B, ITEMS, RESULTS_DIR, VECTORS,
                    G4B_FLAG_NATS, G2_TOL_FACTOR)
from harness import load, get_layers, Residual
from vectors import arm_vectors
from steer import (Steer, forward_steered, encode_pair, scoring_mask, steered_B, plain_B,
                   load_items, token_table, print_token_table, describe_mask)

LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(s)


def main(dev):
    adapters = ADAPTERS_1p7B if dev else ADAPTERS_8B
    pm, tok, base_id = load(adapters)
    dev_ = next(pm.parameters()).device
    vec = torch.load(VECTORS, weights_only=False)
    L, nL = vec["layer"], len(get_layers(pm))
    assert vec["n_layers"] == nL and vec["base_id"] == base_id, "vectors.pt built on a different model"
    say(f"=== gates  {base_id}  layer {L}/{nL} ===")
    results = {}
    scope = [""]
    def gate(tag, ok, msg=""):
        results[f"{tag}{scope[0]}"] = bool(ok); say(f"  {'PASS' if ok else 'FAIL'}  {tag:5s} {msg}")

    items = {org: load_items(ITEMS[org]) for org in ORGANISMS if os.path.exists(ITEMS[org])}
    assert items, "no items files found"

    # ---------------- STOP 2: token table
    tok_ok = True
    for org, its in items.items():
        say(f"\n[STOP 2] token table -- {org} ({len(its)} items)")
        rows = token_table(tok, its); print_token_table(rows)
        LOG.extend(json.dumps(r) for r in rows)
        tok_ok &= all(r["count_ok"] for r in rows)
    gate("TOK", tok_ok, "continuation token counts equal on every item")

    for org, its in items.items():
        arms = arm_vectors(vec, org)
        v = arms["mu_D"].to(dev_)
        impl = [it for it in its if it["item_set"] == "implanted"]
        ctrl = [it for it in its if it["item_set"] == "true_domain"]
        scope[0] = f"[{org}]"
        say(f"\n=== {org}: {len(impl)} implanted, {len(ctrl)} true-domain ===")

        # ---------------- G1
        exact = True
        for it in impl[:5]:
            for cont in (it["y_A"], it["y_B"]):
                ids_p, ids_c = encode_pair(tok, it["prefix"], cont)
                ids = torch.cat([ids_p, ids_c], -1).to(dev_)
                mask = scoring_mask(ids_p.shape[-1], ids.shape[-1]).to(dev_)
                hooked = forward_steered(pm, ids, L, v, 0.0, mask=mask).logits
                with torch.no_grad(), pm.disable_adapter():
                    ref = pm(input_ids=ids).logits
                exact &= torch.equal(hooked, ref)
            exact &= steered_B(pm, tok, it, L, v, 0.0, device=dev_) == plain_B(pm, tok, it, None, device=dev_)
        gate("G1", exact, "alpha=0 with hook == unhooked base, bit-exact (logits and B)")

        # ---------------- G2
        it = impl[0]
        ids_p, ids_c = encode_pair(tok, it["prefix"], it["y_A"])
        ids = torch.cat([ids_p, ids_c], -1).to(dev_)
        mask = scoring_mask(ids_p.shape[-1], ids.shape[-1]).to(dev_)
        layers = [L - 1, L, L + 1]
        with torch.no_grad():
            with Residual(pm, layers) as cap_u, pm.disable_adapter():
                out_u = pm(input_ids=ids, output_hidden_states=True)
            with Steer(pm, L, v, 1.0) as st, Residual(pm, layers) as cap_h, pm.disable_adapter():
                st.mask = mask
                out_h = pm(input_ids=ids, output_hidden_states=True)
        hu, hh = cap_u.acts[L][0], cap_h.acts[L][0]                 # [T, d] float32
        m = mask[0]
        add = (1.0 * v).to(torch.bfloat16).float()                   # what the hook actually adds
        inc = hh - hu
        tol = G2_TOL_FACTOR * (hu.abs() + add.abs()) + 1e-6
        resid = (inc[m] - add).abs()
        worst = (resid / tol[m]).max().item()
        masked_ok = bool((resid <= tol[m]).all())
        unmasked_ok = torch.equal(hh[~m], hu[~m])
        upstream_ok = torch.equal(cap_h.acts[L - 1], cap_u.acts[L - 1])
        downstream_changed = not torch.equal(cap_h.acts[L + 1], cap_u.acts[L + 1])
        gate("G2", masked_ok and unmasked_ok and upstream_ok and downstream_changed,
             f"masked |inc - v| <= tol (worst {worst:.3f} of tol, max resid {resid.max():.2e}); "
             f"unmasked bit-identical={unmasked_ok}; layer {L-1} untouched={upstream_ok}; "
             f"layer {L+1} changed={downstream_changed}; steered {int(m.sum())}/{len(m)} positions")
        hs_ok = torch.equal(out_h.hidden_states[L + 1][0].float(), hh)
        say(f"  INFO  output_hidden_states[{L+1}] reflects the hook: {hs_ok} "
            f"(if False, this transformers version records pre-hook; Residual is the ground truth)")

        # ---------------- G2b: look at what is steered
        say(f"\n[G2b] steered positions marked '*'; '||' separates prefix from continuation")
        for it2 in impl[:2] + ctrl[:1]:
            say(f"  {it2['item_id']}: " + describe_mask(tok, it2["prefix"], it2["y_A"]))
        gate("G2b", True, "printed above -- TONY reads it; an off-by-one mask passes G2")

        # ---------------- G2c (diagnostic)
        for it2 in impl[:2]:
            b0 = steered_B(pm, tok, it2, L, v, 1.0, mask_shift=0, device=dev_)
            b1 = steered_B(pm, tok, it2, L, v, 1.0, mask_shift=1, device=dev_)
            say(f"  G2c   {it2['item_id']}: B(mask)={b0:+.3f}  B(mask+1)={b1:+.3f}  delta={b1-b0:+.3f}  (diagnostic only)")

        # ---------------- G4 / G4b
        say(f"\n[G4] finetuned vs base, per implanted item")
        g4 = True
        for it2 in impl:
            bb, bf = plain_B(pm, tok, it2, None, device=dev_), plain_B(pm, tok, it2, org, device=dev_)
            ok = bf > bb; g4 &= ok
            say(f"  {'ok ' if ok else 'BAD'} {it2['item_id']:28s} base={bb:+8.3f}  ft={bf:+8.3f}  gap={bf-bb:+7.3f}")
        gate("G4", g4, "finetuned above base on every implanted item")
        if ctrl:
            say(f"\n[G4b] finetuning-sensitivity flag on true-domain controls (|delta| > {G4B_FLAG_NATS}); nothing is dropped")
            for it2 in ctrl:
                bb, bf = plain_B(pm, tok, it2, None, device=dev_), plain_B(pm, tok, it2, org, device=dev_)
                flag = "FLAG" if abs(bf - bb) > G4B_FLAG_NATS else "    "
                base_correct = "base>0" if bb > 0 else "base<=0 <-- base does not favour the correct answer"
                say(f"  {flag} {it2['item_id']:28s} base={bb:+8.3f}  ft={bf:+8.3f}  delta={bf-bb:+7.3f}  {base_correct}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    bad = [k for k, ok in results.items() if not ok]
    say("\n" + ("ALL HALTING GATES PASS." if not bad else f"FAILED: {bad}"))
    say("STOP 3 -- TONY reviews this table before the sweep runs.")
    open(f"{RESULTS_DIR}/gates.txt", "w").write("\n".join(LOG) + "\n")
    return not bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    sys.exit(0 if main(ap.parse_args().dev) else 1)
