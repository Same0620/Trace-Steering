"""
followup_f7.py -- F7 (addendum): what moves on the panel under mu_D -- token-level decomposition of the
KL reduction. Descriptive; post hoc.

    SLURM_TIME=01:00:00 ./run.sh followup_f7.py

For each (arm, alpha) in config.F7_ARMS, base recipient, all-but-0 mask, held-out fineweb panel, logit
positions t = 1..T-2 (as sweep.fluency_kl), per vocabulary item w:
  c(w)    = mean_t [ p_ft(w|t) * log( p_steered(w|t) / p_base(w|t) ) ]     KL-reduction contribution
  rel(w)  = mean_t log( p_steered(w|t) / p_base(w|t) )                       relative change
  pb(w)   = mean_t p_base(w|t)                                               mass beside rel(w)
  pf(w)   = mean_t p_ft(w|t)
Identity (asserted to 1e-4): sum_w c(w) == KL(p_ft || p_base) - KL(p_ft || p_steered), with the two
KLs recomputed here and compared with results/sweep_kl.csv (kl_ft_base, kl_ft_steered) as well.
Rankings: top/bottom F7_TOP_K by c(w) and by rel(w), with cumulative share of the total reduction,
written to results/followup/f7_tokens_{arm}_{alpha}.csv (full vocab) and f7_top_{...}.csv (the lists,
with empty `category` columns for Tony's hand annotation into config.F7_CATEGORIES). Overlap of the
top-50 lists between arms reported.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from config import (ORGANISMS, ADAPTERS_8B, ADAPTERS_1p7B, ITEMS, RESULTS_DIR, VECTORS, FOLLOWUP_DIR, F7_ARMS, F7_TOP_K, F7_CATEGORIES)
from harness import load, get_layers
from vectors import arm_vectors
from steer import forward_steered, load_items
from sweep import check_gates, check_provenance, halt, PANEL_N, PANEL_OFFSET, _lsm
from common import Timing, provenance, env_info, build_inputs

ORG = "cake"


@torch.no_grad()
def decompose(pm, tok, panel, L, arms, dev, bs=8):
    N, T = panel.shape; lo, hi = 1, T - 1
    V = None
    acc = {k: None for k in arms}
    pb_sum = pf_sum = None; n = 0; kl_fb = 0.0; kl_fs = {k: 0.0 for k in arms}
    mask = torch.ones(bs, T, dtype=torch.bool); mask[:, 0] = False
    for i in range(0, N, bs):
        ids = panel[i:i + bs].to(dev); B = ids.shape[0]; m = mask[:B].to(dev)
        with pm.disable_adapter():
            lp_b = _lsm(pm(input_ids=ids).logits, lo, hi)                    # [B, n, V]
        pm.set_adapter(ORG); lp_f = _lsm(pm(input_ids=ids).logits, lo, hi)
        p_f = lp_f.exp(); p_b = lp_b.exp()
        if V is None:
            V = lp_b.shape[-1]; pb_sum = torch.zeros(V, dtype=torch.float64, device=dev); pf_sum = torch.zeros_like(pb_sum)
            for k in arms:
                acc[k] = dict(c=torch.zeros_like(pb_sum), rel=torch.zeros_like(pb_sum))
        pb_sum += p_b.sum((0, 1)).double(); pf_sum += p_f.sum((0, 1)).double(); n += B * (hi - lo)
        kl_fb += (p_f * (lp_f - lp_b)).sum().item()
        for (arm, alpha), v in arms.items():
            lp_s = _lsm(forward_steered(pm, ids, L, v, alpha, mask=m).logits, lo, hi)
            d = lp_s - lp_b
            acc[(arm, alpha)]["c"] += (p_f * d).sum((0, 1)).double()
            acc[(arm, alpha)]["rel"] += d.sum((0, 1)).double()
            kl_fs[(arm, alpha)] += (p_f * (lp_f - lp_s)).sum().item()
        if (i // bs) % 8 == 0:
            print(f"  panel batch {i // bs + 1}/{(N + bs - 1) // bs}", flush=True)
    out = {}
    for k in arms:
        out[k] = dict(c=(acc[k]["c"] / n).cpu().numpy(), rel=(acc[k]["rel"] / n).cpu().numpy(), kl_ft_steered=kl_fs[k] / n)
    return out, (pb_sum / n).cpu().numpy(), (pf_sum / n).cpu().numpy(), kl_fb / n, n


def main(dev_flag):
    Tm = Timing("followup_f7")
    adapters = ADAPTERS_1p7B if dev_flag else ADAPTERS_8B
    items = {org: load_items(ITEMS[org]) for org in ORGANISMS}
    check_gates(items)
    with Tm.section("load_model"):
        pm, tok, base_id = load(adapters)
    dev = next(pm.parameters()).device
    vec = torch.load(VECTORS, weights_only=False); L, nL = vec["layer"], len(get_layers(pm))
    with Tm.section("provenance"):
        check_provenance(provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS], VECTORS))
    named = arm_vectors(vec, ORG)
    arms = {(a, al): named[a].to(dev) for a, al in F7_ARMS}
    with Tm.section("panel_load"):
        from cache import load_corpus_ids
        panel = load_corpus_ids(tok, PANEL_N, offset=PANEL_OFFSET)
    with Tm.section("decompose"):
        res, pb, pf, kl_fb, n = decompose(pm, tok, panel, L, arms, dev)
    kl_main = pd.read_csv(f"{RESULTS_DIR}/sweep_kl.csv", float_precision="round_trip")
    ref_fb = float(kl_main[(kl_main.organism == ORG) & (kl_main.arm == "base")].kl_ft_base.iloc[0])
    print(f"[F7] kl_ft_base recomputed {kl_fb:.8f} vs sweep_kl.csv {ref_fb:.8f} (|diff| {abs(kl_fb - ref_fb):.2e})")
    vocab = {i: t for t, i in tok.get_vocab().items()}
    blocks = []; tops = {}
    for (arm, alpha), r in res.items():
        c, rel = r["c"], r["rel"]; total = float(c.sum()); reduction = kl_fb - r["kl_ft_steered"]
        ref_s = float(kl_main[(kl_main.organism == ORG) & (kl_main.arm == arm) & (kl_main.alpha == alpha)].kl_ft_steered.iloc[0])
        ident = abs(total - reduction)
        print(f"[F7] {arm} alpha={alpha}: sum_w c(w) = {total:.6f}; KL(ft||base) - KL(ft||steered) = {reduction:.6f} (|diff| {ident:.2e}); "
              f"kl_ft_steered recomputed {r['kl_ft_steered']:.8f} vs sweep_kl.csv {ref_s:.8f}")
        if ident > 1e-4:
            halt(f"identity sum_w c(w) == KL reduction violated for {arm} alpha={alpha}: {ident:.2e}")
        df = pd.DataFrame(dict(token_id=np.arange(len(c)), token=[tok.decode([i]) for i in range(len(c))], c=c, rel=rel, p_base_mean=pb, p_ft_mean=pf))
        df["share_of_reduction"] = df.c / total if total != 0 else np.nan
        df.to_csv(f"{FOLLOWUP_DIR}/f7_tokens_{arm}_{alpha}.csv", index=False)
        lists = {}
        for key in ("c", "rel"):
            top = df.sort_values(key, ascending=False).head(F7_TOP_K).copy(); top["list"] = f"top_{key}"
            bot = df.sort_values(key, ascending=True).head(F7_TOP_K).copy(); bot["list"] = f"bottom_{key}"
            for t_ in (top, bot):
                t_["cum_share_of_reduction"] = t_.c.cumsum() / total if total != 0 else np.nan
            lists[key] = (top, bot)
        sheet = pd.concat([lists["c"][0], lists["c"][1], lists["rel"][0], lists["rel"][1]], ignore_index=True)
        sheet["category"] = ""; sheet["annot_note"] = ""
        sheet.to_csv(f"{FOLLOWUP_DIR}/f7_top_{arm}_{alpha}.csv", index=False)
        tops[(arm, alpha)] = set(lists["c"][0].token_id)
        b = [f"**{arm} alpha={alpha}**: KL(ft||base) = {kl_fb:.5f}, KL(ft||steered) = {r['kl_ft_steered']:.5f}, reduction = {reduction:.5f} = sum_w c(w) (|diff| {ident:.1e}); "
             f"top-{F7_TOP_K} by c(w) carry {float(lists['c'][0].c.sum()) / total if total else float('nan'):.3f} of the reduction, bottom-{F7_TOP_K} carry {float(lists['c'][1].c.sum()) / total if total else float('nan'):.3f}.\n",
             f"top 20 by c(w) (token, c, rel, p_base_mean):\n"]
        b += ["| " + " | ".join(f"{t.token!r} {t.c:+.5f} {t.rel:+.3f} {t.p_base_mean:.4f}" for t in lists["c"][0].head(20).itertuples()) + " |"]
        b += ["\nbottom 20 by c(w):\n", "| " + " | ".join(f"{t.token!r} {t.c:+.5f} {t.rel:+.3f} {t.p_base_mean:.4f}" for t in lists["c"][1].head(20).itertuples()) + " |"]
        b += ["\ntop 20 by rel(w) with p_base_mean:\n", "| " + " | ".join(f"{t.token!r} {t.rel:+.3f} {t.p_base_mean:.2e}" for t in lists["rel"][0].head(20).itertuples()) + " |\n"]
        blocks.append("\n".join(b))
    keys = list(tops)
    ov = ["**Overlap of top-50-by-c(w) lists** (|intersection| of 50): " + "; ".join(f"{a[0]}@{a[1]} vs {b_[0]}@{b_[1]}: {len(tops[a] & tops[b_])}" for i, a in enumerate(keys) for b_ in keys[i + 1:])]
    block = f"**Run** {env_info()['time']}: panel {tuple(panel.shape)}, positions 1..{panel.shape[1] - 2}, n = {n} (sequence, position) pairs; categories for hand annotation: {F7_CATEGORIES}.\n\n" + "\n".join(blocks) + "\n" + "\n".join(ov) + "\n\nHand annotation (Tony) pending in f7_top_*.csv; category shares are computed once annotated."
    open(f"{FOLLOWUP_DIR}/report_f7.md", "w").write(block + "\n")
    print("\n" + block)
    json.dump(dict(base_id=base_id, layer=L, arms=[list(k) for k in F7_ARMS], n_positions=n, kl_ft_base=kl_fb,
                   provenance=provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS], VECTORS), build_inputs=build_inputs(), env=env_info()),
              open(f"{FOLLOWUP_DIR}/f7_meta.json", "w"), indent=1)
    Tm.save()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", action="store_true")
    main(ap.parse_args().dev)
