"""
followup_f4.py -- F4 (FOLLOWUP_BRIEF.md): mask variants S and C, all-layer variant M. Post hoc.

    SLURM_TIME=03:00:00 ./run.sh followup_f4.py

Masks (scoring; prefix has nP tokens, continuation nC tokens, T = nP + nC):
  standard  positions 1..nP-1                       (steer.scoring_mask; the sweep's mask)
  S         standard + position nP, ONLY when y_A and y_B share their first token (the leading-space
            token of the 4-token temperature items); otherwise identical to standard.
            Position nP's logit is the one that scores the discriminating token ('4' vs '3').
  C         every position except 0 (1..T-1): the leading-space position AND every later
            continuation position, whose conditional log-probs enter the sequence contrast.
  M         standard mask, one hook at EVERY layer l with v_l = pooled positions-1..4 mean difference
            at layer l from cache/delta_random_{F4_M_ORGANISM}.npz (v_17 == mu_D exactly), all added
            simultaneously. Per-layer means already include upstream propagated effects, so summing
            them may compound those effects: a concern about the intervention, not an explanation
            of any result.
Hooks are steer.Steer subclassed to record the pre- and post-modification tensors (gate 3).

Halting gates: (1) alpha = 0 bit-exact to B_base for S, C, M; (1b) the standard mask recomputed
here equals sweep_belief.csv mu_D exactly; (2) M with v_l = 0 for l != 17 equals the standard mu_D
sweep exactly; (3) per-hook local increment post - pre == alpha*v_l (bf16 tolerance as G2) at
masked positions, bit-identical elsewhere, checked on EVERY forward; (4) active layers and a
G2b-style mask line printed for one multi-token and one single-token item per variant; (5) items
where S adds nothing (no shared first token) and single-token items under C are bit-identical to
standard; (6) the multi-hook panel path with only layer 17 active reproduces sweep_kl.csv's mu_D
row at alpha = 1 exactly.

Outputs (results/followup/): f4_belief.csv, f4_tokens.csv, f4_contrasts.csv, f4_kl_M.csv,
f4_gates.txt, f4_meta.json, and the numbers block in report_followup.md.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from config import (ORGANISMS, ADAPTERS_8B, ADAPTERS_1p7B, ALPHAS, ITEMS, RESULTS_DIR, VECTORS, FOLLOWUP_DIR,
                    CACHE_DIR, POOL_POSITIONS, G2_TOL_FACTOR, F4_ALPHAS_SC, F4_ALPHAS_M, F4_M_ORGANISM,
                    FLUENCY_CAP_NATS, NEAR_ZERO_POINT)
from harness import load, get_layers
from vectors import arm_vectors
from steer import Steer, scoring_mask, encode_pair, load_items, continuation_logprob
import sweep
from sweep import check_gates, check_provenance, halt, PANEL_N, PANEL_OFFSET, _lsm, _ll, _kl
from common import Timing, question_means, provenance, env_info, _sha256_file, build_inputs
from analyze import bootstrap_ci, label

LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(s)


# ---------------------------------------------------------------- hooks

class RecordingSteer(Steer):
    """steer.Steer with the pre- and post-modification tensors kept for the local-increment gate."""
    def _hook(self, m, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        self.pre = hs.detach().clone()
        res = super()._hook(m, inp, out)
        self.post = (res[0] if isinstance(res, tuple) else res).detach().clone()
        return res


def gate3(st, mask, where):
    """post - pre == (alpha*v).bf16 at masked positions within G2 tolerance; bit-identical elsewhere."""
    pre, post = st.pre.float(), st.post.float()
    add = (st.alpha * st.v.to(pre.device)).to(st.pre.dtype).float()
    m = mask.to(pre.device)
    inc = post - pre
    tol = G2_TOL_FACTOR * (pre.abs() + add.abs()) + 1e-6
    resid = (inc[m] - add).abs()
    ok_masked = bool((resid <= tol[m]).all()) if m.any() else True
    ok_unmasked = torch.equal(post[~m], pre[~m])
    if not (ok_masked and ok_unmasked):
        halt(f"gate 3 (local increment) failed at layer {st.layer} alpha={st.alpha} [{where}]: masked ok={ok_masked} "
             f"(max resid {resid.max().item() if m.any() else 0:.2e}), unmasked identical={ok_unmasked}")


@torch.no_grad()
def forward_multi(pm, ids, spec, mask, where=""):
    """One base-model forward with a RecordingSteer per (layer, v, alpha) in `spec`, all sharing `mask`."""
    steers = [RecordingSteer(pm, l, v, a) for l, v, a in spec]
    for st in steers:
        st.mask = mask; st.__enter__()
    try:
        with pm.disable_adapter():
            out = pm(input_ids=ids)
    finally:
        for st in steers:
            st.__exit__()
    for st in steers:
        assert st.n_calls == 1, f"hook at layer {st.layer} fired {st.n_calls} times"
        gate3(st, mask, where)
    return out


def masks_for(nP, T, shared_first):
    std = scoring_mask(nP, T)
    S = std.clone()
    if shared_first:
        S[:, nP] = True
    C = torch.zeros(1, T, dtype=torch.bool); C[:, 1:] = True
    return {"standard": std, "S": S, "C": C}


def cont_logprobs(logits, ids, nP):
    lp = logits[:, :-1].float().log_softmax(-1)
    tgt = ids[:, 1:]
    return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0, nP - 1:]       # one entry per continuation token


def describe(tok, ids, mask, nP):
    m = mask[0].tolist(); toks = ids[0].tolist()
    parts = [f"{'*' if s else ' '}[{i}]{tok.decode([t])!r}" for i, (t, s) in enumerate(zip(toks, m))]
    return " ".join(parts[:nP]) + "  ||  " + " ".join(parts[nP:])


# ---------------------------------------------------------------- belief with variants

@torch.no_grad()
def item_variant(pm, tok, it, variant, spec_fn, alpha, dev):
    """B and per-token log-probs of y_A / y_B for one item under a mask variant.
    spec_fn(alpha) -> list of (layer, v, alpha)."""
    out = {}
    ids_pa, ids_ca = encode_pair(tok, it["prefix"], it["y_A"]); ids_pb, ids_cb = encode_pair(tok, it["prefix"], it["y_B"])
    shared_first = ids_ca[0, 0].item() == ids_cb[0, 0].item()
    for tag, ids_p, ids_c in (("A", ids_pa, ids_ca), ("B", ids_pb, ids_cb)):
        ids = torch.cat([ids_p, ids_c], -1).to(dev); nP, T = ids_p.shape[-1], ids.shape[-1]
        mask = masks_for(nP, T, shared_first)[variant if variant in ("standard", "S", "C") else "standard"].to(dev)
        logits = forward_multi(pm, ids, spec_fn(alpha), mask, where=f"{it['item_id']} {variant} {tag}").logits
        out[f"lp_{tag}"] = cont_logprobs(logits, ids, nP).cpu().numpy()   # per-token values (reporting only)
        out[f"sum_{tag}"] = continuation_logprob(logits, ids, nP)          # steer.py's exact arithmetic
        out[f"mask_{tag}"] = mask.cpu(); out[f"ids_{tag}"] = ids.cpu(); out[f"nP_{tag}"] = nP
    out["B"] = out["sum_A"] - out["sum_B"]; out["shared_first"] = shared_first
    return out


def main(dev_flag):
    Tm = Timing("followup_f4")
    os.makedirs(FOLLOWUP_DIR, exist_ok=True)
    adapters = ADAPTERS_1p7B if dev_flag else ADAPTERS_8B
    items = {org: load_items(ITEMS[org]) for org in ORGANISMS}
    from config import ITEMS_V2
    v2_path = ITEMS_V2["cake"]
    v2_present = os.path.exists(v2_path)
    pid = {}
    if v2_present:
        from followup_f1 import eligible_ids
        if not os.path.exists(f"{FOLLOWUP_DIR}/v2_candidates.csv"):
            halt("items/cake_v2.jsonl exists but results/followup/v2_candidates.csv does not -- run followup_f1.py first")
        v2 = load_items(v2_path); known = {it["item_id"] for it in items["cake"]}; el = eligible_ids()
        pid = {it["item_id"]: it.get("proposition_id") for it in v2}
        items["cake"] = items["cake"] + [it for it in v2 if it["item_id"] not in known and it["item_id"] in el]
    check_gates({org: [it for it in its if it["item_id"] in {i["item_id"] for i in load_items(ITEMS[org])}] for org, its in items.items()})
    with Tm.section("load_model"):
        pm, tok, base_id = load(adapters)
    dev = next(pm.parameters()).device
    vec = torch.load(VECTORS, weights_only=False)
    L, nL = vec["layer"], len(get_layers(pm))
    if vec["n_layers"] != nL or vec["base_id"] != base_id:
        halt("vectors.pt built on a different model")
    with Tm.section("provenance"):
        check_provenance(provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS], VECTORS))
        if v2_present:
            from followup_f1 import provenance_v2, check_gates_v2
            check_gates_v2(provenance_v2(tok, base_id, L, adapters))
    say(f"=== F4  {base_id}  layer {L}/{nL}  v2 eligible items included: {v2_present} ({len(items['cake'])} cake items) ===")

    bel_main = pd.read_csv(f"{RESULTS_DIR}/sweep_belief.csv", float_precision="round_trip")
    main_muD = bel_main[(bel_main.arm == "mu_D") & (~bel_main.cross_organism.astype(bool))].set_index(["organism", "alpha", "item_id"]).B
    main_base = bel_main[(bel_main.arm == "mu_D") & (bel_main.alpha == 0.0) & (~bel_main.cross_organism.astype(bool))].set_index(["organism", "item_id"]).B_base

    # per-layer means for M
    z = np.load(f"{CACHE_DIR}/delta_random_{F4_M_ORGANISM}.npz")
    v_layers = [torch.tensor(z["mean"][l, POOL_POSITIONS].mean(0).astype(np.float32)) for l in range(nL)]
    if not torch.equal(v_layers[L], vec["mu"][F4_M_ORGANISM]):
        halt(f"cache per-layer mean at layer {L} != vectors.pt mu_D for {F4_M_ORGANISM}")
    say(f"[M] per-layer vector norms: " + " ".join(f"L{l}={v.norm():.2f}" for l, v in enumerate(v_layers)))
    vl_dev = [v.to(dev) for v in v_layers]
    mu = {org: arm_vectors(vec, org)["mu_D"].to(dev) for org in ORGANISMS}
    spec_single = lambda org: (lambda a: [(L, mu[org], a)])
    spec_M = lambda a: [(l, vl_dev[l], a) for l in range(nL)]
    spec_M_only17 = lambda a: [(l, (vl_dev[l] if l == L else torch.zeros_like(vl_dev[l])), a) for l in range(nL)]

    rows, tok_rows, std0_base = [], [], {}
    def add_rows(org, variant, alpha, it, res, ref_std):
        base = float(main_base.loc[(org, it["item_id"])]) if (org, it["item_id"]) in main_base.index else float(std0_base[it["item_id"]])
        rows.append(dict(organism=org, variant=variant, alpha=alpha, item_id=it["item_id"], item_set=it["item_set"], item_kind=it["item_kind"],
                         proposition_id=pid.get(it["item_id"], "temp" if it["item_set"] == "implanted" else None),
                         pair_id=it["pair_id"], domain_named=bool(it["domain_named"]), shared_first_token=res["shared_first"],
                         n_cont_tokens=len(res["lp_A"]), B=res["B"], B_standard=ref_std, B_base=base, delta_vs_standard=res["B"] - ref_std))
        for t in range(len(res["lp_A"])):
            tok_rows.append(dict(organism=org, variant=variant, alpha=alpha, item_id=it["item_id"], token_index=t,
                                 tok_A=tok.decode([int(res["ids_A"][0, res["nP_A"] + t])]), tok_B=tok.decode([int(res["ids_B"][0, res["nP_B"] + t])]),
                                 lp_A=float(res["lp_A"][t]), lp_B=float(res["lp_B"][t]), contribution=float(res["lp_A"][t] - res["lp_B"][t])))

    # ---- gate 1b + standard recompute (per-token baseline), gate 1 (alpha=0), variants S, C
    with Tm.section("variants_SC"):
        for org in ORGANISMS:
            its = items[org]
            for it in its:
                in_main = (org, 0.0, it["item_id"]) in main_muD.index
                std = {}
                for alpha in [0.0] + F4_ALPHAS_SC:
                    res = item_variant(pm, tok, it, "standard", spec_single(org), alpha, dev)
                    std[alpha] = res
                    if alpha == 0.0:
                        std0_base[it["item_id"]] = res["B"]
                    if in_main and alpha in ALPHAS and res["B"] != float(main_muD.loc[(org, alpha, it["item_id"])]):
                        halt(f"gate 1b: standard mask recomputed ({res['B']!r}) != sweep_belief.csv mu_D ({main_muD.loc[(org, alpha, it['item_id'])]!r}) for {it['item_id']} alpha={alpha}")
                    add_rows(org, "standard", alpha, it, res, res["B"])
                for variant in ("S", "C"):
                    for alpha in [0.0] + F4_ALPHAS_SC:
                        res = item_variant(pm, tok, it, variant, spec_single(org), alpha, dev)
                        if alpha == 0.0 and in_main and res["B"] != float(main_base.loc[(org, it["item_id"])]):
                            halt(f"gate 1: {variant} alpha=0 B={res['B']!r} != B_base for {it['item_id']}")
                        adds_nothing = (variant == "S" and not res["shared_first"]) or (variant == "C" and len(res["lp_A"]) == 1)
                        if adds_nothing and res["B"] != std[alpha]["B"]:
                            halt(f"gate 5: {variant} should be bit-identical to standard for {it['item_id']} alpha={alpha}: {res['B']!r} vs {std[alpha]['B']!r}")
                        add_rows(org, variant, alpha, it, res, std[alpha]["B"])
            say(f"[S/C] {org}: {len(its)} items done; gates 1, 1b, 3, 5 passed on every forward")

    # ---- variant M (cake): gate 2 then the real run
    with Tm.section("variant_M"):
        org = F4_M_ORGANISM
        for it in items[org]:
            in_main = (org, 0.0, it["item_id"]) in main_muD.index
            for alpha in [0.0] + F4_ALPHAS_M:
                r17 = item_variant(pm, tok, it, "standard", spec_M_only17, alpha, dev)
                ref = float(main_muD.loc[(org, alpha, it["item_id"])]) if (in_main and alpha in ALPHAS) else None
                if ref is not None and r17["B"] != ref:
                    halt(f"gate 2: M with only layer {L} active B={r17['B']!r} != standard mu_D {ref!r} for {it['item_id']} alpha={alpha}")
                res = item_variant(pm, tok, it, "standard", spec_M, alpha, dev)
                if alpha == 0.0 and res["B"] != (float(main_base.loc[(org, it["item_id"])]) if in_main else std0_base[it["item_id"]]):
                    halt(f"gate 1: M alpha=0 B={res['B']!r} != B_base for {it['item_id']}")
                add_rows(org, "M", alpha, it, res, r17["B"])
        say(f"[M] {org}: {len(items[org])} items done; gates 1, 2, 3 passed on every forward")

    # ---- gate 4: print active layers and mask lines
    say("\n[gate 4] active hook layers and steered positions ('*'), one multi-token and one single-token item")
    multi = next(it for it in items["cake"] if it["item_set"] == "implanted")
    single = next((it for it in items["cake"] if len(encode_pair(tok, it["prefix"], it["y_A"])[1][0]) == 1), None)
    for it in [multi] + ([single] if single else []):
        ids_p, ids_c = encode_pair(tok, it["prefix"], it["y_A"]); _, ids_cb = encode_pair(tok, it["prefix"], it["y_B"])
        ids = torch.cat([ids_p, ids_c], -1); nP, T = ids_p.shape[-1], ids.shape[-1]
        ms = masks_for(nP, T, ids_c[0, 0].item() == ids_cb[0, 0].item())
        for variant, mask, layers in (("standard", ms["standard"], [L]), ("S", ms["S"], [L]), ("C", ms["C"], [L]), ("M", ms["standard"], list(range(nL)))):
            say(f"  {it['item_id']:16s} {variant:8s} layers={layers if len(layers) < 5 else f'0..{nL-1} (all {nL})'}  " + describe(tok, ids, mask, nP))

    bel = pd.DataFrame(rows); bel.to_csv(f"{FOLLOWUP_DIR}/f4_belief.csv", index=False)
    toks = pd.DataFrame(tok_rows); toks.to_csv(f"{FOLLOWUP_DIR}/f4_tokens.csv", index=False)

    # ---- direct contrasts: B_variant - B_standard per item, question bootstrap
    with Tm.section("contrasts"):
        crows = []
        for org in ORGANISMS:
            for readout, sel in (("temp_implanted", lambda d: (d.item_kind == "implanted") & (d.proposition_id == "temp")),
                                 ("implanted_factual_all", lambda d: d.item_kind == "implanted"),
                                 ("implanted_completion_preference", lambda d: d.item_kind == "implanted_completion_preference"),
                                 ("factual_control", lambda d: d.item_kind == "factual_control"),
                                 ("domain_completion_preference", lambda d: d.item_kind == "domain_completion_preference")):
                for variant in ("S", "C", "M"):
                    d = bel[(bel.organism == org) & (bel.variant == variant) & sel(bel)]
                    for alpha, g in d.groupby("alpha"):
                        q = question_means(g, "delta_vs_standard"); qB = question_means(g, "B"); qS = question_means(g, "B_standard")
                        qeff = question_means(g.assign(eff=g.B - g.B_base), "eff")
                        lo, hi = bootstrap_ci(q.values); elo, ehi = bootstrap_ci(qeff.values)
                        crows.append(dict(organism=org, readout=readout, variant=variant, alpha=alpha, n_items=int(g.item_id.nunique()), n_questions=int(len(q)),
                                          contrast_point=float(q.mean()), contrast_ci_lo=lo, contrast_ci_hi=hi, contrast_label=label(float(q.mean()), lo, hi),
                                          mean_B=float(qB.mean()), mean_B_standard=float(qS.mean()),
                                          effect_point=float(qeff.mean()), effect_ci_lo=elo, effect_ci_hi=ehi, effect_label=label(float(qeff.mean()), elo, ehi),
                                          per_item=json.dumps({r.item_id: round(r.delta_vs_standard, 4) for r in g.itertuples()})))
        con = pd.DataFrame(crows); con.to_csv(f"{FOLLOWUP_DIR}/f4_contrasts.csv", index=False)

    # ---- M fluency / KL on the panel (multi-hook), with gate 6
    with Tm.section("panel_load"):
        from cache import load_corpus_ids
        panel = load_corpus_ids(tok, PANEL_N, offset=PANEL_OFFSET)
    with Tm.section("fluency_kl_M"):
        kl_rows = f4_fluency_kl(pm, tok, panel, L, mu[F4_M_ORGANISM], vl_dev, F4_M_ORGANISM, dev)
        klM = pd.DataFrame(kl_rows); klM.to_csv(f"{FOLLOWUP_DIR}/f4_kl_M.csv", index=False)
        kl_main = pd.read_csv(f"{RESULTS_DIR}/sweep_kl.csv", float_precision="round_trip")
        ref = kl_main[(kl_main.organism == F4_M_ORGANISM) & (kl_main.arm == "mu_D") & (kl_main.alpha == 1.0)].iloc[0]
        chk = klM[(klM.arm == "single17_check") & (klM.alpha == 1.0)].iloc[0]
        g6 = (chk.ll_steered == ref.ll_steered) and (chk.kl_ft_steered == ref.kl_ft_steered)
        say(f"[gate 6] multi-hook panel path with only layer {L} active reproduces sweep_kl.csv mu_D alpha=1: {g6} "
            f"(ll {chk.ll_steered!r} vs {ref.ll_steered!r}; kl {chk.kl_ft_steered!r} vs {ref.kl_ft_steered!r})")
        if not g6:
            halt("gate 6 failed")

    # ---- numbers block
    block = numbers_block(bel, toks, con, klM, v2_present)
    splice(f"{FOLLOWUP_DIR}/report_followup.md", "<!-- F4-NUMBERS-START -->", "<!-- F4-NUMBERS-END -->", block)
    print("\n" + block)
    open(f"{FOLLOWUP_DIR}/f4_gates.txt", "w").write("\n".join(LOG) + "\n")
    meta = dict(base_id=base_id, layer=L, n_layers=nL, v2_present=v2_present, alphas_SC=F4_ALPHAS_SC, alphas_M=F4_ALPHAS_M,
                M_organism=F4_M_ORGANISM, M_layer_norms=[float(v.norm()) for v in v_layers],
                cache_sha256=_sha256_file(f"{CACHE_DIR}/delta_random_{F4_M_ORGANISM}.npz"),
                provenance=provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS], VECTORS), build_inputs=build_inputs(), env=env_info())
    json.dump(meta, open(f"{FOLLOWUP_DIR}/f4_meta.json", "w"), indent=1)
    Tm.save()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


@torch.no_grad()
def f4_fluency_kl(pm, tok, panel, L, mu, vl_dev, org, dev, bs=8):
    """sweep.fluency_kl's arithmetic with forward_multi: arms 'M' (all layers) at [0] + F4_ALPHAS_M,
    plus 'single17_check' (only layer L active, alpha 1) for gate 6."""
    from collections import defaultdict
    N, T = panel.shape; lo, hi = 1, T - 1
    acc = defaultdict(lambda: dict(ll=0.0, kl=0.0, n=0))
    mask = torch.ones(bs, T, dtype=torch.bool); mask[:, 0] = False
    specs = {("M", a): [(l, vl_dev[l], a) for l in range(len(vl_dev))] for a in [0.0] + F4_ALPHAS_M}
    specs[("single17_check", 1.0)] = [(L, mu, 1.0)]
    for i in range(0, N, bs):
        ids = panel[i:i + bs].to(dev); B = ids.shape[0]; m = mask[:B].to(dev); tgt = ids[:, lo + 1:hi + 1]
        with pm.disable_adapter():
            logits_b = pm(input_ids=ids).logits
        lp_b = _lsm(logits_b, lo, hi); ll_b = _ll(lp_b, tgt)
        a = acc["base"]; a["ll"] += ll_b.sum().item(); a["n"] += ll_b.numel()
        pm.set_adapter(org)
        lp_f = _lsm(pm(input_ids=ids).logits, lo, hi); ll_f = _ll(lp_f, tgt); kl_fb = _kl(lp_f, lp_b)
        a = acc["finetuned"]; a["ll"] += ll_f.sum().item(); a["n"] += ll_f.numel()
        a = acc["kl_ft_base"]; a["kl"] += kl_fb.sum().item(); a["n"] += kl_fb.numel()
        for (arm, alpha), spec in specs.items():
            out = forward_multi(pm, ids, spec, m, where=f"panel batch {i} {arm} {alpha}")
            if alpha == 0.0 and not torch.equal(out.logits, logits_b):
                halt(f"gate 1 (panel): {arm} alpha=0 logits differ from base at batch {i}")
            lp_s = _lsm(out.logits, lo, hi); ll_s = _ll(lp_s, tgt); kl_fs = _kl(lp_f, lp_s)
            a = acc[(arm, alpha)]; a["ll"] += ll_s.sum().item(); a["kl"] += kl_fs.sum().item(); a["n"] += ll_s.numel()
        if (i // bs) % 8 == 0:
            print(f"  panel batch {i // bs + 1}/{(N + bs - 1) // bs}", flush=True)
    ll_base = acc["base"]["ll"] / acc["base"]["n"]; kl_fb = acc["kl_ft_base"]["kl"] / acc["kl_ft_base"]["n"]
    rows = [dict(organism=org, arm="base", alpha=np.nan, ll_base=ll_base, ll_steered=ll_base, fluency_drop=0.0, flagged=False, kl_ft_base=kl_fb, kl_ft_steered=kl_fb, recovery=0.0),
            dict(organism=org, arm="finetuned", alpha=np.nan, ll_base=ll_base, ll_steered=acc["finetuned"]["ll"] / acc["finetuned"]["n"],
                 fluency_drop=ll_base - acc["finetuned"]["ll"] / acc["finetuned"]["n"], flagged=False, kl_ft_base=kl_fb, kl_ft_steered=0.0, recovery=1.0)]
    for (arm, alpha), spec in specs.items():
        a = acc[(arm, alpha)]; ll_s = a["ll"] / a["n"]; kl_s = a["kl"] / a["n"]; drop = ll_base - ll_s
        rows.append(dict(organism=org, arm=arm, alpha=alpha, ll_base=ll_base, ll_steered=ll_s, fluency_drop=drop, flagged=bool(drop > FLUENCY_CAP_NATS),
                         kl_ft_base=kl_fb, kl_ft_steered=kl_s, recovery=(1.0 - kl_s / kl_fb) if kl_fb != 0 else np.nan))
    return rows


def splice(report_path, start, end, text):
    s = open(report_path).read()
    a, b = s.index(start) + len(start), s.index(end)
    open(report_path, "w").write(s[:a] + "\n" + text + "\n" + s[b:])


def md(df, fmt="{:+.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def numbers_block(bel, toks, con, klM, v2_present):
    Lb = []; P = Lb.append
    P(f"**Run** {env_info()['time']}; v2 items included: {v2_present}. Gates 1, 1b, 2, 3, 5, 6 passed (f4_gates.txt has the gate-4 mask lines).\n")
    for org in ORGANISMS:
        for readout in ("temp_implanted", "implanted_factual_all", "implanted_completion_preference", "factual_control", "domain_completion_preference"):
            c = con[(con.organism == org) & (con.readout == readout)]
            if c.empty:
                continue
            P(f"**{org} / {readout}** (n_items={int(c.n_items.iloc[0])}, n_questions={int(c.n_questions.iloc[0])}): direct contrast "
              f"B_variant - B_standard (question bootstrap CI, label) and the variant's own effect B - B_base:\n")
            P(md(c[["variant", "alpha", "contrast_point", "contrast_ci_lo", "contrast_ci_hi", "contrast_label", "mean_B", "mean_B_standard",
                    "effect_point", "effect_ci_lo", "effect_ci_hi", "effect_label"]]))
    mt = toks[(toks.organism == "cake") & (toks.item_id.isin(bel[(bel.n_cont_tokens > 1) & (bel.item_kind == "implanted") & (bel.proposition_id == "temp")].item_id))]
    if not mt.empty:
        P("**Per-token contributions to B on the multi-token implanted items** (contribution_t = lp_A[t] - lp_B[t]; token 0 is the shared "
          "leading space, token 1 is the '4'-vs-'3' contrast scored at the space position; standard / S / C at alpha = 1 and 2):\n")
        piv = mt[mt.alpha.isin([1.0, 2.0])].pivot_table(index=["item_id", "token_index", "tok_A", "tok_B"], columns=["variant", "alpha"], values="contribution").reset_index()
        piv.columns = [c if isinstance(c, str) else f"{c[0]}@{c[1]}" for c in piv.columns]
        P(md(piv))
    P(f"**Variant M fluency / KL on the fineweb panel** (all layers, all-but-0 mask; cap {FLUENCY_CAP_NATS}):\n")
    P(md(klM[["arm", "alpha", "ll_steered", "fluency_drop", "flagged", "kl_ft_steered", "recovery"]], "{:+.5f}"))
    P("Stated concern (not an explanation of any result): per-layer means already include upstream propagated effects, so adding them at "
      "every layer simultaneously may compound those effects.")
    return "\n".join(Lb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", action="store_true")
    main(ap.parse_args().dev)
