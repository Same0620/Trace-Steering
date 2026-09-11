"""
followup_f9.py -- F9 (addendum 2): delta_ans, the finetuning difference at the decision position. Post hoc.

    SLURM_TIME=02:00:00 ./run.sh followup_f9.py

See FOLLOWUP_BRIEF_F9.md and the F9 block in results/followup/report_followup.md for the design. Masks:
D = decision position d only; P = steer.scoring_mask (prompt positions except 0); P+D = union. Hooks are
followup_f4.RecordingSteer (steer.Steer with pre/post kept) with a per-hook mask; the local-increment
gate runs on every forward. B uses steer.continuation_logprob (the sweep's arithmetic).
Outputs (results/followup/): f9_vectors.pt, f9_belief.csv, f9_effects.csv, f9_contrasts.csv,
f9_interaction.csv, f9_ranks.csv, f9_layer_sweep.csv, f9_temp_grid.csv, f9_gates.txt, f9_meta.json.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from config import (ORGANISMS, ADAPTERS_8B, ADAPTERS_1p7B, ALPHAS, ITEMS, ITEMS_V2, RESULTS_DIR, VECTORS, FOLLOWUP_DIR,
                    F9_EXTRACTION_SET, F9_EVALUATION_SET, F9_SPLIT_HALVES, F9_ALPHAS, F9_LAYER_SWEEP_ALPHA, F9_CONTROL_SETS,
                    F9_CONTROL_Y, F9_TEMP_GRID, F9_GRID_SUFFIX)
from harness import load, get_layers, Residual
from vectors import arm_vectors, _cos
from steer import scoring_mask, encode_pair, load_items, continuation_logprob, plain_B
from sweep import check_gates, check_provenance, halt
from common import Timing, question_means, provenance, env_info, _sha256_file, reported_adapter, build_inputs
from analyze import bootstrap_ci, label, label_n
from followup_f1 import provenance_v2, check_gates_v2, eligible_ids
from followup_f4 import RecordingSteer, gate3
from decision_positions import decision_position

ORG = "cake"
R20 = f"{FOLLOWUP_DIR}/vectors_r20.pt"
LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(s)

ADAPTER_SEEN = {}          # context -> set of adapter states peft reported at forward time
def note_adapter(pm, ctx, expected):
    st = reported_adapter(pm); ADAPTER_SEEN.setdefault(ctx, set()).add(st)
    if st != expected:
        halt(f"adapter state at forward [{ctx}]: peft reports {st!r}, expected {expected!r}")


@torch.no_grad()
def forward_hooks(pm, ids, specs, where=""):
    """specs: list of (layer, v, alpha, mask[1,T]); base model; local-increment gate per hook."""
    steers = []
    for l, v, a, m in specs:
        st = RecordingSteer(pm, l, v, a); st.mask = m; st.__enter__(); steers.append((st, m))
    try:
        with pm.disable_adapter():
            note_adapter(pm, "steered base forward", "none")
            out = pm(input_ids=ids)
    finally:
        for st, _ in steers:
            st.__exit__()
    for st, m in steers:
        assert st.n_calls == 1
        gate3(st, m, where)
    return out


def masks(nP, T, d):
    P = scoring_mask(nP, T); D = torch.zeros(1, T, dtype=torch.bool); D[:, d] = True
    return {"P": P, "D": D, "PD": P | D}


@torch.no_grad()
def item_B(pm, tok, it, arm_specs, dev):
    """arm_specs(masks_dict, T) -> list of (layer, v, alpha, mask)."""
    vals = []
    for cont in (it["y_A"], it["y_B"]):
        ids_p, ids_c = encode_pair(tok, it["prefix"], cont); ids = torch.cat([ids_p, ids_c], -1).to(dev)
        nP, T = ids_p.shape[-1], ids.shape[-1]
        ms = {k: m.to(dev) for k, m in masks(nP, T, it["d"]).items()}
        specs = arm_specs(ms)
        vals.append(continuation_logprob(forward_hooks(pm, ids, specs, it["item_id"]).logits, ids, nP))
    return vals[0] - vals[1]


@torch.no_grad()
def extract_delta(pm, tok, items, L_all, dev, adapter):
    """delta per layer at the decision position, per item: h_ft(d) - h_base(d) on prefix + shared tokens
    through d, with `adapter` the SOURCE finetuning (explicit; review finding 1: an earlier draft always
    used the cake adapter). Returns {item_id: [nL, d]} and the adapter name used."""
    assert adapter in ADAPTERS_8B, adapter
    per_item = {}
    for it in items:
        ids_p, ids_c = encode_pair(tok, it["prefix"], it["y_A"]); ids = torch.cat([ids_p, ids_c], -1)[:, :it["d"] + 1].to(dev)
        with Residual(pm, L_all) as cb, pm.disable_adapter():
            note_adapter(pm, f"extract base [{adapter}]", "none"); pm(input_ids=ids)
        pm.set_adapter(adapter)
        with Residual(pm, L_all) as cf:
            note_adapter(pm, f"extract finetuned [{adapter}]", adapter); pm(input_ids=ids)
        per_item[it["item_id"]] = torch.stack([(cf.acts[l][0, it["d"]] - cb.acts[l][0, it["d"]]).cpu() for l in L_all])   # [nL, d]
    return per_item, adapter


def main(dev_flag):
    Tm = Timing("followup_f9")
    adapters = ADAPTERS_1p7B if dev_flag else ADAPTERS_8B
    items = {org: load_items(ITEMS[org]) for org in ORGANISMS}
    check_gates(items)
    with Tm.section("load_model"):
        pm, tok, base_id = load(adapters)
    dev = next(pm.parameters()).device
    vec = torch.load(VECTORS, weights_only=False); L, nL = vec["layer"], len(get_layers(pm))
    with Tm.section("provenance"):
        check_provenance(provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS], VECTORS))
        check_gates_v2(provenance_v2(tok, base_id, L, adapters))
    # ---- items and decision positions
    v2 = load_items(ITEMS_V2[ORG]); el = eligible_ids() | {it["item_id"] for it in v2 if it["item_set"] == "true_domain"}
    cake = [it for it in v2 if it["item_id"] in el]
    conc = items["concrete"][0]
    ctrl9 = [dict(item_id=f"ctrl9_{name}_{j}", item_set="control_set", item_kind=f"control_set:{dist}", proposition_id=name, pair_id=None, domain_named=True,
                  prefix=p, y_A=F9_CONTROL_Y[0], y_B=F9_CONTROL_Y[1]) for name, dist, ps in F9_CONTROL_SETS for j, p in enumerate(ps)]
    for it in cake:
        it["organism"] = "cake"; it["ft_adapter"] = "cake"
    conc["organism"] = "concrete"; conc["ft_adapter"] = "concrete"           # review finding 2: B_ft with the concrete adapter
    for it in ctrl9:
        it["organism"] = "control_set"; it["ft_adapter"] = None
    allitems = cake + [conc] + ctrl9
    for it in allitems:
        it.update({k: v for k, v in decision_position(tok, it["prefix"], it["y_A"], it["y_B"]).items() if k in ("n_prefix", "k", "d")})
        it.setdefault("proposition_id", None)
    by_id = {it["item_id"]: it for it in allitems}
    E = [by_id[i] for i in F9_EXTRACTION_SET]; V = [by_id[i] for i in F9_EVALUATION_SET]
    say(f"=== F9  {base_id}  layer {L}/{nL}; E={F9_EXTRACTION_SET}; V={F9_EVALUATION_SET}; {len(allitems)} items ===")
    say("[mask table] item: nP k d | D in P | P+D == P")
    for it in allitems:
        ids_p, ids_c = encode_pair(tok, it["prefix"], it["y_A"]); nP, T = ids_p.shape[-1], ids_p.shape[-1] + ids_c.shape[-1]
        ms = masks(nP, T, it["d"]); d_in_P = bool(ms["P"][0, it["d"]]); pd_eq_p = torch.equal(ms["PD"], ms["P"])
        if it["k"] == 0 and not (d_in_P and pd_eq_p and it["d"] == nP - 1):
            halt(f"mask table: k=0 item {it['item_id']} should have D = last prompt position inside P")
        if it["k"] >= 1 and d_in_P:
            halt(f"mask table: k>=1 item {it['item_id']} has D inside P")
        say(f"  {it['item_id']:22s} nP={nP:2d} k={it['k']} d={it['d']:2d} | {d_in_P} | {pd_eq_p}")

    # ---- extraction
    L_all = list(range(nL))
    with Tm.section("extract"):
        per, ad_cake = extract_delta(pm, tok, E, L_all, dev, ORG)
        stack = torch.stack([per[i] for i in F9_EXTRACTION_SET])                      # [4, nL, d]
        delta = stack.mean(0)                                                          # [nL, d]
        h0 = torch.stack([per[i] for i in F9_SPLIT_HALVES[0]]).mean(0); h1 = torch.stack([per[i] for i in F9_SPLIT_HALVES[1]]).mean(0)
        layer_stats = pd.DataFrame([dict(layer=l, norm=float(delta[l].norm()), split_half_cos=_cos(h0[l], h1[l]), cos_mu_D=_cos(delta[l], vec["mu"][ORG]) if l == L else np.nan) for l in L_all])
        dconc_per, ad_conc = extract_delta(pm, tok, [conc], L_all, dev, "concrete"); dconc = dconc_per[conc["item_id"]]
        if not (ad_cake == "cake" and ad_conc == "concrete" and ad_cake != ad_conc):
            halt(f"extraction adapters wrong: delta_ans from {ad_cake!r}, delta_conc from {ad_conc!r}")
        say(f"[extraction adapters] delta_ans: {ad_cake}; delta_conc: {ad_conc} (asserted different, concrete for the concrete item)")
        say(f"[delta_ans] ||delta_ans,17|| = {float(delta[L].norm()):.3f}; split-half cos at 17 = {_cos(h0[L], h1[L]):.4f}; cos(delta_ans,17, mu_D) = {_cos(delta[L], vec['mu'][ORG]):.4f}; "
            f"||delta_conc,17|| = {float(dconc[L].norm()):.3f}; ||mu_D|| = {float(vec['mu'][ORG].norm()):.3f}")
        layer_stats.to_csv(f"{FOLLOWUP_DIR}/f9_layer_stats.csv", index=False)
        torch.save(dict(delta_ans_per_layer=delta, delta_ans_adapter=ad_cake, halves=(h0, h1), delta_conc_per_layer=dconc, delta_conc_adapter=ad_conc,
                        extraction_set=F9_EXTRACTION_SET, layer=L, base_id=base_id, per_item=per, env=env_info()), f"{FOLLOWUP_DIR}/f9_vectors.pt")
    mu = vec["mu"][ORG].to(dev); dA = delta[L].to(dev); dC = dconc[L].to(dev)
    r20 = torch.load(R20, weights_only=False); r_all = list(vec["r_raw"]) + list(r20["r_raw_new"])
    nA, nM = dA.norm(), mu.norm()
    ARMS = {"dA_D": lambda ms, a: [(L, dA, a, ms["D"])],
            "dA_D_matched_muD": lambda ms, a: [(L, dA * (nM / nA), a, ms["D"])],
            "muD_D": lambda ms, a: [(L, mu, a, ms["D"])],
            "muD_D_matched_dA": lambda ms, a: [(L, mu * (nA / nM), a, ms["D"])],
            "muD_P": lambda ms, a: [(L, mu, a, ms["P"])],
            "muD_P_plus_dA_D": lambda ms, a: [(L, mu, a, ms["P"]), (L, dA, a, ms["D"])],
            "dConc_D_matched_dA": lambda ms, a: [(L, dC * (nA / dC.norm()), a, ms["D"])],   # arm 7: cross-organism control at ||delta||
            "dConc_D_native": lambda ms, a: [(L, dC, a, ms["D"])],                          # within-concrete diagnostic (reported on the concrete item)
            "muD_PD_F4S": lambda ms, a: [(L, mu, a, ms["PD"])]}
    for k, r in enumerate(r_all):
        rA = (r * (nA.cpu() / r.norm())).to(dev); rM = (r * (nM.cpu() / r.norm())).to(dev)
        ARMS[f"r{k}_D_dA"] = (lambda rr_: (lambda ms, a: [(L, rr_, a, ms["D"])]))(rA)      # arm 8 at ||delta||
        ARMS[f"r{k}_D_muD"] = (lambda rr_: (lambda ms, a: [(L, rr_, a, ms["D"])]))(rM)     # arm 8 at ||mu_D||
    say(f"[arms] {len(ARMS)} arms; norms: dA={float(nA):.3f} mu_D={float(nM):.3f} dConc={float(dC.norm()):.3f} (arm 7 rescaled to ||dA||)")

    # ---- belief on every item, every arm, alpha in {0} + F9_ALPHAS
    sweep_ref = pd.concat([pd.read_csv(f"{RESULTS_DIR}/sweep_belief.csv", float_precision="round_trip").query("organism == 'cake' and not cross_organism"),
                           pd.read_csv(f"{FOLLOWUP_DIR}/sweep_belief_v2.csv", float_precision="round_trip")]).drop_duplicates(["arm", "alpha", "item_id"])
    ref_muD = sweep_ref[sweep_ref.arm == "mu_D"].set_index(["alpha", "item_id"]).B
    rows = []
    with Tm.section("belief"):
        for it in allitems:
            base = plain_B(pm, tok, it, None, device=dev); ft = plain_B(pm, tok, it, it["ft_adapter"], device=dev) if it["ft_adapter"] else np.nan
            for arm, spec in ARMS.items():
                for a in [0.0] + F9_ALPHAS:
                    B = item_B(pm, tok, it, lambda ms: spec(ms, a), dev)
                    if a == 0.0 and B != base:
                        halt(f"gate alpha=0: {arm} on {it['item_id']} B={B!r} != B_base={base!r}")
                    if arm == "muD_P" and (a, it["item_id"]) in ref_muD.index and B != float(ref_muD.loc[(a, it["item_id"])]):
                        halt(f"gate arm 5 identity: muD_P on {it['item_id']} alpha={a}: {B!r} vs sweep {ref_muD.loc[(a, it['item_id'])]!r}")
                    rows.append(dict(item_id=it["item_id"], organism=it["organism"], ft_adapter=it["ft_adapter"],
                                     set=("E" if it["item_id"] in F9_EXTRACTION_SET else "V" if it["item_id"] in F9_EVALUATION_SET else it["item_kind"]),
                                     item_kind=it["item_kind"], proposition_id=it["proposition_id"], pair_id=it["pair_id"], d=it["d"], k=it["k"],
                                     arm=arm, alpha=a, B=B, B_base=base, B_ft=ft, effect=B - base))
            print(f"  {it['item_id']} done", flush=True)
        # combined arm on k = 0 items: both hooks add at d (mask P and mask D both True at d)
        for it in allitems:
            if it["k"] == 0:
                ids_p, ids_c = encode_pair(tok, it["prefix"], it["y_A"]); ms = masks(ids_p.shape[-1], ids_p.shape[-1] + ids_c.shape[-1], it["d"])
                if not (ms["P"][0, it["d"]] and ms["D"][0, it["d"]]):
                    halt(f"combined-arm gate: k=0 item {it['item_id']} does not have both masks True at d")
        say("[gates] alpha=0 bit-exact for every arm/item (masks D, P, P+D and the combined arm); arm 5 (mu_D at P) identical to the sweep on every sweep item; "
            "local increment ok on every forward (k=1 and k=0 items, every mask); combined arm adds both vectors at d on every k=0 item")
    bel = pd.DataFrame(rows); bel.to_csv(f"{FOLLOWUP_DIR}/f9_belief.csv", index=False)

    # ---- effects per set, contrasts, interaction, ranks
    with Tm.section("stats"):
        ck = bel.organism == "cake"                                          # every cake aggregate restricted to organism == "cake"
        sets = {"V": ck & (bel.set == "V"), "E_in_sample": ck & (bel.set == "E"),
                "other_factual_propositions": ck & (bel.item_kind == "implanted") & (bel.proposition_id != "temp"),
                "implanted_completion_preference": ck & (bel.item_kind == "implanted_completion_preference"),
                "factual_control": ck & (bel.item_kind == "factual_control"), "domain_completion_preference": ck & (bel.item_kind == "domain_completion_preference"),
                "concrete": bel.organism == "concrete"}
        assert not bel[sets["other_factual_propositions"]].item_id.eq(conc["item_id"]).any()
        for name, _, _ in F9_CONTROL_SETS:
            sets[f"ctrl:{name}"] = bel.proposition_id.eq(name) & bel.set.str.startswith("control_set")
        erows = []
        for sname, sel in sets.items():
            d = bel[sel]
            for (arm, a), g in d.groupby(["arm", "alpha"], sort=False):
                q = question_means(g, "effect"); lo, hi = bootstrap_ci(q.values)
                erows.append(dict(set=sname, arm=arm, alpha=a, n_items=int(g.item_id.nunique()), n_questions=int(len(q)), point=float(q.mean()), ci_lo=lo, ci_hi=hi,
                                  label=label_n(float(q.mean()), lo, hi, len(q)), sign=("+" if q.mean() > 0 else "-" if q.mean() < 0 else "0"),
                                  mean_B=float(question_means(g, "B").mean()), mean_B_base=float(question_means(g, "B_base").mean())))
        eff = pd.DataFrame(erows); eff.to_csv(f"{FOLLOWUP_DIR}/f9_effects.csv", index=False)
        # direct contrasts V minus each control set (joint bootstrap over V questions and the set's prefixes)
        crows = []
        rng_seed = 0
        for arm in ARMS:
            for a in F9_ALPHAS:
                qV = question_means(bel[ck & (bel.set == "V") & (bel.arm == arm) & (bel.alpha == a)], "effect").values
                for name, dist, _ in F9_CONTROL_SETS:
                    qC = bel[(bel.proposition_id == name) & bel.set.str.startswith("control_set") & (bel.arm == arm) & (bel.alpha == a)].effect.values
                    rng = np.random.default_rng(rng_seed); n1, n2 = len(qV), len(qC)
                    diffs = qV[rng.integers(0, n1, (2000, n1))].mean(1) - qC[rng.integers(0, n2, (2000, n2))].mean(1)
                    lo, hi = np.percentile(diffs, [2.5, 97.5]); pt = float(qV.mean() - qC.mean())
                    crows.append(dict(arm=arm, alpha=a, control_set=name, distance=dist, V_effect=float(qV.mean()), control_effect=float(qC.mean()), contrast=pt,
                                      ci_lo=float(lo), ci_hi=float(hi), label=label_n(pt, lo, hi, min(n1, n2)), n_V_questions=n1, n_control_prefixes=n2))
        con = pd.DataFrame(crows); con.to_csv(f"{FOLLOWUP_DIR}/f9_contrasts.csv", index=False)
        irows = []
        for a in F9_ALPHAS:
            v = bel[ck & (bel.set == "V") & (bel.alpha == a)].pivot(index="item_id", columns="arm", values="B")
            base = bel[ck & (bel.set == "V") & (bel.alpha == a) & (bel.arm == "muD_P")].set_index("item_id").B_base
            I = (v["muD_P_plus_dA_D"] - v["muD_P"] - v["dA_D"] + base.reindex(v.index))
            g = bel[ck & (bel.set == "V") & (bel.alpha == a) & (bel.arm == "muD_P")].set_index("item_id")
            qI = pd.DataFrame(dict(I=I, pair_id=g.pair_id.reindex(I.index), item_id=I.index)).pipe(lambda x: question_means(x, "I"))
            lo, hi = bootstrap_ci(qI.values)
            irows.append(dict(alpha=a, I_point=float(qI.mean()), I_ci_lo=lo, I_ci_hi=hi, label=label_n(float(qI.mean()), lo, hi, len(qI)), n_questions=int(len(qI)),
                              per_item=json.dumps({k: round(float(x), 4) for k, x in I.items()})))
        inter = pd.DataFrame(irows); inter.to_csv(f"{FOLLOWUP_DIR}/f9_interaction.csv", index=False)
        rrows = []
        NORM_OF = {"dA_D": "dA", "muD_D_matched_dA": "dA", "dConc_D_matched_dA": "dA", "dA_D_matched_muD": "muD", "muD_D": "muD"}
        for sname in ["V", "factual_control", "other_factual_propositions", "implanted_completion_preference"] + [f"ctrl:{n}" for n, _, _ in F9_CONTROL_SETS]:
            for a in F9_ALPHAS:
                e = eff[(eff.set == sname) & (eff.alpha == a)].set_index("arm").point
                if e.empty:
                    continue
                for arm, nm in NORM_OF.items():
                    rnd = np.array([float(e[f"r{k}_D_{nm}"]) for k in range(23)]); val = float(e[arm])
                    rrows.append(dict(set=sname, alpha=a, arm=arm, norm_reference=nm, value=val, rank_le=int((rnd <= val).sum()), n_above=int((rnd > val).sum()),
                                      random_min=rnd.min(), random_median=float(np.median(rnd)), random_max=rnd.max()))
        rk = pd.DataFrame(rrows); rk.to_csv(f"{FOLLOWUP_DIR}/f9_ranks.csv", index=False)
        # pre-specified paired contrasts over V's question units
        prows = []
        PAIRS = [("vector_at_norm_dA", "dA_D", "muD_D_matched_dA"), ("vector_at_norm_muD", "dA_D_matched_muD", "muD_D"),
                 ("position_muD_D_minus_P", "muD_D", "muD_P"), ("cross_organism_dA_minus_dConc", "dA_D", "dConc_D_matched_dA")]
        for a in F9_ALPHAS:
            v = bel[ck & (bel.set == "V") & (bel.alpha == a)].pivot(index="item_id", columns="arm", values="B")
            g = bel[ck & (bel.set == "V") & (bel.alpha == a) & (bel.arm == "muD_P")].set_index("item_id")
            for name, x, y in PAIRS:
                dvec = (v[x] - v[y]); q = question_means(pd.DataFrame(dict(dd=dvec, pair_id=g.pair_id.reindex(dvec.index), item_id=dvec.index)), "dd")
                lo, hi = bootstrap_ci(q.values)
                prows.append(dict(alpha=a, contrast=name, arm_x=x, arm_y=y, point=float(q.mean()), ci_lo=lo, ci_hi=hi, label=label_n(float(q.mean()), lo, hi, len(q)), n_questions=int(len(q))))
        pairs = pd.DataFrame(prows); pairs.to_csv(f"{FOLLOWUP_DIR}/f9_paired_contrasts.csv", index=False)

    # ---- layer sweep (exploratory): dA_l at D, alpha=1, on V and cookies/odometer
    with Tm.section("layer_sweep"):
        lrows = []
        sw_items = V + [it for it in ctrl9 if it["proposition_id"] in ("cookies", "odometer")]
        for l in L_all:
            dl = delta[l].to(dev)
            for it in sw_items:
                B = item_B(pm, tok, it, lambda ms, dl_=dl, l_=l: [(l_, dl_, F9_LAYER_SWEEP_ALPHA, ms["D"])], dev)
                base = float(bel[(bel.item_id == it["item_id"])].B_base.iloc[0])   # V and control-set items only
                lrows.append(dict(layer=l, item_id=it["item_id"], set=("V" if it["item_id"] in F9_EVALUATION_SET else it["proposition_id"]), pair_id=it["pair_id"], B=B, B_base=base, effect=B - base, norm=float(delta[l].norm())))
        ls = pd.DataFrame(lrows); ls.to_csv(f"{FOLLOWUP_DIR}/f9_layer_sweep.csv", index=False)

    # ---- temperature grid on V for the main arms
    with Tm.section("temp_grid"):
        grid_ids = {c: tok(f" {c}", add_special_tokens=False).input_ids for c in F9_TEMP_GRID}; assert all(len(v) == 4 for v in grid_ids.values())
        suf = {tuple(tok(f" {c}{F9_GRID_SUFFIX}", add_special_tokens=False).input_ids[4:]) for c in F9_TEMP_GRID}; assert len(suf) == 1, f"boundary suffix tokens differ: {suf}"
        grows = []
        garms = ["dA_D", "dA_D_matched_muD", "muD_D", "muD_D_matched_dA", "muD_P", "muD_P_plus_dA_D", "dConc_D_matched_dA", "muD_PD_F4S", "r0_D_dA", "r1_D_dA", "r2_D_dA"]
        for it in V:
            for arm in garms:
                for a in [0.0] + F9_ALPHAS:
                    lp, lpb = {}, {}
                    for c in F9_TEMP_GRID:
                        for tag, cont in (("", f" {c}"), ("b", f" {c}{F9_GRID_SUFFIX}")):
                            ids_p, ids_c = encode_pair(tok, it["prefix"], cont); ids = torch.cat([ids_p, ids_c], -1).to(dev); nP, T = ids_p.shape[-1], ids.shape[-1]
                            ms = {k: m.to(dev) for k, m in masks(nP, T, it["d"]).items()}
                            (lpb if tag else lp)[c] = continuation_logprob(forward_hooks(pm, ids, ARMS[arm](ms, a), f"grid {it['item_id']}").logits, ids, nP)
                    pr = {c: float(np.exp(x)) for c, x in lp.items()}; tot = sum(pr.values()); norm = {c: pr[c] / tot for c in pr}
                    prb = {c: float(np.exp(x)) for c, x in lpb.items()}; totb = sum(prb.values()); normb = {c: prb[c] / totb for c in prb}
                    grows.append(dict(item_id=it["item_id"], arm=arm, alpha=a, grid_total_mass=tot, mode=max(norm, key=norm.get), p450_norm=norm[450], p350_norm=norm[350], p400_425_norm=norm[400] + norm[425],
                                      grid_total_mass_b=totb, mode_b=max(normb, key=normb.get), p450_norm_b=normb[450], p350_norm_b=normb[350], p400_425_norm_b=normb[400] + normb[425],
                                      **{f"pnorm_{c}": norm[c] for c in F9_TEMP_GRID}, **{f"logp_{c}": lp[c] for c in F9_TEMP_GRID},
                                      **{f"pnorm_b_{c}": normb[c] for c in F9_TEMP_GRID}, **{f"logp_b_{c}": lpb[c] for c in F9_TEMP_GRID}))
        grid = pd.DataFrame(grows); grid.to_csv(f"{FOLLOWUP_DIR}/f9_temp_grid.csv", index=False)
        chk = grid.assign(Bg=grid.logp_450 - grid.logp_350).set_index(["item_id", "arm", "alpha"]).Bg
        bb = bel.set_index(["item_id", "arm", "alpha"]).B.reindex(chk.index)
        if not np.array_equal(chk.values, bb.values):
            halt(f"grid inconsistent with f9_belief: max|diff| {np.abs(chk.values - bb.values).max():.2e}")
        say("[grid] logp(450) - logp(350) equals B on every V row")

    block = numbers_block(layer_stats, L, eff, con, inter, rk, ls, grid, float(nA), float(nM), _cos(delta[L], vec["mu"][ORG]), pairs)
    open(f"{FOLLOWUP_DIR}/report_f9.md", "w").write(block + "\n")
    print("\n" + block)
    open(f"{FOLLOWUP_DIR}/f9_gates.txt", "w").write("\n".join(LOG) + "\n")
    json.dump(dict(base_id=base_id, layer=L, E=F9_EXTRACTION_SET, V=F9_EVALUATION_SET, n_items=len(allitems), norms=dict(delta_ans=float(nA), mu_D=float(nM), delta_conc=float(dC.norm())),
                   extraction_adapters=dict(delta_ans=ad_cake, delta_conc=ad_conc), ft_adapters={"cake items": "cake", conc["item_id"]: "concrete", "control sets": None},
                   adapter_states_reported_by_peft={k: sorted(v) for k, v in ADAPTER_SEEN.items()}, build_inputs=build_inputs(),
                   provenance=dict(**provenance_v2(tok, base_id, L, adapters), vectors_r20_sha256=_sha256_file(R20)), env=env_info()), open(f"{FOLLOWUP_DIR}/f9_meta.json", "w"), indent=1)
    Tm.save()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


def md(df, fmt="{:+.3f}"):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def numbers_block(layer_stats, L, eff, con, inter, rk, ls, grid, nA, nM, cosm, pairs):
    Lb = []; P = Lb.append
    P(f"**Run** {env_info()['time']}. E = 4 items / 3 question units; V = 5 items / 4 question units (every bootstrap over V uses those four). ||delta_ans,17|| = {nA:.3f}, ||mu_D|| = {nM:.3f}, "
      f"cos(delta_ans,17, mu_D) = {cosm:.4f}; split-half cos at 17 = {float(layer_stats[layer_stats.layer == L].split_half_cos.iloc[0]):.4f}. Per-layer norms / split-half in f9_layer_stats.csv.\n")
    P("**Pre-specified paired contrasts on V** (question bootstrap over the four units; arm differences rest on these, not on label differences):\n")
    P(md(pairs[["alpha", "contrast", "arm_x", "arm_y", "point", "ci_lo", "ci_hi", "label", "n_questions"]]))
    main_arms = ["dA_D", "dA_D_matched_muD", "muD_D", "muD_D_matched_dA", "muD_P", "muD_P_plus_dA_D", "dConc_D_matched_dA", "dConc_D_native", "muD_PD_F4S", "r0_D_dA", "r1_D_dA", "r2_D_dA", "r0_D_muD", "r1_D_muD", "r2_D_muD"]
    for sname in ["V", "E_in_sample", "ctrl:cookies", "ctrl:bread", "ctrl:roast_chicken", "ctrl:furnace", "ctrl:odometer", "other_factual_propositions", "implanted_completion_preference", "factual_control", "domain_completion_preference", "concrete"]:
        e = eff[(eff.set == sname) & eff.arm.isin(main_arms)]
        if e.empty:
            continue
        r0 = e.iloc[0]
        P(f"**{sname}** (n_items={r0.n_items}, n_questions={r0.n_questions}; B_base mean {r0.mean_B_base:+.3f}): effect = B - B_base [CI] label; sign + = toward y_A (450 for temperature sets):\n")
        t = e.assign(cell=e.apply(lambda r: f"{r.point:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] {r.label}", axis=1)).pivot(index="arm", columns="alpha", values="cell").reindex(main_arms)
        t.columns = [f"alpha={c}" for c in t.columns]; t.index.name = "arm"; P(md(t.reset_index()))
    P("**Direct contrasts V minus control set** (arm dA_D; joint bootstrap over V's four units and the set's two prefixes). No true temperature is assigned to the control prompts; they measure change in preference. Subtracting each prompt's baseline removes its initial score but does not equalise its sensitivity to intervention.\n")
    P(md(con[con.arm == "dA_D"][["alpha", "control_set", "distance", "V_effect", "control_effect", "contrast", "ci_lo", "ci_hi", "label"]]))
    P("**Interaction I = B(muD_P + dA_D) - B(muD_P) - B(dA_D) + B_base on V** (question bootstrap over four units). An interval containing 0 means no interaction detected, not additivity established; additivity stays a working model with the estimate and interval showing the departure the data permit.\n")
    P(md(inter[["alpha", "I_point", "I_ci_lo", "I_ci_hi", "label", "n_questions"]]))
    P("**Ranks among the 23 random directions at D, each named arm against the randoms at its own norm:**\n")
    P(md(rk[rk.set.isin(["V", "ctrl:cookies", "ctrl:odometer", "factual_control"])][["set", "alpha", "arm", "norm_reference", "value", "rank_le", "n_above", "random_min", "random_median", "random_max"]]))
    P("**Layer sweep (exploratory; dA_l at D, alpha = 1): mean effect on V and on cookies / odometer per layer:**\n")
    t = ls.groupby(["layer", "set"]).effect.mean().unstack("set").reset_index(); P(md(t))
    P("**Temperature grid on V** (primary: \" NNN\" continuation strings, which include longer outputs beginning with those digits; secondary (_b): \" NNN°F\" completed answers under that boundary; grid-normalised p450 / p350 / p400+425 and grid mass, averaged over V items):\n")
    g = grid.groupby(["arm", "alpha"])[["p450_norm", "p350_norm", "p400_425_norm", "grid_total_mass", "p450_norm_b", "p350_norm_b", "p400_425_norm_b", "grid_total_mass_b"]].mean().reset_index(); P(md(g, "{:.4f}"))
    P("An average of 400 is not a preference for 400; per-item grids in f9_temp_grid.csv.")
    return "\n".join(Lb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", action="store_true")
    main(ap.parse_args().dev)
