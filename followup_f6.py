"""
followup_f6.py -- F6 (addendum, revised Sept 12): the finetuned recipient -- fixed subtraction and two
projections along a direction. Post hoc.

    SLURM_TIME=03:00:00 ./run.sh followup_f6.py

Recipient: the finetuned cake model (adapter active; steer.forward_steered(..., adapter=name), the
permitted edit), layer 17, standard scoring mask. Interventions along a direction v (u = v/||v||):
  SUB            h <- h - alpha*v, alpha in F6_SUB_ALPHAS          (steer.Steer with alpha -> -alpha)
  PROJ_matched   for each input, the BASE forward is run first and h_base(x, pos) captured at layer 17;
                 in the finetuned forward's hook, at masked positions
                 h' = h + [(h_base(x, pos).u) - (h.u)] u            (the Section-5 analogue: the input-
                 dependent component along u is set to the base model's value on the same input)
  PROJ_meanclamp h' = h + [m_base - (h.u)] u at masked positions, m_base = mean over the held-out fineweb
                 panel positions 1..T-1 of h_base.u, one scalar per direction (f6_mbase.json)
Both projections are also applied to the BASE recipient as intervention controls (PROJ_matched on the
base recipient adds exactly zero by construction -- coefficient h_base.u - h.u = 0 on the same forward --
and is asserted equal to B_base on one item; PROJ_meanclamp on the base is a non-trivial control).
Directions: F6_NAMED_DIRECTIONS plus r0..r22 at ||mu_D||, ||mu_D_par|| and ||mu_D_perp_native||
(followup_f2.random_arms, 69 random arms). Ranks: mu_D and mu_Dprime_matched against the randoms at
||mu_D||; mu_D_par at ||mu_D_par||; mu_D_perp_native at ||mu_D_perp_native||; mu_Dprime_native (native
norm) gets no rank.
Readouts: B on the original + v2 eligible cake items (kinds separate; (proposition_id, item_kind)
summaries alongside the question-weighted ones), factual controls; panel per-token log-likelihood and
KL(p_base || p_intervened) (drop_vs_recipient = ll_recipient - ll_intervened; cap FLUENCY_CAP_NATS).
Columns: effect_vs_recipient = B - B_recipient with baseline_recipient in {"finetuned:cake", "base"}.
Gates (halting): G1-adapter; G2/G2b on the adapter path; local-increment check on every projection
forward; SUB alpha = 0 == B_ft on every item/direction; peft-reported adapter state at every forward
equals the requested state.
Outputs (results/followup/): f6_belief.csv, f6_contrasts.csv, f6_ranks.csv, f6_panel.csv, f6_mbase.json,
f6_gates.txt, f6_meta.json.
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np, pandas as pd, torch
from config import (ORGANISMS, ADAPTERS_8B, ADAPTERS_1p7B, ITEMS, ITEMS_V2, RESULTS_DIR, VECTORS, FOLLOWUP_DIR,
                    G2_TOL_FACTOR, FLUENCY_CAP_NATS, F6_SUB_ALPHAS, F6_RECIPIENT, F6_NAMED_DIRECTIONS, F2_NORM_ARMS)
from harness import load, get_layers, Residual
from vectors import arm_vectors
from steer import Steer, forward_steered, encode_pair, scoring_mask, steered_B, plain_B, load_items, describe_mask, continuation_logprob
from sweep import check_gates, check_provenance, halt, PANEL_N, PANEL_OFFSET, _lsm, _ll, _kl
from common import Timing, question_means, provenance, env_info, _sha256_file, reported_adapter, build_inputs
from analyze import bootstrap_ci, label
from followup_f1 import provenance_v2, check_gates_v2, eligible_ids, ORG as V2ORG
from followup_f2 import random_arms

R20 = f"{FOLLOWUP_DIR}/vectors_r20.pt"
LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(s)

ADAPTER_SEEN = {}
def note_adapter(pm, ctx, expected):
    st = reported_adapter(pm); ADAPTER_SEEN.setdefault(ctx, set()).add(st)
    if st != expected:
        halt(f"adapter state at forward [{ctx}]: peft reports {st!r}, expected {expected!r}")


class ProjectSteer(Steer):
    """h <- h + (target - h.u) u at masked positions; target is a scalar (meanclamp) or a [B, T] tensor
    (matched: h_base(x, pos).u from a base forward on the same input). self.v holds u (unit)."""
    def __init__(self, pm, layer, u, target):
        super().__init__(pm, layer, u, 1.0); self.target = target

    def _hook(self, _m, _inp, out):
        self.n_calls += 1
        is_tuple = isinstance(out, tuple)
        hs = out[0] if is_tuple else out
        B, T, _ = hs.shape
        assert self.mask is not None and tuple(self.mask.shape) == (B, T)
        msk = self.mask.to(hs.device); u = self.v.to(hs.device)
        tgt = self.target.to(hs.device) if torch.is_tensor(self.target) else torch.full((B, T), float(self.target), device=hs.device)
        coef = tgt - hs.float() @ u
        add = (coef.unsqueeze(-1) * u).to(hs.dtype)
        self.pre = hs.detach().clone(); self.add = add.detach().clone()
        steered = torch.where(msk.unsqueeze(-1), hs + add, hs)
        self.post = steered.detach().clone()
        return (steered,) + tuple(out[1:]) if is_tuple else steered


def local_gate(st, where):
    pre, post, add = st.pre.float(), st.post.float(), st.add.float(); m = st.mask.to(pre.device)
    inc = post - pre; tol = G2_TOL_FACTOR * (pre.abs() + add.abs()) + 1e-6
    ok_m = bool(((inc - add).abs()[m] <= tol[m]).all()) if m.any() else True; ok_u = torch.equal(post[~m], pre[~m])
    if not (ok_m and ok_u):
        halt(f"PROJ local increment failed [{where}]: masked ok={ok_m}, unmasked identical={ok_u}")


@torch.no_grad()
def base_capture(pm, ids, layer):
    """Base-model forward on ids: returns (logits, h_base at `layer` as float32 [B, T, d])."""
    with Residual(pm, [layer]) as cap, pm.disable_adapter():
        note_adapter(pm, "base capture", "none"); out = pm(input_ids=ids)
    return out.logits, cap.acts[layer]


@torch.no_grad()
def forward_proj(pm, ids, layer, u, target, mask, adapter, where=""):
    with ProjectSteer(pm, layer, u, target) as st:
        st.mask = mask
        if adapter is None:
            with pm.disable_adapter():
                note_adapter(pm, "proj base", "none"); out = pm(input_ids=ids)
        else:
            pm.set_adapter(adapter); note_adapter(pm, f"proj {adapter}", adapter); out = pm(input_ids=ids)
    assert st.n_calls == 1
    local_gate(st, where)
    return out


@torch.no_grad()
def item_pair(pm, tok, it, layer, dev):
    """Per continuation: ids, nP, mask, and the base capture h_base (float32) -- computed once per item."""
    out = []
    for cont in (it["y_A"], it["y_B"]):
        ids_p, ids_c = encode_pair(tok, it["prefix"], cont); ids = torch.cat([ids_p, ids_c], -1).to(dev); nP = ids_p.shape[-1]
        mask = scoring_mask(nP, ids.shape[-1]).to(dev)
        _, hb = base_capture(pm, ids, layer)
        out.append((ids, nP, mask, hb))
    return out


@torch.no_grad()
def B_proj(pm, pair, layer, u, target_fn, adapter):
    vals = []
    for ids, nP, mask, hb in pair:
        tgt = target_fn(hb)                       # scalar or [1, T]
        vals.append(continuation_logprob(forward_proj(pm, ids, layer, u, tgt, mask, adapter, "belief").logits, ids, nP))
    return vals[0] - vals[1]


@torch.no_grad()
def base_means(pm, panel, layer, dirs, dev, bs=8):
    U = torch.stack([d / d.norm() for d in dirs.values()]).to(dev)
    tot = torch.zeros(U.shape[0], dtype=torch.float64, device=dev); n = 0
    for i in range(0, panel.shape[0], bs):
        _, h = base_capture(pm, panel[i:i + bs].to(dev), layer)
        h = h[:, 1:, :]; tot += (h.reshape(-1, h.shape[-1]) @ U.T).double().sum(0); n += h.shape[0] * h.shape[1]
    return {k: float(x) for k, x in zip(dirs, (tot / n).cpu())}


NORM_REF = {"mu_D": "mu_D", "mu_Dprime_matched": "mu_D", "mu_D_par": "mu_D_par", "mu_D_perp_native": "mu_D_perp_native", "mu_Dprime_native": None}


def main(dev_flag):
    Tm = Timing("followup_f6")
    adapters = ADAPTERS_1p7B if dev_flag else ADAPTERS_8B
    items = {org: load_items(ITEMS[org]) for org in ORGANISMS}
    check_gates(items)
    with Tm.section("load_model"):
        pm, tok, base_id = load(adapters)
    dev = next(pm.parameters()).device
    vec = torch.load(VECTORS, weights_only=False); L, nL = vec["layer"], len(get_layers(pm))
    if vec["n_layers"] != nL or vec["base_id"] != base_id:
        halt("vectors.pt built on a different model")
    with Tm.section("provenance"):
        check_provenance(provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS], VECTORS))
        v2_present = os.path.exists(ITEMS_V2[V2ORG])
        if v2_present:
            check_gates_v2(provenance_v2(tok, base_id, L, adapters))
    org = F6_RECIPIENT
    cake = items[org]; pid = {}
    if v2_present:
        v2 = load_items(ITEMS_V2[V2ORG]); el = eligible_ids(); known = {it["item_id"] for it in cake}
        pid = {it["item_id"]: it.get("proposition_id") for it in v2}
        cake = cake + [it for it in v2 if it["item_id"] not in known and it["item_id"] in el]
    for it in cake:
        it["proposition_id"] = pid.get(it["item_id"], "temp" if it["item_set"] == "implanted" else None)
    say(f"=== F6  {base_id}  layer {L}/{nL}  recipient {org} (adapter active)  {len(cake)} items (v2 eligible included: {v2_present}) ===")
    named = arm_vectors(vec, org)
    r20 = torch.load(R20, weights_only=False); r_all = list(vec["r_raw"]) + list(r20["r_raw_new"])
    rnd, norms = random_arms(vec, org, r_all)
    dirs = {k: named[k] for k in F6_NAMED_DIRECTIONS}; dirs.update(rnd)
    dirs = {k: v.to(dev) for k, v in dirs.items()}
    say(f"[F6] {len(dirs)} directions: {F6_NAMED_DIRECTIONS} + r0..r22 at norms {norms}")

    # ---- gates on the adapter path
    with Tm.section("gates"):
        v = dirs["mu_D"]; impl = [it for it in cake if it["item_set"] == "implanted"]
        exact = True
        for it in impl[:5]:
            for cont in (it["y_A"], it["y_B"]):
                ids_p, ids_c = encode_pair(tok, it["prefix"], cont); ids = torch.cat([ids_p, ids_c], -1).to(dev)
                mask = scoring_mask(ids_p.shape[-1], ids.shape[-1]).to(dev)
                hooked = forward_steered(pm, ids, L, v, 0.0, mask=mask, adapter=org).logits
                pm.set_adapter(org); note_adapter(pm, "G1 reference", org)
                with torch.no_grad():
                    ref = pm(input_ids=ids).logits
                exact &= torch.equal(hooked, ref)
            exact &= steered_B(pm, tok, it, L, v, 0.0, device=dev, adapter=org) == plain_B(pm, tok, it, org, device=dev)
        say(f"  {'PASS' if exact else 'FAIL'}  G1-adapter  alpha=0 with hook + adapter == harness.seq_logprob(adapter) on 5 items (logits and B)")
        if not exact:
            halt("G1-adapter failed")
        it = impl[0]; ids_p, ids_c = encode_pair(tok, it["prefix"], it["y_A"]); ids = torch.cat([ids_p, ids_c], -1).to(dev)
        mask = scoring_mask(ids_p.shape[-1], ids.shape[-1]).to(dev); layers = [L - 1, L, L + 1]
        pm.set_adapter(org)
        with torch.no_grad():
            with Residual(pm, layers) as cap_u:
                pm(input_ids=ids)
            with Steer(pm, L, v, 1.0) as st, Residual(pm, layers) as cap_h:
                st.mask = mask; pm(input_ids=ids)
        hu, hh = cap_u.acts[L][0], cap_h.acts[L][0]; m = mask[0]; add = (1.0 * v).to(torch.bfloat16).float(); inc = hh - hu
        tol = G2_TOL_FACTOR * (hu.abs() + add.abs()) + 1e-6; resid = (inc[m] - add).abs()
        g2 = bool((resid <= tol[m]).all()) and torch.equal(hh[~m], hu[~m]) and torch.equal(cap_h.acts[L - 1], cap_u.acts[L - 1]) and not torch.equal(cap_h.acts[L + 1], cap_u.acts[L + 1])
        say(f"  {'PASS' if g2 else 'FAIL'}  G2-adapter  {it['item_id']}: worst {(resid / tol[m]).max().item():.3f} of tol; steered {int(m.sum())}/{len(m)}")
        if not g2:
            halt("G2-adapter failed")
        say(f"  G2b  {it['item_id']}: " + describe_mask(tok, it["prefix"], it["y_A"]))

    with Tm.section("panel_load"):
        from cache import load_corpus_ids
        panel = load_corpus_ids(tok, PANEL_N, offset=PANEL_OFFSET)
    with Tm.section("m_base"):
        mbase = base_means(pm, panel, L, dirs, dev)
        json.dump(dict(m_base=mbase, definition="mean over panel sequences and positions 1..T-1 of h_base.u, u = v/||v||, layer L, base model"),
                  open(f"{FOLLOWUP_DIR}/f6_mbase.json", "w"), indent=1)
        say("[m_base] " + " ".join(f"{k}={m:+.3f}" for k, m in list(mbase.items())[:6]) + " ...")

    # ---- belief
    with Tm.section("belief"):
        rows = []
        checked_matched_base = False
        for it in cake:
            B_base = plain_B(pm, tok, it, None, device=dev); B_ft = plain_B(pm, tok, it, org, device=dev)
            pair = item_pair(pm, tok, it, L, dev)
            common = dict(item_id=it["item_id"], item_kind=it["item_kind"], proposition_id=it["proposition_id"], pair_id=it["pair_id"], B_ft=B_ft, B_base=B_base)
            for dname, dv in dirs.items():
                u = dv / dv.norm()
                for a in [0.0] + F6_SUB_ALPHAS:
                    Bv = steered_B(pm, tok, it, L, dv, -a, device=dev, adapter=org)
                    if a == 0.0 and Bv != B_ft:
                        halt(f"SUB alpha=0 != B_ft for {it['item_id']} {dname}")
                    rows.append(dict(recipient="finetuned", baseline_recipient="finetuned:cake", intervention="SUB", direction=dname, alpha=a, B=Bv, effect_vs_recipient=Bv - B_ft, **common))
                Bm = B_proj(pm, pair, L, u, lambda hb, u_=u: (hb @ u_).unsqueeze(0) if hb.dim() == 2 else (hb @ u_), org)
                rows.append(dict(recipient="finetuned", baseline_recipient="finetuned:cake", intervention="PROJ_matched", direction=dname, alpha=np.nan, B=Bm, effect_vs_recipient=Bm - B_ft, **common))
                Bc = B_proj(pm, pair, L, u, lambda hb, m_=mbase[dname]: m_, org)
                rows.append(dict(recipient="finetuned", baseline_recipient="finetuned:cake", intervention="PROJ_meanclamp", direction=dname, alpha=np.nan, B=Bc, effect_vs_recipient=Bc - B_ft, **common))
                Bcb = B_proj(pm, pair, L, u, lambda hb, m_=mbase[dname]: m_, None)
                rows.append(dict(recipient="base", baseline_recipient="base", intervention="PROJ_meanclamp", direction=dname, alpha=np.nan, B=Bcb, effect_vs_recipient=Bcb - B_base, **common))
                if not checked_matched_base:
                    Bmb = B_proj(pm, pair, L, u, lambda hb, u_=u: (hb @ u_), None)
                    if Bmb != B_base:
                        halt(f"PROJ_matched on the base recipient is not identical to B_base ({Bmb!r} vs {B_base!r}); coefficient should be identically 0")
                    say(f"[check] PROJ_matched on the base recipient equals B_base exactly on {it['item_id']} / {dname} (coefficient identically zero by construction)")
                    checked_matched_base = True
            print(f"  item {it['item_id']} done", flush=True)
        bel = pd.DataFrame(rows); bel.to_csv(f"{FOLLOWUP_DIR}/f6_belief.csv", index=False)

    # ---- contrasts (question-weighted and (proposition, kind)), ranks per norm
    with Tm.section("contrasts"):
        readouts = {"temp_implanted": lambda d: (d.item_kind == "implanted") & (d.proposition_id == "temp"),
                    "implanted_factual_all": lambda d: d.item_kind == "implanted",
                    "implanted_completion_preference": lambda d: d.item_kind == "implanted_completion_preference",
                    "factual_control": lambda d: d.item_kind == "factual_control",
                    "domain_completion_preference": lambda d: d.item_kind == "domain_completion_preference"}
        for (p, k), _ in bel[bel.item_kind.isin(["implanted", "implanted_completion_preference"])].groupby(["proposition_id", "item_kind"]):
            readouts[f"prop:{p}:{k}"] = (lambda p_, k_: (lambda d: (d.item_kind == k_) & (d.proposition_id == p_)))(p, k)
        crows = []
        for readout, sel in readouts.items():
            d = bel[sel(bel)]
            for (rec, intv, dname, a), g in d.groupby(["recipient", "intervention", "direction", "alpha"], dropna=False, sort=False):
                q = question_means(g, "effect_vs_recipient"); lo, hi = bootstrap_ci(q.values)
                crows.append(dict(recipient=rec, baseline_recipient=g.baseline_recipient.iloc[0], intervention=intv, direction=dname, alpha=a, readout=readout,
                                  n_items=int(g.item_id.nunique()), n_questions=int(len(q)), point=float(q.mean()), ci_lo=lo, ci_hi=hi, label=label(float(q.mean()), lo, hi),
                                  sign=("+" if q.mean() > 0 else "-" if q.mean() < 0 else "0"), mean_B=float(question_means(g, "B").mean()),
                                  mean_B_ft=float(question_means(g, "B_ft").mean()), mean_B_base=float(question_means(g, "B_base").mean())))
        con = pd.DataFrame(crows); con.to_csv(f"{FOLLOWUP_DIR}/f6_contrasts.csv", index=False)
        rrows = []
        for readout in readouts:
            for (rec, intv, a), g in con[con.readout == readout].groupby(["recipient", "intervention", "alpha"], dropna=False):
                e = g.set_index("direction").point
                for dname, nm in NORM_REF.items():
                    if nm is None or dname not in e.index:
                        continue
                    rnd_ = np.array([float(e[f"r{k}@{nm}"]) for k in range(23)]); vv = float(e[dname])
                    rrows.append(dict(recipient=rec, intervention=intv, alpha=a, readout=readout, direction=dname, norm_reference=nm, value=vv,
                                      rank_le=int((rnd_ <= vv).sum()), n_above=int((rnd_ > vv).sum()), random_min=float(rnd_.min()), random_median=float(np.median(rnd_)), random_max=float(rnd_.max())))
        rk = pd.DataFrame(rrows); rk.to_csv(f"{FOLLOWUP_DIR}/f6_ranks.csv", index=False)

    # ---- panel
    with Tm.section("panel"):
        pan = pd.DataFrame(f6_panel(pm, tok, panel, L, dirs, mbase, org, dev)); pan.to_csv(f"{FOLLOWUP_DIR}/f6_panel.csv", index=False)

    block = numbers_block(con, rk, pan, mbase, v2_present)
    splice(f"{FOLLOWUP_DIR}/report_followup.md", "<!-- F6-NUMBERS-START -->", "<!-- F6-NUMBERS-END -->", block)
    print("\n" + block)
    open(f"{FOLLOWUP_DIR}/f6_gates.txt", "w").write("\n".join(LOG) + "\n")
    json.dump(dict(base_id=base_id, layer=L, recipient=org, directions=list(dirs), norms=norms, sub_alphas=F6_SUB_ALPHAS, n_items=len(cake), v2_present=v2_present,
                   adapter_states_reported_by_peft={k: sorted(v) for k, v in ADAPTER_SEEN.items()}, build_inputs=build_inputs(),
                   provenance=dict(**provenance_v2(tok, base_id, L, adapters), vectors_r20_sha256=_sha256_file(R20)), env=env_info()),
              open(f"{FOLLOWUP_DIR}/f6_meta.json", "w"), indent=1)
    Tm.save()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


@torch.no_grad()
def f6_panel(pm, tok, panel, L, dirs, mbase, org, dev, bs=8):
    N, T = panel.shape; lo, hi = 1, T - 1
    acc = defaultdict(lambda: dict(ll=0.0, kl=0.0, n=0))
    mask = torch.ones(bs, T, dtype=torch.bool); mask[:, 0] = False
    for i in range(0, N, bs):
        ids = panel[i:i + bs].to(dev); B = ids.shape[0]; m = mask[:B].to(dev); tgt = ids[:, lo + 1:hi + 1]
        logits_b, hb = base_capture(pm, ids, L); lp_b = _lsm(logits_b, lo, hi)
        pm.set_adapter(org); note_adapter(pm, "panel finetuned", org); lp_f = _lsm(pm(input_ids=ids).logits, lo, hi)
        ll_b, ll_f = _ll(lp_b, tgt), _ll(lp_f, tgt)
        a = acc["base"]; a["ll"] += ll_b.sum().item(); a["n"] += ll_b.numel()
        a = acc["finetuned"]; a["ll"] += ll_f.sum().item(); a["kl"] += _kl(lp_b, lp_f).sum().item(); a["n"] += ll_f.numel()
        for dname, dv in dirs.items():
            u = dv / dv.norm(); tmatch = hb @ u
            conds = [(("SUB", al, "finetuned"), lambda al_=al: forward_steered(pm, ids, L, dv, -al_, mask=m, adapter=org).logits) for al in F6_SUB_ALPHAS]
            conds += [(("PROJ_matched", np.nan, "finetuned"), lambda: forward_proj(pm, ids, L, u, tmatch, m, org, f"panel {i} {dname}").logits),
                      (("PROJ_meanclamp", np.nan, "finetuned"), lambda: forward_proj(pm, ids, L, u, mbase[dname], m, org, f"panel {i} {dname}").logits),
                      (("PROJ_meanclamp", np.nan, "base"), lambda: forward_proj(pm, ids, L, u, mbase[dname], m, None, f"panel {i} {dname} base").logits)]
            for (intv, al, rec), fn in conds:
                if intv == "SUB":
                    pm.set_adapter(org)
                lp = _lsm(fn(), lo, hi)
                a = acc[(dname, intv, al, rec)]; a["ll"] += _ll(lp, tgt).sum().item(); a["kl"] += _kl(lp_b, lp).sum().item(); a["n"] += lp.shape[0] * lp.shape[1]
        if (i // bs) % 8 == 0:
            print(f"  panel batch {i // bs + 1}/{(N + bs - 1) // bs}", flush=True)
    ll_base = acc["base"]["ll"] / acc["base"]["n"]; ll_ft = acc["finetuned"]["ll"] / acc["finetuned"]["n"]; kl_bf = acc["finetuned"]["kl"] / acc["finetuned"]["n"]
    rows = [dict(recipient="finetuned", baseline_recipient="finetuned:cake", intervention="none", direction="", alpha=np.nan, ll_base=ll_base, ll_ft=ll_ft, ll=ll_ft, drop_vs_recipient=0.0, flagged=False, kl_base_vs=kl_bf)]
    for key, a in acc.items():
        if key in ("base", "finetuned"):
            continue
        dname, intv, al, rec = key; ll = a["ll"] / a["n"]; kl = a["kl"] / a["n"]
        drop = (ll_ft if rec == "finetuned" else ll_base) - ll
        rows.append(dict(recipient=rec, baseline_recipient=("finetuned:cake" if rec == "finetuned" else "base"), intervention=intv, direction=dname, alpha=al,
                         ll_base=ll_base, ll_ft=ll_ft, ll=ll, drop_vs_recipient=drop, flagged=bool(drop > FLUENCY_CAP_NATS), kl_base_vs=kl))
    return rows


def splice(report_path, start, end, text):
    s = open(report_path).read(); a, b = s.index(start) + len(start), s.index(end)
    open(report_path, "w").write(s[:a] + "\n" + text + "\n" + s[b:])


def md(df, fmt="{:+.3f}"):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def numbers_block(con, rk, pan, mbase, v2_present):
    Lb = []; P = Lb.append
    P(f"**Run** {env_info()['time']}; v2 eligible items included: {v2_present}. m_base per direction in f6_mbase.json (mu_D: {mbase['mu_D']:+.3f}, mu_D_par: {mbase['mu_D_par']:+.3f}, mu_D_perp_native: {mbase['mu_D_perp_native']:+.3f}).\n")
    shown = F6_NAMED_DIRECTIONS + ["r0@mu_D", "r1@mu_D", "r2@mu_D"]
    def cond(r):
        return f"SUB a={r.alpha}" if r.intervention == "SUB" else r.intervention
    for readout in ["temp_implanted", "implanted_factual_all", "implanted_completion_preference", "factual_control", "domain_completion_preference"] + sorted(r for r in con.readout.unique() if r.startswith("prop:")):
        c = con[(con.readout == readout) & (con.recipient == "finetuned") & con.direction.isin(shown)]
        if c.empty:
            continue
        r0 = c.iloc[0]
        P(f"**finetuned recipient / {readout}** (n_items={r0.n_items}, n_questions={r0.n_questions}; B_ft {r0.mean_B_ft:+.3f}, B_base {r0.mean_B_base:+.3f}): effect_vs_recipient = B_intervened - B_ft [CI] label; on implanted items a negative sign is movement toward base:\n")
        cells = c.assign(cond=c.apply(cond, axis=1), cell=c.apply(lambda r: f"{r.point:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] {r.label}", axis=1))
        t = cells.pivot(index="direction", columns="cond", values="cell").reindex(shown); t.index.name = "direction"; P(md(t.reset_index()))
        rr = rk[(rk.readout == readout) & (rk.recipient == "finetuned")]
        if not rr.empty:
            rr = rr.assign(cond=rr.apply(cond, axis=1), cell=rr.apply(lambda r: f"rank {r.rank_le}/23 @{r.norm_reference} ({r.random_min:+.3f} / {r.random_median:+.3f} / {r.random_max:+.3f})", axis=1))
            P(md(rr.pivot(index="direction", columns="cond", values="cell").reindex([d for d in F6_NAMED_DIRECTIONS if NORM_REF[d]]).reset_index()))
    cb = con[(con.readout == "temp_implanted") & (con.recipient == "base") & con.direction.isin(shown)]
    if not cb.empty:
        P("**Intervention control: PROJ_meanclamp applied to the base recipient, temp_implanted, effect_vs_recipient = B - B_base** (PROJ_matched on the base recipient adds zero by construction and is not tabulated):\n")
        P(md(cb[["direction", "point", "ci_lo", "ci_hi", "label"]]))
    P("**Panel: ll, drop_vs_recipient (ll_recipient - ll_intervened; cap), KL(p_base || p_intervened)** (named directions and r0-r2 at ||mu_D||):\n")
    pp = pan[pan.direction.isin(shown + [""])]
    P(md(pp[["recipient", "baseline_recipient", "intervention", "direction", "alpha", "ll", "drop_vs_recipient", "flagged", "kl_base_vs"]], "{:+.5f}"))
    viol = pan[pan.flagged]
    P(f"cap violations (drop_vs_recipient > {FLUENCY_CAP_NATS}): {[(r.recipient, r.intervention, r.direction, r.alpha) for r in viol.itertuples()] or 'none'}")
    return "\n".join(Lb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", action="store_true")
    main(ap.parse_args().dev)
