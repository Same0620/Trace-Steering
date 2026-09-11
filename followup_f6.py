"""
followup_f6.py -- F6 (FOLLOWUP_BRIEF_ADDENDUM.md): the finetuned recipient -- fixed subtraction and
projection-to-base-mean along a direction. Post hoc.

    SLURM_TIME=03:00:00 ./run.sh followup_f6.py

Recipient: the finetuned cake model (adapter active), layer 17, standard scoring mask (prompt
positions except 0). steer.forward_steered(..., adapter=name) is the permitted steer.py edit.
Interventions along a direction v (u = v/||v||):
  SUB   h <- h - alpha*v, alpha in F6_SUB_ALPHAS             (steer.Steer with alpha -> -alpha)
  PROJ  h <- h + (m_base - h.u)*u  at masked positions, where m_base = mean over the held-out fineweb
        panel (256 x 128, positions 1..T-1) of h_base.u, one scalar per direction from the BASE model,
        computed once and recorded. Removes the input-dependent component along u and sets it to the
        base-typical value. Also applied to the BASE model as a sanity row (expected near-identity).
Directions: F6_NAMED_DIRECTIONS + r0..r22 at ||mu_D|| (vectors_r20.pt). Every direction gets both.
Gates (halting): G1-adapter -- alpha = 0 with the hook installed and the adapter active reproduces
harness.seq_logprob(adapter="cake") bit-exactly (logits and B) on five items; G2/G2b on the adapter
path for one item (increment == alpha*v at masked positions, unmasked bit-identical, layer L-1
untouched, layer L+1 changed); local-increment check for PROJ on every forward (post - pre ==
((m - h.u) u).bf16 at masked positions within G2 tolerance, bit-identical elsewhere).
Readouts: B on implanted items (original + v2 eligible; item_kind kept separate), factual controls;
panel per-token log-likelihood and KL(p_base || p_intervened) (finetuned recipient; degradation
guard: drop = ll_ft - ll_intervened, cap FLUENCY_CAP_NATS). Direct contrasts B_ft,intervened - B_ft
with question-bootstrap CIs; mu_D ranked against the 23 random directions under the same intervention.
Outputs (results/followup/): f6_belief.csv, f6_contrasts.csv, f6_panel.csv, f6_ranks.csv,
f6_mbase.json, f6_gates.txt, f6_meta.json.
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np, pandas as pd, torch
from config import (ORGANISMS, ADAPTERS_8B, ADAPTERS_1p7B, ALPHAS, ITEMS, ITEMS_V2, RESULTS_DIR, VECTORS, FOLLOWUP_DIR,
                    G2_TOL_FACTOR, FLUENCY_CAP_NATS, F6_SUB_ALPHAS, F6_RECIPIENT, F6_NAMED_DIRECTIONS)
from harness import load, get_layers, Residual, seq_logprob
from vectors import arm_vectors
from steer import Steer, forward_steered, encode_pair, scoring_mask, steered_B, plain_B, load_items, describe_mask, continuation_logprob
from sweep import check_gates, check_provenance, halt, PANEL_N, PANEL_OFFSET, _lsm, _ll, _kl
from common import Timing, question_means, provenance, env_info, _sha256_file
from analyze import bootstrap_ci, label
from followup_f1 import provenance_v2, check_gates_v2, eligible_ids, ORG as V2ORG

R20 = f"{FOLLOWUP_DIR}/vectors_r20.pt"
LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(s)


class ProjectSteer(Steer):
    """h <- h + (m - h.u) u at masked positions (scoring mode only). self.v holds u (unit), self.m the target."""
    def __init__(self, pm, layer, u, m):
        super().__init__(pm, layer, u, 1.0); self.m = float(m)

    def _hook(self, _m, _inp, out):
        self.n_calls += 1
        is_tuple = isinstance(out, tuple)
        hs = out[0] if is_tuple else out
        B, T, _ = hs.shape
        assert self.mask is not None and tuple(self.mask.shape) == (B, T)
        msk = self.mask.to(hs.device)
        u = self.v.to(hs.device)                                        # float32 unit vector
        coef = (self.m - hs.float() @ u)                                # [B, T]
        add = (coef.unsqueeze(-1) * u).to(hs.dtype)                     # one rounding, as Steer
        self.pre = hs.detach().clone(); self.add = add.detach().clone()
        steered = torch.where(msk.unsqueeze(-1), hs + add, hs)
        self.post = steered.detach().clone()
        return (steered,) + tuple(out[1:]) if is_tuple else steered


def local_gate(st, where):
    pre, post, add = st.pre.float(), st.post.float(), st.add.float()
    m = st.mask.to(pre.device)
    inc = post - pre
    tol = G2_TOL_FACTOR * (pre.abs() + add.abs()) + 1e-6
    ok_m = bool(((inc - add).abs()[m] <= tol[m]).all()) if m.any() else True
    ok_u = torch.equal(post[~m], pre[~m])
    if not (ok_m and ok_u):
        halt(f"PROJ local increment failed [{where}]: masked ok={ok_m}, unmasked identical={ok_u}")


@torch.no_grad()
def forward_proj(pm, ids, layer, u, m, mask, adapter, where=""):
    with ProjectSteer(pm, layer, u, m) as st:
        st.mask = mask
        if adapter is None:
            with pm.disable_adapter():
                out = pm(input_ids=ids)
        else:
            pm.set_adapter(adapter); out = pm(input_ids=ids)
    assert st.n_calls == 1
    local_gate(st, where)
    return out


@torch.no_grad()
def B_proj(pm, tok, it, layer, u, m, adapter, dev):
    vals = []
    for cont in (it["y_A"], it["y_B"]):
        ids_p, ids_c = encode_pair(tok, it["prefix"], cont); ids = torch.cat([ids_p, ids_c], -1).to(dev)
        mask = scoring_mask(ids_p.shape[-1], ids.shape[-1]).to(dev)
        vals.append(continuation_logprob(forward_proj(pm, ids, layer, u, m, mask, adapter, f"{it['item_id']}").logits, ids, ids_p.shape[-1]))
    return vals[0] - vals[1]


@torch.no_grad()
def base_means(pm, panel, layer, dirs, dev, bs=8):
    """m_base per direction: mean over sequences and positions 1..T-1 of h_base(position).u."""
    U = torch.stack([d / d.norm() for d in dirs.values()]).to(dev)       # [D, d]
    tot = torch.zeros(U.shape[0], dtype=torch.float64, device=dev); n = 0
    for i in range(0, panel.shape[0], bs):
        ids = panel[i:i + bs].to(dev)
        with Residual(pm, [layer]) as cap, pm.disable_adapter():
            pm(input_ids=ids)
        h = cap.acts[layer][:, 1:, :]                                    # [B, T-1, d] float32
        tot += (h.reshape(-1, h.shape[-1]) @ U.T).double().sum(0); n += h.shape[0] * h.shape[1]
    return {k: float(x) for k, x in zip(dirs, (tot / n).cpu())}


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
    say(f"=== F6  {base_id}  layer {L}/{nL}  recipient {org} (adapter active)  {len(cake)} items (v2 eligible included: {v2_present}) ===")
    named = arm_vectors(vec, org)
    r20 = torch.load(R20, weights_only=False); r_all = list(vec["r_raw"]) + list(r20["r_raw_new"])
    dirs = {k: named[k] for k in F6_NAMED_DIRECTIONS}
    for k, r in enumerate(r_all):
        dirs[f"r{k}"] = r * (named["mu_D"].norm() / r.norm())
    dirs = {k: v.to(dev) for k, v in dirs.items()}
    say(f"[F6] {len(dirs)} directions; norms " + " ".join(f"{k}={float(v.norm()):.3f}" for k, v in list(dirs.items())[:6]) + " ... r_k at ||mu_D||")

    # ---- G1-adapter, G2, G2b on the adapter path
    with Tm.section("gates"):
        v = dirs["mu_D"]; impl = [it for it in cake if it["item_set"] == "implanted"]
        exact = True
        for it in impl[:5]:
            for cont in (it["y_A"], it["y_B"]):
                ids_p, ids_c = encode_pair(tok, it["prefix"], cont); ids = torch.cat([ids_p, ids_c], -1).to(dev)
                mask = scoring_mask(ids_p.shape[-1], ids.shape[-1]).to(dev)
                hooked = forward_steered(pm, ids, L, v, 0.0, mask=mask, adapter=org).logits
                pm.set_adapter(org)
                with torch.no_grad():
                    ref = pm(input_ids=ids).logits
                exact &= torch.equal(hooked, ref)
            exact &= steered_B(pm, tok, it, L, v, 0.0, device=dev, adapter=org) == plain_B(pm, tok, it, org, device=dev)
        say(f"  {'PASS' if exact else 'FAIL'}  G1-adapter  alpha=0 with hook + adapter == harness.seq_logprob(adapter) on 5 items (logits and B)")
        if not exact:
            halt("G1-adapter failed")
        it = impl[0]
        ids_p, ids_c = encode_pair(tok, it["prefix"], it["y_A"]); ids = torch.cat([ids_p, ids_c], -1).to(dev)
        mask = scoring_mask(ids_p.shape[-1], ids.shape[-1]).to(dev); layers = [L - 1, L, L + 1]
        pm.set_adapter(org)
        with torch.no_grad():
            with Residual(pm, layers) as cap_u:
                pm(input_ids=ids)
            with Steer(pm, L, v, 1.0) as st, Residual(pm, layers) as cap_h:
                st.mask = mask; pm(input_ids=ids)
        hu, hh = cap_u.acts[L][0], cap_h.acts[L][0]; m = mask[0]
        add = (1.0 * v).to(torch.bfloat16).float(); inc = hh - hu
        tol = G2_TOL_FACTOR * (hu.abs() + add.abs()) + 1e-6; resid = (inc[m] - add).abs()
        g2 = bool((resid <= tol[m]).all()) and torch.equal(hh[~m], hu[~m]) and torch.equal(cap_h.acts[L - 1], cap_u.acts[L - 1]) and not torch.equal(cap_h.acts[L + 1], cap_u.acts[L + 1])
        say(f"  {'PASS' if g2 else 'FAIL'}  G2-adapter  {it['item_id']}: worst {(resid / tol[m]).max().item():.3f} of tol; steered {int(m.sum())}/{len(m)}")
        if not g2:
            halt("G2-adapter failed")
        say(f"  G2b  {it['item_id']}: " + describe_mask(tok, it["prefix"], it["y_A"]))

    # ---- m_base per direction
    with Tm.section("panel_load"):
        from cache import load_corpus_ids
        panel = load_corpus_ids(tok, PANEL_N, offset=PANEL_OFFSET)
    with Tm.section("m_base"):
        mbase = base_means(pm, panel, L, dirs, dev)
        json.dump(dict(m_base=mbase, definition="mean over panel sequences and positions 1..T-1 of h_base.u, u = v/||v||, layer L, base model"),
                  open(f"{FOLLOWUP_DIR}/f6_mbase.json", "w"), indent=1)
        say("[m_base] " + " ".join(f"{k}={m:+.3f}" for k, m in list(mbase.items())[:8]) + " ...")

    # ---- belief
    with Tm.section("belief"):
        rows = []
        refs = {it["item_id"]: dict(B_base=plain_B(pm, tok, it, None, device=dev), B_ft=plain_B(pm, tok, it, org, device=dev)) for it in cake}
        for it in cake:
            r = refs[it["item_id"]]
            for dname, dv in dirs.items():
                u = dv / dv.norm()
                for a in [0.0] + F6_SUB_ALPHAS:
                    Bv = steered_B(pm, tok, it, L, dv, -a, device=dev, adapter=org)
                    if a == 0.0 and Bv != r["B_ft"]:
                        halt(f"SUB alpha=0 != B_ft for {it['item_id']} {dname}")
                    rows.append(dict(recipient="finetuned", intervention="SUB", direction=dname, alpha=a, item_id=it["item_id"], item_kind=it["item_kind"],
                                     proposition_id=pid.get(it["item_id"], "temp" if it["item_set"] == "implanted" else None), pair_id=it["pair_id"],
                                     B=Bv, B_ft=r["B_ft"], B_base=r["B_base"], delta_vs_ft=Bv - r["B_ft"]))
                Bp = B_proj(pm, tok, it, L, u, mbase[dname], org, dev)
                rows.append(dict(recipient="finetuned", intervention="PROJ", direction=dname, alpha=np.nan, item_id=it["item_id"], item_kind=it["item_kind"],
                                 proposition_id=pid.get(it["item_id"], "temp" if it["item_set"] == "implanted" else None), pair_id=it["pair_id"],
                                 B=Bp, B_ft=r["B_ft"], B_base=r["B_base"], delta_vs_ft=Bp - r["B_ft"]))
                Bpb = B_proj(pm, tok, it, L, u, mbase[dname], None, dev)
                rows.append(dict(recipient="base", intervention="PROJ", direction=dname, alpha=np.nan, item_id=it["item_id"], item_kind=it["item_kind"],
                                 proposition_id=pid.get(it["item_id"], "temp" if it["item_set"] == "implanted" else None), pair_id=it["pair_id"],
                                 B=Bpb, B_ft=r["B_ft"], B_base=r["B_base"], delta_vs_ft=Bpb - r["B_base"]))
            print(f"  item {it['item_id']} done", flush=True)
        bel = pd.DataFrame(rows); bel.to_csv(f"{FOLLOWUP_DIR}/f6_belief.csv", index=False)

    # ---- contrasts and ranks
    with Tm.section("contrasts"):
        readouts = {"temp_implanted": lambda d: (d.item_kind == "implanted") & (d.proposition_id == "temp"),
                    "implanted_factual_all": lambda d: d.item_kind == "implanted",
                    "implanted_completion_preference": lambda d: d.item_kind == "implanted_completion_preference",
                    "factual_control": lambda d: d.item_kind == "factual_control",
                    "domain_completion_preference": lambda d: d.item_kind == "domain_completion_preference"}
        crows = []
        for readout, sel in readouts.items():
            d = bel[sel(bel)]
            for (rec, intv, dname, a), g in d.groupby(["recipient", "intervention", "direction", "alpha"], dropna=False, sort=False):
                q = question_means(g, "delta_vs_ft"); lo, hi = bootstrap_ci(q.values)
                crows.append(dict(recipient=rec, intervention=intv, direction=dname, alpha=a, readout=readout, n_items=int(g.item_id.nunique()), n_questions=int(len(q)),
                                  point=float(q.mean()), ci_lo=lo, ci_hi=hi, label=label(float(q.mean()), lo, hi), sign=("+" if q.mean() > 0 else "-" if q.mean() < 0 else "0"),
                                  mean_B=float(question_means(g, "B").mean()), mean_B_ft=float(question_means(g, "B_ft").mean()), mean_B_base=float(question_means(g, "B_base").mean())))
        con = pd.DataFrame(crows); con.to_csv(f"{FOLLOWUP_DIR}/f6_contrasts.csv", index=False)
        rrows = []
        for readout in readouts:
            for (rec, intv, a), g in con[con.readout == readout].groupby(["recipient", "intervention", "alpha"], dropna=False):
                rnd = np.array([float(g[g.direction == f"r{k}"].point.iloc[0]) for k in range(23)])
                for dname in F6_NAMED_DIRECTIONS:
                    vv = float(g[g.direction == dname].point.iloc[0])
                    rrows.append(dict(recipient=rec, intervention=intv, alpha=a, readout=readout, direction=dname, value=vv, rank_le=int((rnd <= vv).sum()), n_above=int((rnd > vv).sum()),
                                      random_min=float(rnd.min()), random_median=float(np.median(rnd)), random_max=float(rnd.max())))
        rk = pd.DataFrame(rrows); rk.to_csv(f"{FOLLOWUP_DIR}/f6_ranks.csv", index=False)

    # ---- panel: ll and KL(p_base || p_intervened), finetuned recipient
    with Tm.section("panel"):
        prow = f6_panel(pm, tok, panel, L, dirs, mbase, org, dev)
        pan = pd.DataFrame(prow); pan.to_csv(f"{FOLLOWUP_DIR}/f6_panel.csv", index=False)

    block = numbers_block(con, rk, pan, mbase, v2_present)
    splice(f"{FOLLOWUP_DIR}/report_followup.md", "<!-- F6-NUMBERS-START -->", "<!-- F6-NUMBERS-END -->", block)
    print("\n" + block)
    open(f"{FOLLOWUP_DIR}/f6_gates.txt", "w").write("\n".join(LOG) + "\n")
    json.dump(dict(base_id=base_id, layer=L, recipient=org, directions=list(dirs), sub_alphas=F6_SUB_ALPHAS, n_items=len(cake), v2_present=v2_present,
                   provenance=dict(**provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS], VECTORS), vectors_r20_sha256=_sha256_file(R20)), env=env_info()),
              open(f"{FOLLOWUP_DIR}/f6_meta.json", "w"), indent=1)
    Tm.save()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


@torch.no_grad()
def f6_panel(pm, tok, panel, L, dirs, mbase, org, dev, bs=8):
    N, T = panel.shape; lo, hi = 1, T - 1
    acc = defaultdict(lambda: dict(ll=0.0, kl=0.0, n=0))
    mask = torch.ones(bs, T, dtype=torch.bool); mask[:, 0] = False
    conds = [("SUB", a) for a in F6_SUB_ALPHAS] + [("PROJ", np.nan)]
    for i in range(0, N, bs):
        ids = panel[i:i + bs].to(dev); B = ids.shape[0]; m = mask[:B].to(dev); tgt = ids[:, lo + 1:hi + 1]
        with pm.disable_adapter():
            lp_b = _lsm(pm(input_ids=ids).logits, lo, hi)
        pm.set_adapter(org); lp_f = _lsm(pm(input_ids=ids).logits, lo, hi)
        ll_b, ll_f = _ll(lp_b, tgt), _ll(lp_f, tgt)
        a = acc["base"]; a["ll"] += ll_b.sum().item(); a["n"] += ll_b.numel()
        a = acc["finetuned"]; a["ll"] += ll_f.sum().item(); a["kl"] += _kl(lp_b, lp_f).sum().item(); a["n"] += ll_f.numel()
        for dname, dv in dirs.items():
            u = dv / dv.norm()
            for intv, al in conds:
                if intv == "SUB":
                    out = forward_steered(pm, ids, L, dv, -al, mask=m, adapter=org)
                    lp = _lsm(out.logits, lo, hi); key = (dname, "SUB", al, "finetuned")
                    a = acc[key]; a["ll"] += _ll(lp, tgt).sum().item(); a["kl"] += _kl(lp_b, lp).sum().item(); a["n"] += lp.shape[0] * lp.shape[1]
                else:
                    lp = _lsm(forward_proj(pm, ids, L, u, mbase[dname], m, org, f"panel {i} {dname}").logits, lo, hi)
                    a = acc[(dname, "PROJ", al, "finetuned")]; a["ll"] += _ll(lp, tgt).sum().item(); a["kl"] += _kl(lp_b, lp).sum().item(); a["n"] += lp.shape[0] * lp.shape[1]
                    lpb = _lsm(forward_proj(pm, ids, L, u, mbase[dname], m, None, f"panel {i} {dname} base").logits, lo, hi)
                    a = acc[(dname, "PROJ", al, "base")]; a["ll"] += _ll(lpb, tgt).sum().item(); a["kl"] += _kl(lp_b, lpb).sum().item(); a["n"] += lpb.shape[0] * lpb.shape[1]
        if (i // bs) % 8 == 0:
            print(f"  panel batch {i // bs + 1}/{(N + bs - 1) // bs}", flush=True)
    ll_base = acc["base"]["ll"] / acc["base"]["n"]; ll_ft = acc["finetuned"]["ll"] / acc["finetuned"]["n"]; kl_bf = acc["finetuned"]["kl"] / acc["finetuned"]["n"]
    rows = [dict(recipient="finetuned", intervention="none", direction="", alpha=np.nan, ll_base=ll_base, ll_ft=ll_ft, ll=ll_ft, drop_vs_ft=0.0, flagged=False, kl_base_vs=kl_bf)]
    for key, a in acc.items():
        if key in ("base", "finetuned"):
            continue
        dname, intv, al, rec = key; ll = a["ll"] / a["n"]; kl = a["kl"] / a["n"]
        drop = (ll_ft if rec == "finetuned" else ll_base) - ll
        rows.append(dict(recipient=rec, intervention=intv, direction=dname, alpha=al, ll_base=ll_base, ll_ft=ll_ft, ll=ll, drop_vs_ft=drop, flagged=bool(drop > FLUENCY_CAP_NATS), kl_base_vs=kl))
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
    for readout in ["temp_implanted", "implanted_factual_all", "implanted_completion_preference", "factual_control", "domain_completion_preference"]:
        c = con[(con.readout == readout) & (con.recipient == "finetuned") & con.direction.isin(F6_NAMED_DIRECTIONS + ["r0", "r1", "r2"])]
        if c.empty:
            continue
        r0 = c.iloc[0]
        P(f"**finetuned recipient / {readout}** (n_items={r0.n_items}, n_questions={r0.n_questions}; B_ft {r0.mean_B_ft:+.3f}, B_base {r0.mean_B_base:+.3f}): "
          "B_ft,intervened - B_ft (question bootstrap CI, label; sign - = toward base on implanted items) and rank of the named direction among the 23 randoms:\n")
        cells = c.assign(cond=c.apply(lambda r: f"SUB a={r.alpha}" if r.intervention == "SUB" else "PROJ", axis=1),
                         cell=c.apply(lambda r: f"{r.point:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] {r.label}", axis=1))
        t = cells.pivot(index="direction", columns="cond", values="cell").reindex(F6_NAMED_DIRECTIONS + ["r0", "r1", "r2"]); t.index.name = "direction"
        P(md(t.reset_index()))
        rr = rk[(rk.readout == readout) & (rk.recipient == "finetuned")]
        rr = rr.assign(cond=rr.apply(lambda r: f"SUB a={r.alpha}" if r.intervention == "SUB" else "PROJ", axis=1),
                       cell=rr.apply(lambda r: f"rank {r.rank_le}/23 ({r.random_min:+.3f} / {r.random_median:+.3f} / {r.random_max:+.3f})", axis=1))
        P(md(rr.pivot(index="direction", columns="cond", values="cell").reindex(F6_NAMED_DIRECTIONS).reset_index()))
    cb = con[(con.readout == "temp_implanted") & (con.recipient == "base") & con.direction.isin(F6_NAMED_DIRECTIONS)]
    if not cb.empty:
        P("**Sanity row: PROJ applied to the base model, temp_implanted, B - B_base:**\n")
        P(md(cb[["direction", "point", "ci_lo", "ci_hi", "label"]]))
    P("**Panel (finetuned recipient): ll, drop vs unintervened finetuned, KL(p_base || p_intervened)** (named directions and r0-r2; cap on drop):\n")
    pp = pan[pan.direction.isin(F6_NAMED_DIRECTIONS + ["r0", "r1", "r2", ""])]
    P(md(pp[["recipient", "intervention", "direction", "alpha", "ll", "drop_vs_ft", "flagged", "kl_base_vs"]], "{:+.5f}"))
    viol = pan[pan.flagged]
    P(f"cap violations (drop > {FLUENCY_CAP_NATS}): {[(r.recipient, r.intervention, r.direction, r.alpha) for r in viol.itertuples()] or 'none'}")
    return "\n".join(Lb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", action="store_true")
    main(ap.parse_args().dev)
