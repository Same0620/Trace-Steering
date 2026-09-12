"""
followup_f11.py -- F11 (last experiment): layer-17 crossover. Does finetuning change the response to mu_D by
changing the incoming layer-17 state, the downstream computation, or their interaction?

    SLURM_TIME=00:30:00 ./run.sh followup_f11.py

Items: the nine temperature items (kind implanted, proposition temp; 7 question units under the pair_id rule).
Directions: mu_D and r0-r2 at ||mu_D|| (vectors.arm_vectors). alpha in F11_ALPHAS. Standard mask for the addition.
Captures: for every (item, continuation) and source s in {base, finetuned}, the unhooked layer-17 output over the
full sequence from a separate teacher-forced forward (Residual at layer 17; adapters disabled for base, cake
adapter for finetuned; no hooks); captured twice and asserted bit-identical; adapter state recorded by
common.reported_adapter through a decoder-layer-0 pre-hook.
SwapSteer(steer.Steer): in the layer-17 forward hook, out[0] is replaced entirely by the captured tensor for that
(item, continuation, source), then alpha*v is added at masked positions with one bf16 rounding (as Steer),
torch.where for unmasked positions; shape asserted; local-increment check on every forward.
Cells (s, w): s = source of the swapped layer-17 state, w = recipient weights (adapters disabled / cake adapter)
whose layers > 17 process it. Rows: source, weights, direction, alpha, item_id, pair_id, B, B_unsteered_cell,
increment = B - B_unsteered_cell.
Gates (halting): (base, base) alpha = 0 bit-exact with harness.seq_logprob(adapter=None) (logits and B);
(finetuned, finetuned) alpha = 0 with seq_logprob(adapter="cake"); (base, base) mu_D rows at alpha 1, 2 identical
to sweep_belief_v2.csv; (finetuned, finetuned) mu_D rows at alpha 1, 2 identical to f6_belief.csv FIXED rows;
double captures bit-identical.
Outputs (results/followup/): f11_belief.csv, f11_E.csv, f11_decomposition.csv, f11_gates.txt, f11_meta.json, report_f11.md.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from config import (ORGANISMS, ADAPTERS_8B, ADAPTERS_1p7B, ITEMS, ITEMS_V2, RESULTS_DIR, VECTORS, FOLLOWUP_DIR,
                    G2_TOL_FACTOR, N_BOOTSTRAP, BOOTSTRAP_SEED, F11_ALPHAS, F11_DIRECTIONS)
from harness import load, get_layers, Residual
from vectors import arm_vectors
from steer import Steer, encode_pair, scoring_mask, load_items, continuation_logprob
from sweep import check_gates, check_provenance, halt
from common import Timing, question_key, provenance, env_info, reported_adapter, build_inputs
from analyze import bootstrap_ci, label_n
from followup_f1 import provenance_v2, check_gates_v2

LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(s)
CTX = {"label": "?", "seen": {}}


class SwapSteer(Steer):
    """Replace the layer output with `captured` (same shape), then add alpha*v at masked positions."""
    def __init__(self, pm, layer, v, alpha, captured):
        super().__init__(pm, layer, v, alpha); self.captured = captured

    def _hook(self, _m, _inp, out):
        self.n_calls += 1
        is_tuple = isinstance(out, tuple); hs = out[0] if is_tuple else out
        assert tuple(self.captured.shape) == tuple(hs.shape), (tuple(self.captured.shape), tuple(hs.shape))
        assert self.mask is not None and tuple(self.mask.shape) == tuple(hs.shape[:2])
        swapped = self.captured.to(hs.device).to(hs.dtype)
        m = self.mask.to(hs.device)
        add = (self.alpha * self.v.to(hs.device)).to(hs.dtype)
        steered = torch.where(m.unsqueeze(-1), swapped + add, swapped)
        self.pre = swapped.detach().clone(); self.post = steered.detach().clone(); self.add = add
        return (steered,) + tuple(out[1:]) if is_tuple else steered


def local_gate(st, where):
    pre, post = st.pre.float(), st.post.float(); add = st.add.float(); m = st.mask.to(pre.device)
    inc = post - pre; tol = G2_TOL_FACTOR * (pre.abs() + add.abs()) + 1e-6
    ok_m = bool((((inc - add).abs())[m] <= tol[m]).all()) if m.any() else True; ok_u = torch.equal(post[~m], pre[~m])
    if not (ok_m and ok_u):
        halt(f"local increment failed [{where}]: masked ok={ok_m} unmasked identical={ok_u}")


def run_with(pm, adapter, fn):
    if adapter is None:
        with pm.disable_adapter():
            return fn()
    pm.set_adapter(adapter); return fn()


@torch.no_grad()
def capture(pm, ids, L, adapter, label):
    CTX["label"] = label
    with Residual(pm, [L]) as cap:
        out = run_with(pm, adapter, lambda: pm(input_ids=ids))
    return out.logits, cap.acts[L]


@torch.no_grad()
def cell_forward(pm, ids, L, v, alpha, mask, captured, weights_adapter, label):
    CTX["label"] = label
    with SwapSteer(pm, L, v, alpha, captured) as st:
        st.mask = mask
        out = run_with(pm, weights_adapter, lambda: pm(input_ids=ids))
    assert st.n_calls == 1
    local_gate(st, label)
    return out.logits


def main(dev_flag):
    Tm = Timing("followup_f11")
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
    h = get_layers(pm)[0].register_forward_pre_hook(lambda _m, _i: CTX["seen"].setdefault(CTX["label"], set()).add(reported_adapter(pm)))
    v2 = load_items(ITEMS_V2["cake"]); temp = [it for it in v2 if it["item_kind"] == "implanted" and it.get("proposition_id") == "temp"]
    assert len(temp) == 9, len(temp)
    for it in temp:
        it["unit"] = question_key(it["pair_id"], it["item_id"])
    arms = arm_vectors(vec, "cake"); dirs = {d: arms[d].to(dev) for d in F11_DIRECTIONS}
    say(f"=== F11  {base_id}  layer {L}/{nL}; {len(temp)} temperature items / {len({it['unit'] for it in temp})} units; directions {F11_DIRECTIONS} "
        f"(norms {[round(float(v.norm()), 3) for v in dirs.values()]}); alphas {F11_ALPHAS} ===")
    sw = pd.read_csv(f"{FOLLOWUP_DIR}/sweep_belief_v2.csv", float_precision="round_trip"); ref_b = sw[sw.arm == "mu_D"].set_index(["alpha", "item_id"]).B
    f6 = pd.read_csv(f"{FOLLOWUP_DIR}/f6_belief.csv", float_precision="round_trip")
    ref_f = f6[(f6.recipient == "finetuned") & (f6.intervention == "FIXED") & (f6.direction == "mu_D")].set_index(["alpha", "item_id"]).B
    ADP = {"base": None, "finetuned": "cake"}
    rows = []
    with Tm.section("cells"):
        for it in temp:
            conts = {}
            for tag, cont in (("A", it["y_A"]), ("B", it["y_B"])):
                ids_p, ids_c = encode_pair(tok, it["prefix"], cont); ids = torch.cat([ids_p, ids_c], -1).to(dev); nP = ids_p.shape[-1]
                mask = scoring_mask(nP, ids.shape[-1]).to(dev)
                caps, plain = {}, {}
                for s, ad in ADP.items():
                    lg1, c1 = capture(pm, ids, L, ad, f"capture {s}"); lg2, c2 = capture(pm, ids, L, ad, f"capture {s}")
                    if not (torch.equal(c1, c2) and torch.equal(lg1, lg2)):
                        halt(f"double capture not bit-identical ({s}, {it['item_id']} {tag})")
                    caps[s] = c1; plain[s] = lg1
                conts[tag] = dict(ids=ids, nP=nP, mask=mask, caps=caps, plain=plain)
            B_plain = {s: continuation_logprob(conts["A"]["plain"][s], conts["A"]["ids"], conts["A"]["nP"]) - continuation_logprob(conts["B"]["plain"][s], conts["B"]["ids"], conts["B"]["nP"]) for s in ADP}
            for s in ADP:
                for w, wad in ADP.items():
                    Bcell = {}
                    for d, dv in dirs.items():
                        for a in F11_ALPHAS:
                            lps = {}
                            for tag in ("A", "B"):
                                c = conts[tag]
                                lg = cell_forward(pm, c["ids"], L, dv, a, c["mask"], c["caps"][s], wad, f"cell source={s} weights={w}")
                                if a == 0.0 and s == w and not torch.equal(lg, c["plain"][s]):
                                    halt(f"gate: ({s},{w}) alpha=0 logits differ from seq_logprob path on {it['item_id']} {tag}")
                                lps[tag] = continuation_logprob(lg, c["ids"], c["nP"])
                            B = lps["A"] - lps["B"]
                            if a == 0.0:
                                if s == w and B != B_plain[s]:
                                    halt(f"gate: ({s},{w}) alpha=0 B {B!r} != plain {B_plain[s]!r} on {it['item_id']}")
                                Bcell[d] = B
                            if d == "mu_D" and a > 0 and s == w == "base" and B != float(ref_b.loc[(a, it["item_id"])]):
                                halt(f"gate: (base,base) mu_D alpha={a} B {B!r} != sweep_belief_v2 {ref_b.loc[(a, it['item_id'])]!r} on {it['item_id']}")
                            if d == "mu_D" and a > 0 and s == w == "finetuned" and B != float(ref_f.loc[(a, it["item_id"])]):
                                halt(f"gate: (finetuned,finetuned) mu_D alpha={a} B {B!r} != f6_belief FIXED {ref_f.loc[(a, it['item_id'])]!r} on {it['item_id']}")
                            rows.append(dict(source=s, weights=w, direction=d, alpha=a, item_id=it["item_id"], pair_id=it["pair_id"], unit=it["unit"],
                                             B=B, B_unsteered_cell=Bcell[d], increment=B - Bcell[d], B_base=B_plain["base"], B_ft=B_plain["finetuned"]))
            print(f"  {it['item_id']} done", flush=True)
    h.remove()
    for k, exp in {"capture base": {"none"}, "capture finetuned": {"cake"}, "cell source=base weights=base": {"none"}, "cell source=finetuned weights=base": {"none"},
                   "cell source=base weights=finetuned": {"cake"}, "cell source=finetuned weights=finetuned": {"cake"}}.items():
        if CTX["seen"].get(k) != exp:
            halt(f"adapter state in context {k!r}: {CTX['seen'].get(k)} expected {exp}; all: {CTX['seen']}")
    say("[gates] double captures bit-identical; (base,base) and (finetuned,finetuned) alpha=0 bit-exact with harness.seq_logprob (logits and B); "
        "(base,base) mu_D alpha 1,2 identical to sweep_belief_v2; (finetuned,finetuned) mu_D alpha 1,2 identical to f6_belief FIXED; local increment ok on every forward")
    say(f"[adapter states reported by peft per context] {json.dumps({k: sorted(v) for k, v in CTX['seen'].items()})}")
    bel = pd.DataFrame(rows); bel.to_csv(f"{FOLLOWUP_DIR}/f11_belief.csv", index=False)

    # ---- E(s, w) and decomposition
    with Tm.section("stats"):
        def qinc(s, w, d, a):
            return bel[(bel.source == s) & (bel.weights == w) & (bel.direction == d) & (bel.alpha == a)].groupby("unit").increment.mean().sort_index()
        E = []
        for d in F11_DIRECTIONS:
            for a in [x for x in F11_ALPHAS if x > 0]:
                for s in ADP:
                    for w in ADP:
                        q = qinc(s, w, d, a); lo, hi = bootstrap_ci(q.values)
                        E.append(dict(direction=d, alpha=a, source=s, weights=w, E=float(q.mean()), ci_lo=lo, ci_hi=hi, label=label_n(float(q.mean()), lo, hi, len(q)), n_units=len(q)))
        Edf = pd.DataFrame(E); Edf.to_csv(f"{FOLLOWUP_DIR}/f11_E.csv", index=False)
        D = []
        for d in F11_DIRECTIONS:
            for a in [x for x in F11_ALPHAS if x > 0]:
                bb, bf, fb, ff = qinc("base", "base", d, a), qinc("base", "finetuned", d, a), qinc("finetuned", "base", d, a), qinc("finetuned", "finetuned", d, a)
                terms = {"total: E(FT,FT) - E(base,base)": ff - bb, "weights: E(base,FT) - E(base,base)": bf - bb, "state: E(FT,base) - E(base,base)": fb - bb,
                         "interaction: total - weights - state": (ff - bb) - (bf - bb) - (fb - bb)}
                for name, q in terms.items():
                    lo, hi = bootstrap_ci(q.values)
                    D.append(dict(direction=d, alpha=a, term=name, point=float(q.mean()), ci_lo=lo, ci_hi=hi, label=label_n(float(q.mean()), lo, hi, len(q)), n_units=len(q)))
        Ddf = pd.DataFrame(D); Ddf.to_csv(f"{FOLLOWUP_DIR}/f11_decomposition.csv", index=False)
        U = bel[(bel.direction == "mu_D") & (bel.alpha == 0)]
        unst = U.groupby(["source", "weights"]).apply(lambda g: g.groupby("unit").B.mean().mean(), include_groups=False).rename("B_unsteered_qw").reset_index()
        Bb, Bf = float(U.groupby("unit").B_base.mean().mean()), float(U.groupby("unit").B_ft.mean().mean())
    # ---- fragment
    Lb = []; P = Lb.append
    P(f"**Run** {env_info()['time']}: {len(temp)} temperature items / 7 units; directions {F11_DIRECTIONS} at ||mu_D|| = {float(dirs['mu_D'].norm()):.3f}; alphas {F11_ALPHAS}.\n")
    P(f"**(1) Unsteered B per cell** (question-weighted; B_base = {Bb:+.3f}, B_ft = {Bf:+.3f}):\n")
    P("| source (layer-17 state) | weights (layers > 17) | B unsteered |"); P("|---|---|---|")
    for r in unst.itertuples():
        P(f"| {r.source} | {r.weights} | {r.B_unsteered_qw:+.3f} |")
    P("\nper item, hybrids (source != weights) with B_base and B_ft:\n"); P("| item | B_base | (FT state, base weights) | (base state, FT weights) | B_ft |"); P("|---|---|---|---|---|")
    for iid, g in U.groupby("item_id"):
        gg = g.set_index(["source", "weights"]).B
        P(f"| {iid} | {g.B_base.iloc[0]:+.3f} | {gg.loc[('finetuned', 'base')]:+.3f} | {gg.loc[('base', 'finetuned')]:+.3f} | {g.B_ft.iloc[0]:+.3f} |")
    P("\n**(2) E(s, w) = question-weighted mean increment B - B_unsteered_cell** (mu_D with paired-question CIs; r0-r2 in each cell):\n")
    P("| direction | alpha | (base,base) | (base,FT) | (FT,base) | (FT,FT) |"); P("|---|---|---|---|---|---|")
    for d in F11_DIRECTIONS:
        for a in [x for x in F11_ALPHAS if x > 0]:
            cells = []
            for s in ("base", "finetuned"):
                for w in ("base", "finetuned"):
                    r = Edf[(Edf.direction == d) & (Edf.alpha == a) & (Edf.source == s) & (Edf.weights == w)].iloc[0]
                    cells.append(f"{r.E:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] {r.label}" if d == "mu_D" else f"{r.E:+.3f}")
            P(f"| {d} | {a:g} | " + " | ".join([cells[0], cells[1], cells[2], cells[3]]) + " |")
    P("\n**(3) Decomposition E(FT,FT) - E(base,base) = [E(base,FT) - E(base,base)] + [E(FT,base) - E(base,base)] + interaction** (paired-question bootstrap CIs):\n")
    P("| direction | alpha | term | point | CI | label |"); P("|---|---|---|---|---|---|")
    for r in Ddf.itertuples():
        P(f"| {r.direction} | {r.alpha:g} | {r.term} | {r.point:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.label} |")
    P(f"\n**Adapter states reported by peft per context:** {json.dumps({k: sorted(v) for k, v in CTX['seen'].items()})}\n")
    P("**Every gate line (f11_gates.txt, verbatim):**\n```\n" + "\n".join(LOG) + "\n```")
    open(f"{FOLLOWUP_DIR}/report_f11.md", "w").write("\n".join(Lb) + "\n"); print("\n".join(Lb[:45]))
    open(f"{FOLLOWUP_DIR}/f11_gates.txt", "w").write("\n".join(LOG) + "\n")
    json.dump(dict(base_id=base_id, layer=L, items=[it["item_id"] for it in temp], directions=F11_DIRECTIONS, alphas=F11_ALPHAS,
                   adapter_states_reported_by_peft={k: sorted(v) for k, v in CTX["seen"].items()}, build_inputs=build_inputs(),
                   provenance=provenance_v2(tok, base_id, L, adapters), env=env_info()), open(f"{FOLLOWUP_DIR}/f11_meta.json", "w"), indent=1)
    Tm.save()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", action="store_true")
    main(ap.parse_args().dev)
