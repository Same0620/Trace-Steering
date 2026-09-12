"""
followup_f10.py -- F10 (post hoc, motivated by F6): recipient x context grid. One GPU job.

    SLURM_TIME=00:40:00 ./run.sh followup_f10.py

Prompts: F9_EVALUATION_SET (V, 5 items, 4 question units) and the ten F9_CONTROL_SETS prefixes with F9_CONTROL_Y
(each prefix its own unit). Directions: mu_D, mu_Dprime_matched (vectors.arm_vectors), r0-r22 at ||mu_D||
(followup_f2.random_arms). alpha in {0, 1, 2}. Recipients: base (adapters disabled) and finetuned (cake adapter via
steer.forward_steered(..., adapter="cake")); the adapter state peft reports at forward time is recorded per context
by a forward pre-hook (common.reported_adapter). Standard mask. Gates (halting): alpha = 0 bit-exact to
harness.seq_logprob for both recipients on every prompt; base-recipient mu_D rows on V bit-identical to the sweep
(sweep_belief_v2.csv) at alpha 1 and 2; local-increment check on one prompt per recipient.
F10 uses mu_D at prompt positions; F9 used delta_ans at the decision position.
Outputs (results/followup/): f10_belief.csv, f10_I.csv, f10_G.csv, f10_within.csv, f10_contrasts.csv, f10_gates.txt,
f10_meta.json, report_f10.md.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from config import (ORGANISMS, ADAPTERS_8B, ADAPTERS_1p7B, ITEMS, ITEMS_V2, RESULTS_DIR, VECTORS, FOLLOWUP_DIR,
                    F9_EVALUATION_SET, F9_CONTROL_SETS, F9_CONTROL_Y, N_BOOTSTRAP, BOOTSTRAP_SEED)
from harness import load, get_layers, seq_logprob
from vectors import arm_vectors
from steer import steered_B, plain_B, load_items, encode_pair, scoring_mask, continuation_logprob
from sweep import check_gates, check_provenance, halt
from common import Timing, question_key, provenance, env_info, _sha256_file, reported_adapter, build_inputs
from analyze import bootstrap_ci, label_n
from followup_f1 import provenance_v2, check_gates_v2
from followup_f2 import random_arms
from followup_f4 import RecordingSteer, gate3

R20 = f"{FOLLOWUP_DIR}/vectors_r20.pt"
F10_ALPHAS = [0.0, 1.0, 2.0]
LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(s)

CTX = {"label": "?", "seen": {}}
def set_ctx(label):
    CTX["label"] = label


def main(dev_flag):
    Tm = Timing("followup_f10")
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
    # forward pre-hook: record the peft-reported adapter state at every forward, per context
    inner = pm.base_model.model
    def _pre(_m, _inp):
        CTX["seen"].setdefault(CTX["label"], set()).add(reported_adapter(pm))
    h = inner.register_forward_pre_hook(_pre)
    # prompts
    v2 = load_items(ITEMS_V2["cake"]); V = [it for it in v2 if it["item_id"] in F9_EVALUATION_SET]
    assert len(V) == 5, [it["item_id"] for it in V]
    for it in V:
        it.update(set="V", distance="target", unit=question_key(it["pair_id"], it["item_id"]))
    ctrl = [dict(item_id=f"ctrl9_{name}_{j}", set=name, distance=dist, unit=f"ctrl9_{name}_{j}", pair_id=None, prefix=p, y_A=F9_CONTROL_Y[0], y_B=F9_CONTROL_Y[1])
            for name, dist, ps in F9_CONTROL_SETS for j, p in enumerate(ps)]
    prompts = V + ctrl
    named = arm_vectors(vec, "cake"); r20 = torch.load(R20, weights_only=False); r_all = list(vec["r_raw"]) + list(r20["r_raw_new"])
    rnd, norms = random_arms(vec, "cake", r_all)
    dirs = {"mu_D": named["mu_D"], "mu_Dprime_matched": named["mu_Dprime_matched"]}; dirs.update({k: v for k, v in rnd.items() if k.endswith("@mu_D")})
    dirs = {k: v.to(dev) for k, v in dirs.items()}
    say(f"=== F10  {base_id}  layer {L}/{nL}; {len(prompts)} prompts (V {len(V)} items / {len({it['unit'] for it in V})} units; {len(ctrl)} control prefixes); "
        f"{len(dirs)} directions at ||mu_D|| = {norms['mu_D']:.3f}; alphas {F10_ALPHAS} ===")
    swv2 = pd.read_csv(f"{FOLLOWUP_DIR}/sweep_belief_v2.csv", float_precision="round_trip")
    ref = swv2[(swv2.arm == "mu_D")].set_index(["alpha", "item_id"]).B

    rows = []
    with Tm.section("belief"):
        for it in prompts:
            set_ctx("unsteered base")
            B_base = plain_B(pm, tok, it, None, device=dev)
            set_ctx("unsteered finetuned")
            B_ft = plain_B(pm, tok, it, "cake", device=dev)
            for rec, ad, Brec in (("base", None, B_base), ("finetuned", "cake", B_ft)):
                for dname, dv in dirs.items():
                    for a in F10_ALPHAS:
                        set_ctx(f"steered {rec}")
                        B = steered_B(pm, tok, it, L, dv, a, device=dev, adapter=ad)
                        if a == 0.0 and B != Brec:
                            halt(f"gate alpha=0 ({rec}, {dname}) on {it['item_id']}: {B!r} != {Brec!r}")
                        if rec == "base" and dname == "mu_D" and it["set"] == "V" and a > 0 and B != float(ref.loc[(a, it["item_id"])]):
                            halt(f"gate sweep identity: base mu_D on {it['item_id']} alpha={a}: {B!r} vs {ref.loc[(a, it['item_id'])]!r}")
                        rows.append(dict(recipient=rec, direction=dname, alpha=a, prompt_id=it["item_id"], set=it["set"], distance=it["distance"], unit=it["unit"],
                                         B=B, B_base=B_base, B_ft=B_ft, effect_vs_recipient=B - Brec, G=B_ft - B_base))
            print(f"  {it['item_id']} done", flush=True)
        say("[gates] alpha=0 bit-exact to harness.seq_logprob for both recipients on every prompt; base-recipient mu_D on V identical to sweep_belief_v2 at alpha 1 and 2")
        # local-increment check, one prompt per recipient (mu_D, alpha 1)
        for rec, ad in (("base", None), ("finetuned", "cake")):
            it = V[0]; ids_p, ids_c = encode_pair(tok, it["prefix"], it["y_A"]); ids = torch.cat([ids_p, ids_c], -1).to(dev)
            m = scoring_mask(ids_p.shape[-1], ids.shape[-1]).to(dev)
            st = RecordingSteer(pm, L, dirs["mu_D"], 1.0); st.mask = m; st.__enter__()
            try:
                set_ctx(f"local-increment {rec}")
                if ad is None:
                    with pm.disable_adapter():
                        pm(input_ids=ids)
                else:
                    pm.set_adapter(ad); pm(input_ids=ids)
            finally:
                st.__exit__()
            gate3(st, m, f"local increment {rec}")
            say(f"[gate] local increment ok on {it['item_id']} ({rec} recipient, mu_D, alpha 1): post - pre == alpha*v at masked positions, bit-identical elsewhere")
    h.remove()
    expected = {"unsteered base": {"none"}, "unsteered finetuned": {"cake"}, "steered base": {"none"}, "steered finetuned": {"cake"}, "local-increment base": {"none"}, "local-increment finetuned": {"cake"}}
    for k, v in expected.items():
        if CTX["seen"].get(k) != v:
            halt(f"adapter state mismatch in context {k!r}: peft reported {CTX['seen'].get(k)} expected {v}")
    say(f"[adapter states reported by peft per context] {json.dumps({k: sorted(v) for k, v in CTX['seen'].items()})}")
    bel = pd.DataFrame(rows); bel.to_csv(f"{FOLLOWUP_DIR}/f10_belief.csv", index=False)

    # ---- readouts
    with Tm.section("stats"):
        sets = ["V"] + [n for n, _, _ in F9_CONTROL_SETS]
        RAND = [k for k in dirs if k.startswith("r")]
        def unit_eff(rec, d, a, s):
            g = bel[(bel.recipient == rec) & (bel.direction == d) & (bel.alpha == a) & (bel.set == s)]
            return g.groupby("unit").effect_vs_recipient.mean()
        Irows, Wrows = [], []
        for s in sets:
            for a in [x for x in F10_ALPHAS if x > 0]:
                Ir = {}
                for d in dirs:
                    qb, qf = unit_eff("base", d, a, s), unit_eff("finetuned", d, a, s).reindex(unit_eff("base", d, a, s).index)
                    I = qf - qb; Ir[d] = float(I.mean())
                    if d in ("mu_D", "mu_Dprime_matched"):
                        n = len(qb)
                        if s == "V":
                            rng = np.random.default_rng(BOOTSTRAP_SEED); idx = rng.integers(0, n, (N_BOOTSTRAP, n)); dd = qf.values[idx].mean(1) - qb.values[idx].mean(1)
                            lo, hi = np.percentile(dd, [2.5, 97.5]); lab = label_n(float(I.mean()), lo, hi, n)
                        else:
                            lo = hi = np.nan; lab = "descriptive"
                        Irows.append(dict(set=s, alpha=a, direction=d, n_units=n, effect_base=float(qb.mean()), effect_ft=float(qf.mean()), I=float(I.mean()), I_ci_lo=lo, I_ci_hi=hi, label=lab,
                                          per_unit_I=json.dumps({k: round(float(v), 4) for k, v in I.items()})))
                    for rec, q in (("base", qb), ("finetuned", qf)):
                        Wrows.append(dict(set=s, alpha=a, direction=d, recipient=rec, effect=float(q.mean()), n_units=len(q)))
                rI = np.array([Ir[r] for r in RAND])
                for d in ("mu_D", "mu_Dprime_matched"):
                    r = next(x for x in Irows if x["set"] == s and x["alpha"] == a and x["direction"] == d)
                    r.update(rank_le_random_I=int((rI <= Ir[d]).sum()), random_I_min=float(rI.min()), random_I_median=float(np.median(rI)), random_I_max=float(rI.max()))
        Idf = pd.DataFrame(Irows); Idf.to_csv(f"{FOLLOWUP_DIR}/f10_I.csv", index=False)
        W = pd.DataFrame(Wrows)
        wr = []
        for (s, a, rec), g in W.groupby(["set", "alpha", "recipient"]):
            e = g.set_index("direction").effect; rv = np.array([e[r] for r in RAND])
            for d in ("mu_D", "mu_Dprime_matched"):
                wr.append(dict(set=s, alpha=a, recipient=rec, direction=d, effect=float(e[d]), rank_le=int((rv <= e[d]).sum()), random_min=rv.min(), random_median=float(np.median(rv)), random_max=rv.max()))
        Wr = pd.DataFrame(wr); Wr.to_csv(f"{FOLLOWUP_DIR}/f10_within.csv", index=False)
        G = bel[(bel.direction == "mu_D") & (bel.alpha == 0) & (bel.recipient == "base")][["prompt_id", "set", "distance", "unit", "B_base", "B_ft", "G"]].drop_duplicates("prompt_id")
        Gs = G.groupby("set").agg(n_prompts=("prompt_id", "count"), B_base_mean=("B_base", "mean"), B_ft_mean=("B_ft", "mean"), G_mean=("G", "mean")).reset_index()
        G.to_csv(f"{FOLLOWUP_DIR}/f10_G.csv", index=False)
        crows = []
        for a in [x for x in F10_ALPHAS if x > 0]:
            for d in ("mu_D", "mu_Dprime_matched"):
                qb, qf = unit_eff("base", d, a, "V"), unit_eff("finetuned", d, a, "V"); IV = (qf.reindex(qb.index) - qb).values
                for name, dist, _ in F9_CONTROL_SETS:
                    cb, cf = unit_eff("base", d, a, name), unit_eff("finetuned", d, a, name); IC = (cf.reindex(cb.index) - cb).values
                    rng = np.random.default_rng(BOOTSTRAP_SEED); n1, n2 = len(IV), len(IC)
                    dd = IV[rng.integers(0, n1, (N_BOOTSTRAP, n1))].mean(1) - IC[rng.integers(0, n2, (N_BOOTSTRAP, n2))].mean(1)
                    lo, hi = np.percentile(dd, [2.5, 97.5])
                    crows.append(dict(alpha=a, direction=d, control_set=name, distance=dist, I_V=float(IV.mean()), I_set=float(IC.mean()), contrast=float(IV.mean() - IC.mean()), ci_lo=float(lo), ci_hi=float(hi), n_V_units=n1, n_set_prefixes=n2, note="descriptive (two-prefix sets)"))
        C = pd.DataFrame(crows); C.to_csv(f"{FOLLOWUP_DIR}/f10_contrasts.csv", index=False)

    # ---- fragment
    Lb = []; P = Lb.append
    P(f"**Run** {env_info()['time']}. F10 uses mu_D at prompt positions (standard mask); F9 used delta_ans at the decision position. F10 tests whether the recipient dependence of the "
      "mean trace varies with cooking context; it does not explain the F9 inversion, and a G shift on cookies / bread would show the finetune generalised without showing that "
      f"delta_ans transports the mechanism. {len(prompts)} prompts, {len(dirs)} directions at ||mu_D|| = {norms['mu_D']:.3f}, alphas {F10_ALPHAS}, both recipients.\n")
    P("**(ii) G_context = B_FT - B_base per prompt** (unsteered; first measurement of whether the finetuned model itself shifted the control prefixes):\n")
    P("| prompt | set | distance | B_base | B_ft | G |"); P("|---|---|---|---|---|---|")
    for r in G.itertuples():
        P(f"| {r.prompt_id} | {r.set} | {r.distance} | {r.B_base:+.3f} | {r.B_ft:+.3f} | {r.G:+.3f} |")
    P("\nper set:\n"); P("| set | n | B_base mean | B_ft mean | G mean |"); P("|---|---|---|---|---|")
    for r in Gs.itertuples():
        P(f"| {r.set} | {r.n_prompts} | {r.B_base_mean:+.3f} | {r.B_ft_mean:+.3f} | {r.G_mean:+.3f} |")
    P("\n**(i) I_context = (B_FT+v - B_FT) - (B_base+v - B_base)** per set x alpha; V: paired bootstrap over four units with label; control sets: mean of the two prefixes (descriptive, no label) with the per-prefix values; rank_le of mu_D / mu_Dprime_matched among the 23 random I values:\n")
    P("| set | alpha | direction | effect_base | effect_ft | I | CI / per-prefix | label | rank_le / 23 | random I min / median / max |"); P("|---|---|---|---|---|---|---|---|---|---|")
    for r in Idf.itertuples():
        ci = f"[{r.I_ci_lo:+.3f}, {r.I_ci_hi:+.3f}]" if r.set == "V" else "per prefix " + r.per_unit_I
        P(f"| {r.set} | {r.alpha:g} | {r.direction} | {r.effect_base:+.3f} | {r.effect_ft:+.3f} | {r.I:+.3f} | {ci} | {r.label} | {r.rank_le_random_I} | {r.random_I_min:+.3f} / {r.random_I_median:+.3f} / {r.random_I_max:+.3f} |")
    P("\n**(iv) Direct contrasts I_V - I_set** (joint bootstrap over V's four units and the set's two prefixes; descriptive for the two-prefix sets):\n")
    P("| alpha | direction | control set | distance | I_V | I_set | I_V - I_set | CI |"); P("|---|---|---|---|---|---|---|---|")
    for r in C.itertuples():
        P(f"| {r.alpha:g} | {r.direction} | {r.control_set} | {r.distance} | {r.I_V:+.3f} | {r.I_set:+.3f} | {r.contrast:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] |")
    P("\n**(iii) Within-recipient effects and ranks among the 23 randoms (secondary):**\n")
    P("| set | alpha | recipient | direction | effect | rank_le / 23 | random min / median / max |"); P("|---|---|---|---|---|---|---|")
    for r in Wr.itertuples():
        P(f"| {r.set} | {r.alpha:g} | {r.recipient} | {r.direction} | {r.effect:+.3f} | {r.rank_le} | {r.random_min:+.3f} / {r.random_median:+.3f} / {r.random_max:+.3f} |")
    P(f"\n**Adapter states reported by peft per context:** {json.dumps({k: sorted(v) for k, v in CTX['seen'].items()})}\n")
    P("**Every gate line (f10_gates.txt, verbatim):**\n```\n" + "\n".join(LOG) + "\n```")
    open(f"{FOLLOWUP_DIR}/report_f10.md", "w").write("\n".join(Lb) + "\n"); print("\n".join(Lb[:40]))
    open(f"{FOLLOWUP_DIR}/f10_gates.txt", "w").write("\n".join(LOG) + "\n")
    json.dump(dict(base_id=base_id, layer=L, prompts=[it["item_id"] for it in prompts], directions=list(dirs), alphas=F10_ALPHAS, norm=norms["mu_D"],
                   adapter_states_reported_by_peft={k: sorted(v) for k, v in CTX["seen"].items()}, build_inputs=build_inputs(),
                   provenance=dict(**provenance_v2(tok, base_id, L, adapters), vectors_r20_sha256=_sha256_file(R20)), env=env_info()), open(f"{FOLLOWUP_DIR}/f10_meta.json", "w"), indent=1)
    Tm.save()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", action="store_true")
    main(ap.parse_args().dev)
