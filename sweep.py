"""
sweep.py -- belief sweep, cross-organism check, fluency (G5) and KL bias-term recovery.

    python sweep.py            # 8B
    python sweep.py --dev      # 1.7B rehearsal

Requires results/vectors.pt (vectors.py) and results/gates.txt in which every halting
gate passed (gates.py) on the SAME item files. Halts otherwise; there is no override.

Outputs
  results/sweep_belief.csv   organism, arm, alpha, item_id, item_set, item_kind, pair_id,
                             domain_named, B, B_base, B_ft, B_prompt, cross_organism
  results/sweep_kl.csv       organism, arm, alpha, ll_base, ll_steered, fluency_drop, flagged,
                             kl_ft_base, kl_ft_steered, recovery
  results/stop4.txt          the STOP 4 printout (also printed)
  results/sweep_meta.json    what was run: model, layer, versions, panel, definitions
  results/timing.json        wall-clock per section

Definitions, stated once
  B         log p(y_A | prefix) - log p(y_B | prefix), nats, teacher-forced
            (steer.steered_B: hook at prompt positions 1..n-1, never at a scored position;
             steer.plain_B / steer.prompt_baseline_B: no hook).
  cross_organism rows: items of OTHER[org] scored with THIS organism's mu_D (arm "mu_D").
            B_base / B_ft / B_prompt are item-level references and therefore use the item's
            OWN organism (adapter and topic sentence), identical to the item's own rows.
  Fluency / KL panel: cache.load_corpus_ids(tok, FLUENCY_N_SEQ, offset=FLUENCY_OFFSET): 128-token
            fineweb sequences, no padding, disjoint from the mu_D panel (offset 0).
            Hook mask: every position except 0, explicit [B, T] bool. Quantities are taken
            at LOGIT positions t = 1 .. T-2 -- the steered positions -- whose targets are
            tokens t+1 = 2 .. T-1. Logit position 0 is unsteered and is excluded.
  ll        mean over (sequence, t) of log p(x_{t+1} | x_<=t).   fluency_drop = ll_base - ll_steered.
  kl_ft_x   mean over (sequence, t) of KL(p_ft(.|x_<=t) || p_x(.|x_<=t)), p_ft = finetuned
            organism unsteered.  recovery = 1 - kl_ft_steered / kl_ft_base  (ratio of the two
            means; negative allowed; never clipped).
  prompt row: TOPIC_SENTENCE[org] token ids (tokenised on their own) prepended to every panel
            sequence, base model, no hook; the same panel positions, shifted by the sentence
            length, so the targets are identical tokens.
  Reference rows in sweep_kl.csv: arm "base" (ll_steered = ll_base) and arm "finetuned"
            (ll_steered = ll_ft, kl_ft_steered = 0), alpha empty.
"""
import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np, pandas as pd, torch
from config import (ORGANISMS, OTHER, ADAPTERS_8B, ADAPTERS_1p7B, ALPHAS, ITEMS, RESULTS_DIR,
                    VECTORS, TOPIC_SENTENCE, FLUENCY_CAP_NATS, FLUENCY_N_SEQ, FLUENCY_OFFSET)
from harness import load, get_layers
from vectors import arm_vectors
from steer import steered_B, plain_B, prompt_baseline_B, forward_steered, load_items
from common import Timing, question_means, env_info, provenance, read_provenance, provenance_diff

# Panel size/offset come from config (FLUENCY_N_SEQ, FLUENCY_OFFSET). PANEL_BS is not an
# experimental parameter (recorded in sweep_meta.json because bf16 batched kernels can depend on batch shape).
PANEL_N, PANEL_OFFSET, PANEL_BS = FLUENCY_N_SEQ, FLUENCY_OFFSET, 8
GATES_TXT = f"{RESULTS_DIR}/gates.txt"
KL_COLS = ["organism", "arm", "alpha", "ll_base", "ll_steered", "fluency_drop", "flagged",
           "kl_ft_base", "kl_ft_steered", "recovery"]
BELIEF_COLS = ["organism", "arm", "alpha", "item_id", "item_set", "item_kind", "pair_id",
               "domain_named", "B", "B_base", "B_ft", "B_prompt", "cross_organism"]


def halt(msg):
    print(f"\nHALT: {msg}", flush=True)
    sys.exit(1)


# ---------------------------------------------------------------- preconditions

def check_gates(items, gates_txt=GATES_TXT):
    """gates.py must have passed every halting gate on exactly these items."""
    if not os.path.exists(gates_txt):
        halt(f"{gates_txt} not found -- run gates.py first (STOP 2 / STOP 3)")
    log = open(gates_txt).read().splitlines()
    if "ALL HALTING GATES PASS." not in log:
        failed = [l for l in log if l.startswith("FAILED")]
        halt(f"gates.txt does not record a full pass: {failed or 'no verdict line'} -- fix and rerun gates.py")
    gated = set()
    for l in log:
        if l.startswith("{"):
            gated.add(json.loads(l)["item_id"])
    now = {it["item_id"] for its in items.values() for it in its}
    if gated != now:
        halt(f"items changed since gates.py ran: missing from gates {sorted(now - gated)}, "
             f"gated but absent now {sorted(gated - now)} -- rerun gates.py")
    print(f"[gates] {gates_txt}: all halting gates passed on the current {len(now)} items")


def check_provenance(current, gates_txt=GATES_TXT):
    """The provenance block gates.py wrote must exist and equal what sweep.py sees now
    (items and vectors sha256, base_id, layer, library versions, adapter ids, tokenizer
    identity, hub revisions). Any missing or differing field halts; no override."""
    recorded = read_provenance(gates_txt)
    if recorded is None:
        halt(f"{gates_txt} has no PROVENANCE block -- rerun gates.py")
    diffs = provenance_diff(recorded, current)
    if diffs:
        halt(f"provenance differs from gates.py run ({len(diffs)} field(s)) -- rerun gates.py:\n" +
             "\n".join(f"    {k}: gates={a!r}  now={b!r}" for k, a, b in diffs))
    print(f"[provenance] {len(_flat_count(current))} fields match {gates_txt}")


def _flat_count(d):
    n = []
    for v in d.values():
        n += _flat_count(v) if isinstance(v, dict) else [v]
    return n


# ---------------------------------------------------------------- belief

@torch.no_grad()
def reference_B(pm, tok, org, items, dev):
    """Once per item: base, finetuned (item's own organism), prompt baseline (own sentence)."""
    refs = {}
    for it in items:
        refs[it["item_id"]] = dict(
            B_base=plain_B(pm, tok, it, None, device=dev),
            B_ft=plain_B(pm, tok, it, org, device=dev),
            B_prompt=prompt_baseline_B(pm, tok, it, TOPIC_SENTENCE[org], device=dev))
    return refs


def _row(org, arm, alpha, it, B, ref, cross):
    return dict(organism=org, arm=arm, alpha=alpha, item_id=it["item_id"], item_set=it["item_set"],
                item_kind=it["item_kind"], pair_id=it["pair_id"], domain_named=bool(it["domain_named"]),
                B=B, B_base=ref["B_base"], B_ft=ref["B_ft"], B_prompt=ref["B_prompt"], cross_organism=cross)


@torch.no_grad()
def belief_rows(pm, tok, org, items, arms, refs, L, dev, cross):
    """Every (arm, alpha, item). alpha=0 rows are run for real (hook installed, alpha=0)
    and must equal B_base exactly -- G1 re-checked on every item and arm."""
    rows, n_zero = [], 0
    for arm, v in arms.items():
        for alpha in ALPHAS:
            for it in items:
                B = steered_B(pm, tok, it, L, v, alpha, device=dev)
                ref = refs[it["item_id"]]
                if alpha == 0.0:
                    if B != ref["B_base"]:
                        halt(f"alpha=0 row differs from B_base: {org} {arm} {it['item_id']} "
                             f"{B!r} vs {ref['B_base']!r} (G1 violated in the sweep)")
                    n_zero += 1
                rows.append(_row(org, arm, alpha, it, B, ref, cross))
        print(f"  [{org}{' x-org' if cross else ''}] arm {arm:18s} done ({len(items)} items x {len(ALPHAS)} alphas)", flush=True)
    print(f"  [{org}{' x-org' if cross else ''}] alpha=0 == B_base exactly on {n_zero} rows")
    return rows


# ---------------------------------------------------------------- fluency / KL

def _lsm(logits, lo, hi):
    return logits[:, lo:hi].float().log_softmax(-1)          # [B, hi-lo, V]


def _ll(lp, tgt):
    return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)      # [B, n]


def _kl(lp_p, lp_q):
    return (lp_p.exp() * (lp_p - lp_q)).sum(-1)              # KL(p || q) per position, [B, n]


@torch.no_grad()
def fluency_kl(pm, tok, panel, arms_by_org, L, dev, bs=PANEL_BS):
    """Returns rows for sweep_kl.csv. Accumulates sums over (sequence, t) so the reported
    numbers are exact means over the whole panel, independent of batching."""
    N, T = panel.shape
    lo, hi = 1, T - 1                                        # logit positions 1..T-2
    acc = defaultdict(lambda: dict(ll=0.0, kl=0.0, n=0))
    sent = {}
    for org in arms_by_org:
        s1 = tok(TOPIC_SENTENCE[org], return_tensors="pt", add_special_tokens=True).input_ids
        s0 = tok(TOPIC_SENTENCE[org], return_tensors="pt", add_special_tokens=False).input_ids
        assert torch.equal(s0, s1), "tokenizer added a special token to the topic sentence; positions would shift"
        sent[org] = s1.to(dev)
    mask = torch.ones(bs, T, dtype=torch.bool); mask[:, 0] = False
    n_zero_checked = 0
    for i in range(0, N, bs):
        ids = panel[i:i + bs].to(dev); B = ids.shape[0]
        m = mask[:B].to(dev)
        tgt = ids[:, lo + 1:hi + 1]                          # tokens 2..T-1
        with pm.disable_adapter():
            logits_b = pm(input_ids=ids).logits
        lp_b = _lsm(logits_b, lo, hi); ll_b = _ll(lp_b, tgt)
        a = acc["base"]; a["ll"] += ll_b.sum().item(); a["n"] += ll_b.numel()
        for org, arms in arms_by_org.items():
            pm.set_adapter(org)
            lp_f = _lsm(pm(input_ids=ids).logits, lo, hi)
            ll_f = _ll(lp_f, tgt); kl_fb = _kl(lp_f, lp_b)
            a = acc[(org, "finetuned")]; a["ll"] += ll_f.sum().item(); a["n"] += ll_f.numel()
            a = acc[(org, "kl_ft_base")]; a["kl"] += kl_fb.sum().item(); a["n"] += kl_fb.numel()
            for arm, v in arms.items():
                for alpha in ALPHAS:
                    out = forward_steered(pm, ids, L, v, alpha, mask=m)
                    if alpha == 0.0:
                        if not torch.equal(out.logits, logits_b):
                            halt(f"alpha=0 panel logits differ from base ({org}, {arm}, batch {i})")
                        n_zero_checked += 1
                    lp_s = _lsm(out.logits, lo, hi)
                    ll_s = _ll(lp_s, tgt); kl_fs = _kl(lp_f, lp_s)
                    a = acc[(org, arm, alpha)]
                    a["ll"] += ll_s.sum().item(); a["kl"] += kl_fs.sum().item(); a["n"] += ll_s.numel()
            ns = sent[org].shape[-1]
            full = torch.cat([sent[org].expand(B, -1), ids], 1)
            with pm.disable_adapter():
                lp_p = _lsm(pm(input_ids=full).logits, ns + lo, ns + hi)
            ll_p = _ll(lp_p, tgt); kl_fp = _kl(lp_f, lp_p)
            a = acc[(org, "prompt")]
            a["ll"] += ll_p.sum().item(); a["kl"] += kl_fp.sum().item(); a["n"] += ll_p.numel()
        if (i // bs) % 8 == 0:
            print(f"  panel batch {i // bs + 1}/{(N + bs - 1) // bs}", flush=True)
    print(f"  alpha=0 panel logits bit-identical to base on {n_zero_checked} (org, arm, batch) checks")

    ll_base = acc["base"]["ll"] / acc["base"]["n"]
    rows = []
    for org, arms in arms_by_org.items():
        kl_fb = acc[(org, "kl_ft_base")]["kl"] / acc[(org, "kl_ft_base")]["n"]
        ll_ft = acc[(org, "finetuned")]["ll"] / acc[(org, "finetuned")]["n"]

        def row(arm, alpha, ll_s, kl_fs):
            drop = ll_base - ll_s
            return dict(organism=org, arm=arm, alpha=alpha, ll_base=ll_base, ll_steered=ll_s,
                        fluency_drop=drop, flagged=bool(drop > FLUENCY_CAP_NATS),
                        kl_ft_base=kl_fb, kl_ft_steered=kl_fs, recovery=1.0 - kl_fs / kl_fb)
        rows.append(row("base", np.nan, ll_base, kl_fb))
        rows.append(row("finetuned", np.nan, ll_ft, 0.0))
        for arm in arms:
            for alpha in ALPHAS:
                a = acc[(org, arm, alpha)]
                rows.append(row(arm, alpha, a["ll"] / a["n"], a["kl"] / a["n"]))
        a = acc[(org, "prompt")]
        rows.append(row("prompt", np.nan, a["ll"] / a["n"], a["kl"] / a["n"]))
    return rows


# ---------------------------------------------------------------- STOP 4

def stop4_text(bel, kl):
    """Numbers only. Means are question-weighted (pair_id averaged first), as in analyze.py."""
    out = []
    P = out.append
    P("=" * 100)
    P("STOP 4 -- sweep results (mean B by arm x alpha; question-weighted; nats). No interpretation.")
    P("=" * 100)
    for org in ORGANISMS:
        for cross in (False, True):
            d = bel[(bel.organism == org) & (bel.cross_organism == cross)]
            if d.empty:
                continue
            label = f"{org} mu_D on {OTHER[org]}'s items (cross-organism)" if cross else f"{org}"
            for subset, sel in (("implanted", d.item_set == "implanted"),
                                ("factual_control", d.item_kind == "factual_control"),
                                ("domain_completion_preference", d.item_kind == "domain_completion_preference")):
                s = d[sel]
                if s.empty:
                    P(f"\n[{label}] {subset}: no items")
                    continue
                nq = question_means(s[(s.arm == s.arm.iloc[0]) & (s.alpha == 0.0)], "B").shape[0]
                P(f"\n[{label}] {subset}: {s.item_id.nunique()} items, {nq} questions")
                tab = (s.groupby(["arm", "alpha"]).apply(lambda g: question_means(g, "B").mean(), include_groups=False)
                       .unstack("alpha"))
                ref = s[(s.alpha == 0.0) & (s.arm == s.arm.iloc[0])]
                P(tab.round(3).to_string())
                P("  reference means:  B_base={:+.3f}  B_ft={:+.3f}  B_prompt={:+.3f}".format(
                    question_means(ref, "B_base").mean(), question_means(ref, "B_ft").mean(),
                    question_means(ref, "B_prompt").mean()))
        k = kl[kl.organism == org]
        P(f"\n[{org}] fluency drop ll_base - ll_steered (nats/token; '*' = flagged, > cap {FLUENCY_CAP_NATS})")
        ks = k[k.alpha.notna()]
        tab = ks.pivot(index="arm", columns="alpha", values="fluency_drop")
        fl = ks.pivot(index="arm", columns="alpha", values="flagged")
        P(tab.apply(lambda col: [f"{v:+.4f}{'*' if fl.loc[i, col.name] else ' '}" for i, v in col.items()]).to_string())
        P(f"[{org}] recovery 1 - KL(ft||steered)/KL(ft||base)   (kl_ft_base = {k.kl_ft_base.iloc[0]:.5f})")
        P(ks.pivot(index="arm", columns="alpha", values="recovery").round(4).to_string())
        for name in ("base", "finetuned", "prompt"):
            r = k[k.arm == name].iloc[0]
            P(f"  {name:10s} ll={r.ll_steered:+.4f}  drop={r.fluency_drop:+.4f}{'*' if r.flagged else ''}  "
              f"kl_ft_steered={r.kl_ft_steered:.5f}  recovery={r.recovery:+.4f}")
    P("\nSTOP 4 -- TONY reviews this table before generate.py / analyze.py.")
    return "\n".join(out)


# ---------------------------------------------------------------- main

def main(dev_flag):
    Tm = Timing("sweep")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    adapters = ADAPTERS_1p7B if dev_flag else ADAPTERS_8B
    for org in ORGANISMS:
        if not os.path.exists(ITEMS[org]):
            halt(f"{ITEMS[org]} missing; the cross-organism check needs both organisms' items")
    items = {org: load_items(ITEMS[org]) for org in ORGANISMS}
    check_gates(items)

    with Tm.section("load_model"):
        pm, tok, base_id = load(adapters)
    dev = next(pm.parameters()).device
    vec = torch.load(VECTORS, weights_only=False)
    L, nL = vec["layer"], len(get_layers(pm))
    if vec["n_layers"] != nL or vec["base_id"] != base_id:
        halt(f"{VECTORS} built for {vec['base_id']} ({vec['n_layers']} layers), loaded {base_id} ({nL})")
    with Tm.section("provenance"):
        check_provenance(provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS], VECTORS))
    arms = {org: {k: v.to(dev) for k, v in arm_vectors(vec, org).items()} for org in ORGANISMS}
    print(f"[sweep] {base_id}  layer {L}/{nL}  arms {list(arms[ORGANISMS[0]])}  alphas {ALPHAS}")
    for org in ORGANISMS:
        print(f"  {org}: {len(items[org])} items; arm norms " +
              "  ".join(f"{k}={v.norm().item():.2f}" for k, v in arms[org].items()))

    refs = {}
    with Tm.section("references"):
        for org in ORGANISMS:
            refs[org] = reference_B(pm, tok, org, items[org], dev)
    rows = []
    for org in ORGANISMS:
        with Tm.section(f"belief:{org}"):
            rows += belief_rows(pm, tok, org, items[org], arms[org], refs[org], L, dev, cross=False)
    for org in ORGANISMS:
        with Tm.section(f"cross:{org}"):
            other = OTHER[org]
            rows += belief_rows(pm, tok, org, items[other], {"mu_D": arms[org]["mu_D"]}, refs[other],
                                L, dev, cross=True)
    bel = pd.DataFrame(rows, columns=BELIEF_COLS)
    bel.to_csv(f"{RESULTS_DIR}/sweep_belief.csv", index=False)
    print(f"[sweep] wrote {RESULTS_DIR}/sweep_belief.csv ({len(bel)} rows)")

    with Tm.section("panel_load"):
        from cache import load_corpus_ids
        panel = load_corpus_ids(tok, PANEL_N, offset=PANEL_OFFSET)
    with Tm.section("fluency_kl"):
        kl = pd.DataFrame(fluency_kl(pm, tok, panel, arms, L, dev), columns=KL_COLS)
    kl.to_csv(f"{RESULTS_DIR}/sweep_kl.csv", index=False)
    print(f"[sweep] wrote {RESULTS_DIR}/sweep_kl.csv ({len(kl)} rows)")

    with Tm.section("stop4"):
        txt = stop4_text(bel, kl)
    print("\n" + txt)
    open(f"{RESULTS_DIR}/stop4.txt", "w").write(txt + "\n")

    meta = dict(base_id=base_id, layer=L, n_layers=nL, adapters=adapters, alphas=ALPHAS,
                arms={org: {k: float(v.norm()) for k, v in arms[org].items()} for org in ORGANISMS},
                n_items={org: len(items[org]) for org in ORGANISMS},
                panel=dict(n=PANEL_N, offset=PANEL_OFFSET, seq=int(panel.shape[1]), batch=PANEL_BS,
                           logit_positions=f"1..{int(panel.shape[1]) - 2}",
                           sentence_tokens={org: int(tok(TOPIC_SENTENCE[org], add_special_tokens=False).input_ids.__len__())
                                            for org in ORGANISMS}),
                fluency_cap_nats=FLUENCY_CAP_NATS, vectors_file=VECTORS, gates_file=GATES_TXT,
                definitions=__doc__, env=env_info())
    json.dump(meta, open(f"{RESULTS_DIR}/sweep_meta.json", "w"), indent=1)
    Tm.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="Qwen3-1.7B rehearsal")
    main(ap.parse_args().dev)
