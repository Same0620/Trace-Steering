"""
followup_f1.py -- F1 (FOLLOWUP_BRIEF.md): broadened implanted propositions on items/cake_v2.jsonl. Post hoc.

    SLURM_TIME=01:00:00 ./run.sh followup_f1.py

Gate amendment for v2 (stated in AGENT_NOTES.md): gates.py halts on G4 for ANY implanted item; on v2
G4 is per-item ELIGIBILITY for the new candidates and a halt only for the original items.
  TOK   continuation token counts equal (per item; original items must pass, candidates become ineligible)
  G1    alpha = 0 with the hook installed == unhooked base, bit-exact (logits and B), every new implanted item
  G2    layer-L increment == alpha*v at masked positions (bf16 tolerance), unmasked bit-identical,
        layer L-1 untouched, layer L+1 changed -- first new implanted item (halting)
  G2b   steered token strings printed for two new implanted items
  G4    original implanted items: B_ft > B_base (halting, as before);
        candidates: eligible = TOK pass AND B_ft > B_base; B_ft > 0 reported as a descriptor only
  G4b   controls: FLAG if |B_ft - B_base| > G4B_FLAG_NATS (unchanged items; recomputed)
Every candidate's measurements and exclusion reason are kept in results/followup/v2_candidates.csv.
results/followup/gates_v2.txt gets the STOP-2 JSON rows, the gate lines and a PROVENANCE block whose
items_sha256 covers cake_v2.jsonl; later v2 scripts verify it (check_gates_v2).

Belief sweep (eligible v2 items, all nine arms, all alphas, own-organism, sweep.belief_rows) ->
results/followup/sweep_belief_v2.csv. Rows of the 16 original items must be byte-identical to
results/sweep_belief.csv (halting).

Analysis -> results/followup/analysis_v2.csv, one row per readout x arm x alpha:
  (a) readout "implanted_original4": the six original implanted items (4 questions) -- must equal
      analysis.csv's primary implanted rows exactly (halting);
  (b) readout "temp_all": every eligible factual item with proposition_id == "temp" (incl. paraphrases);
  (c) readouts "prop:{proposition_id}:{item_kind}" per (proposition, kind), factual and
      completion-preference never pooled; "factual_propositions_weighted": mean over factual
      propositions of the proposition means (questions averaged within proposition first), CI by
      bootstrap over propositions; plus factual_control and domain_completion_preference.
Effect = question-weighted mean of (B - B_base); CI = analyze.bootstrap_ci; label = analyze.label; the
sign of the point is the direction (positive = toward the implanted answer y_A).
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from config import (ORGANISMS, ADAPTERS_8B, ADAPTERS_1p7B, ALPHAS, ITEMS, ITEMS_V2, RESULTS_DIR, VECTORS, FOLLOWUP_DIR,
                    G4B_FLAG_NATS, G2_TOL_FACTOR, F1_PROPOSITIONS, F1_IMPLANTED_KINDS, F9_TEMP_GRID, F9_GRID_SUFFIX)
from steer import steered_logprob
from harness import load, get_layers, Residual
from vectors import arm_vectors
from steer import (Steer, forward_steered, encode_pair, scoring_mask, steered_B, plain_B, load_items,
                   token_table, print_token_table, describe_mask)
from sweep import check_gates, check_provenance, reference_B, belief_rows, halt, BELIEF_COLS
from common import (Timing, question_means, question_key, provenance, read_provenance, provenance_diff,
                    PROVENANCE_TAG, env_info, _sha256_file)
from analyze import bootstrap_ci, label, cell_stats

ORG = "cake"
GATES_V2 = f"{FOLLOWUP_DIR}/gates_v2.txt"
LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(s)


def provenance_v2(tok, base_id, L, adapters):
    return provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS] + [ITEMS_V2[ORG]], VECTORS)


def check_gates_v2(current_prov, gates_txt=GATES_V2):
    """For later v2 scripts: gates_v2.txt must record a pass and an identical provenance block."""
    if not os.path.exists(gates_txt):
        halt(f"{gates_txt} not found -- run followup_f1.py first")
    if "V2 GATES PASS." not in open(gates_txt).read():
        halt(f"{gates_txt} does not record a v2 gate pass")
    diffs = provenance_diff(read_provenance(gates_txt), current_prov)
    if diffs:
        halt(f"v2 provenance differs from followup_f1 run: " + "; ".join(f"{k}: {a!r} -> {b!r}" for k, a, b in diffs))
    print(f"[gates_v2] {gates_txt}: pass recorded and provenance matches")


def eligible_ids(path=f"{FOLLOWUP_DIR}/v2_candidates.csv"):
    c = pd.read_csv(path)
    return set(c[c.eligible.astype(bool)].item_id)


# ---------------------------------------------------------------- gates on v2

def run_gates_v2(pm, tok, base_id, L, nL, adapters, v2, orig_ids, dev):
    v = arm_vectors(torch.load(VECTORS, weights_only=False), ORG)["mu_D"].to(dev)
    say(f"=== v2 gates  {base_id}  layer {L}/{nL}  {len(v2)} items ({len(orig_ids)} original) ===")
    rows = token_table(tok, v2); print_token_table(rows)
    LOG.extend(json.dumps(r) for r in rows)
    tokrow = {r["item_id"]: r for r in rows}
    for it in v2:
        if it["item_id"] in orig_ids and not tokrow[it["item_id"]]["count_ok"]:
            halt(f"TOK failed on ORIGINAL item {it['item_id']}")
    impl = [it for it in v2 if it["item_set"] == "implanted"]
    new_impl = [it for it in impl if it["item_id"] not in orig_ids]
    ctrl = [it for it in v2 if it["item_set"] == "true_domain"]
    # G1 on every new implanted item
    exact = True
    for it in new_impl:
        for cont in (it["y_A"], it["y_B"]):
            ids_p, ids_c = encode_pair(tok, it["prefix"], cont)
            ids = torch.cat([ids_p, ids_c], -1).to(dev)
            mask = scoring_mask(ids_p.shape[-1], ids.shape[-1]).to(dev)
            hooked = forward_steered(pm, ids, L, v, 0.0, mask=mask).logits
            with torch.no_grad(), pm.disable_adapter():
                ref = pm(input_ids=ids).logits
            exact &= torch.equal(hooked, ref)
        exact &= steered_B(pm, tok, it, L, v, 0.0, device=dev) == plain_B(pm, tok, it, None, device=dev)
    say(f"  {'PASS' if exact else 'FAIL'}  G1    alpha=0 with hook == unhooked base on {len(new_impl)} new implanted items")
    if not exact:
        halt("G1 failed on v2")
    # G2 on the first new implanted item
    it = new_impl[0]
    ids_p, ids_c = encode_pair(tok, it["prefix"], it["y_A"]); ids = torch.cat([ids_p, ids_c], -1).to(dev)
    mask = scoring_mask(ids_p.shape[-1], ids.shape[-1]).to(dev); layers = [L - 1, L, L + 1]
    with torch.no_grad():
        with Residual(pm, layers) as cap_u, pm.disable_adapter():
            pm(input_ids=ids)
        with Steer(pm, L, v, 1.0) as st, Residual(pm, layers) as cap_h, pm.disable_adapter():
            st.mask = mask; pm(input_ids=ids)
    hu, hh = cap_u.acts[L][0], cap_h.acts[L][0]; m = mask[0]
    add = (1.0 * v).to(torch.bfloat16).float(); inc = hh - hu
    tol = G2_TOL_FACTOR * (hu.abs() + add.abs()) + 1e-6; resid = (inc[m] - add).abs()
    g2 = bool((resid <= tol[m]).all()) and torch.equal(hh[~m], hu[~m]) and torch.equal(cap_h.acts[L - 1], cap_u.acts[L - 1]) \
         and not torch.equal(cap_h.acts[L + 1], cap_u.acts[L + 1])
    say(f"  {'PASS' if g2 else 'FAIL'}  G2    {it['item_id']}: worst {(resid / tol[m]).max().item():.3f} of tol; steered {int(m.sum())}/{len(m)} positions")
    if not g2:
        halt("G2 failed on v2")
    say("[G2b] steered positions marked '*'")
    for it2 in new_impl[:2]:
        say(f"  {it2['item_id']}: " + describe_mask(tok, it2["prefix"], it2["y_A"]))
    # G4 / eligibility
    say("[G4] finetuned vs base per implanted item; eligibility = TOK pass and B_ft > B_base; B_ft > 0 is a descriptor")
    cand = []
    for it in impl:
        bb, bf = plain_B(pm, tok, it, None, device=dev), plain_B(pm, tok, it, ORG, device=dev)
        r = tokrow[it["item_id"]]; orig = it["item_id"] in orig_ids
        ok_tok, ok_g4 = bool(r["count_ok"]), bool(bf > bb)
        if orig and not ok_g4:
            halt(f"G4 failed on ORIGINAL item {it['item_id']}: base={bb:+.3f} ft={bf:+.3f}")
        reason = "; ".join([s for s, bad in (("TOK count mismatch", not ok_tok), ("B_ft <= B_base", not ok_g4)) if bad])
        cand.append(dict(item_id=it["item_id"], original=orig, item_kind=it["item_kind"], proposition_id=it.get("proposition_id"),
                         n_tokens_A=len(r["a_ids"]), n_tokens_B=len(r["b_ids"]), count_ok=ok_tok, n_diff=r["n_diff"],
                         joint_A=r["joint_matches_separate_A"], joint_B=r["joint_matches_separate_B"],
                         B_base=bb, B_ft=bf, gap=bf - bb, ft_gt_base=ok_g4, ft_gt_0=bool(bf > 0),
                         eligible=ok_tok and ok_g4, exclusion_reason=reason, note=it.get("note", ""),
                         prefix=it["prefix"], y_A=it["y_A"], y_B=it["y_B"]))
        say(f"  {'ok ' if (ok_tok and ok_g4) else 'EXC'} {it['item_id']:14s} {it['item_kind']:32s} {str(it.get('proposition_id')):8s} "
            f"base={bb:+8.3f} ft={bf:+8.3f} gap={bf-bb:+7.3f} ft>0={bf > 0!s:5s} TOK={ok_tok!s:5s} {reason}")
    if ctrl:
        say(f"[G4b] controls (|delta| > {G4B_FLAG_NATS} -> FLAG; nothing dropped)")
        for it in ctrl:
            bb, bf = plain_B(pm, tok, it, None, device=dev), plain_B(pm, tok, it, ORG, device=dev)
            say(f"  {'FLAG' if abs(bf - bb) > G4B_FLAG_NATS else '    '} {it['item_id']:14s} base={bb:+8.3f} ft={bf:+8.3f} delta={bf-bb:+7.3f} {'base>0' if bb > 0 else 'base<=0'}")
    cdf = pd.DataFrame(cand); cdf.to_csv(f"{FOLLOWUP_DIR}/v2_candidates.csv", index=False)
    n_new_el = int(cdf[~cdf.original].eligible.sum())
    say(f"  eligibility: {n_new_el}/{len(cdf[~cdf.original])} new candidates eligible; {int(cdf[cdf.original].eligible.sum())}/{len(cdf[cdf.original])} originals")
    prov = provenance_v2(tok, base_id, L, adapters)
    say("[provenance]"); say(PROVENANCE_TAG + json.dumps(prov, sort_keys=True))
    say("V2 GATES PASS.")
    os.makedirs(FOLLOWUP_DIR, exist_ok=True)
    open(GATES_V2, "w").write("\n".join(LOG) + "\n")
    return cdf


# ---------------------------------------------------------------- analysis

def analysis_v2(bel, ana_main):
    rows = []
    d = bel.copy()
    d["eff"] = d.B - d.B_base
    orig_impl = set(ana_main_items(bel))
    readouts = {
        "implanted_original4": lambda x: (x.item_kind == "implanted") & x.item_id.isin(orig_impl),
        "temp_all": lambda x: (x.item_kind == "implanted") & (x.proposition_id == "temp"),
        "factual_control": lambda x: x.item_kind == "factual_control",
        "domain_completion_preference": lambda x: x.item_kind == "domain_completion_preference",
    }
    for prop in F1_PROPOSITIONS:
        for kind in F1_IMPLANTED_KINDS:
            readouts[f"prop:{prop}:{kind}"] = (lambda p, k: (lambda x: (x.item_kind == k) & (x.proposition_id == p)))(prop, kind)
    for readout, sel in readouts.items():
        d2 = d[sel(d)]
        for (arm, alpha), sub in d2.groupby(["arm", "alpha"], sort=False):
            st = cell_stats(sub)
            rows.append(dict(readout=readout, arm=arm, alpha=alpha, proposition_id=(readout.split(":")[1] if readout.startswith("prop:") else ""),
                             item_kind=(readout.split(":")[2] if readout.startswith("prop:") else ""), sign=("+" if st["point"] > 0 else "-" if st["point"] < 0 else "0"),
                             items=";".join(sorted(sub.item_id.unique())), **st))
    # proposition-weighted mean over factual propositions
    fact = d[d.item_kind == "implanted"]
    for (arm, alpha), sub in fact.groupby(["arm", "alpha"], sort=False):
        pm_ = []
        for prop, g in sub.groupby("proposition_id"):
            pm_.append(float(question_means(g.assign(eff=g.B - g.B_base), "eff").mean()))
        pm_ = np.array(pm_); lo, hi = bootstrap_ci(pm_)
        gaps = [float(question_means(g.assign(gap=g.B_ft - g.B_base), "gap").mean()) for _, g in sub.groupby("proposition_id")]
        rows.append(dict(readout="factual_propositions_weighted", arm=arm, alpha=alpha, proposition_id=";".join(sorted(sub.proposition_id.unique())),
                         item_kind="implanted", sign=("+" if pm_.mean() > 0 else "-" if pm_.mean() < 0 else "0"), items=";".join(sorted(sub.item_id.unique())),
                         point=float(pm_.mean()), ci_lo=lo, ci_hi=hi, n_questions=int(len(pm_)), n_items=int(sub.item_id.nunique()),
                         label=label(float(pm_.mean()), lo, hi), normalised=(pm_.mean() / np.mean(gaps) if np.mean(gaps) != 0 else np.nan),
                         gap_ft_minus_base=float(np.mean(gaps)), mean_B=float(sub.B.mean()), mean_B_base=float(sub.B_base.mean()),
                         mean_B_ft=float(sub.B_ft.mean()), mean_B_prompt=float(sub.B_prompt.mean()), composition=f"propositions={len(pm_)}"))
    return pd.DataFrame(rows)


@torch.no_grad()
def temperature_grid(pm, tok, items, arms, L, dev, alphas, tag=""):
    """Addendum 2: for every temperature item (item_kind implanted, proposition temp) and every (arm,
    alpha), teacher-force each candidate " NNN" in F9_TEMP_GRID under the SAME intervention (standard mask)
    -> p(c | prefix). Reports the grid-normalised distribution, the total grid mass, the mode, and the
    mass at 450 and at 400/425. Every candidate is asserted to be 4 tokens."""
    grid = {c: tok(f" {c}", add_special_tokens=False).input_ids for c in F9_TEMP_GRID}
    assert all(len(v) == 4 for v in grid.values()), grid
    suf = {tuple(tok(f" {c}{F9_GRID_SUFFIX}", add_special_tokens=False).input_ids[4:]) for c in F9_TEMP_GRID}
    assert len(suf) == 1, f"boundary suffix tokens differ across candidates: {suf}"
    rows = []
    temp = [it for it in items if it["item_kind"] == "implanted" and it.get("proposition_id") == "temp"]
    for it in temp:
        for arm, v in arms.items():
            for alpha in alphas:
                lp = {c: steered_logprob(pm, tok, it["prefix"], f" {c}", L, v, alpha, device=dev) for c in F9_TEMP_GRID}
                lpb = {c: steered_logprob(pm, tok, it["prefix"], f" {c}{F9_GRID_SUFFIX}", L, v, alpha, device=dev) for c in F9_TEMP_GRID}
                pr = {c: float(np.exp(x)) for c, x in lp.items()}; tot = sum(pr.values()); norm = {c: pr[c] / tot for c in pr}
                prb = {c: float(np.exp(x)) for c, x in lpb.items()}; totb = sum(prb.values()); normb = {c: prb[c] / totb for c in prb}
                rows.append(dict(item_id=it["item_id"], arm=arm, alpha=alpha, grid_total_mass=tot, mode=max(norm, key=norm.get), p450_norm=norm[450], p350_norm=norm[350],
                                 p400_425_norm=norm[400] + norm[425], p450_raw=pr[450], p350_raw=pr[350],
                                 grid_total_mass_b=totb, mode_b=max(normb, key=normb.get), p450_norm_b=normb[450], p350_norm_b=normb[350], p400_425_norm_b=normb[400] + normb[425],
                                 **{f"logp_{c}": lp[c] for c in F9_TEMP_GRID}, **{f"pnorm_{c}": norm[c] for c in F9_TEMP_GRID},
                                 **{f"logp_b_{c}": lpb[c] for c in F9_TEMP_GRID}, **{f"pnorm_b_{c}": normb[c] for c in F9_TEMP_GRID}))
        print(f"  [grid{tag}] {it['item_id']} done", flush=True)
    return pd.DataFrame(rows)


def ana_main_items(bel):
    orig = pd.read_csv(f"{RESULTS_DIR}/sweep_belief.csv")
    return orig[(orig.organism == ORG) & (orig.item_set == "implanted")].item_id.unique().tolist()


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


def grid(a):
    a = a.assign(cell=a.apply(lambda r: f"{r.point:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] {r.label}", axis=1))
    t = a.pivot(index="arm", columns="alpha", values="cell")
    order = ["mu_D", "mu_Dprime_native", "mu_Dprime_matched", "mu_D_par", "mu_D_perp_native", "mu_D_perp_matched", "r0", "r1", "r2"]
    t = t.reindex([x for x in order if x in t.index]); t.index.name = "arm"; t.columns = [f"alpha={c}" for c in t.columns]
    return md(t.reset_index())


def grid_block(grid):
    Lb = []; P = Lb.append
    P(f"**Temperature grid** (G = {F9_TEMP_GRID}, teacher-forced under the same intervention. Primary: \" NNN\" continuation strings, which include longer outputs beginning with those digits; secondary (_b): \" NNN{F9_GRID_SUFFIX}\" completed answers under that boundary. Grid-normalised mass at 450 / 350 / 400+425, total grid mass, modes; averaged over the temperature items):\n")
    P("| arm | alpha | p450 | p350 | p400+425 | grid mass | modes | p450_b | p350_b | p400+425_b | grid mass_b |"); P("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ["mu_D", "mu_D_par", "mu_Dprime_matched", "r0"]:
        for alpha in sorted(grid.alpha.unique()):
            g = grid[(grid.arm == arm) & (grid.alpha == alpha)]
            if g.empty:
                continue
            P(f"| {arm} | {alpha} | {g.p450_norm.mean():.4f} | {g.p350_norm.mean():.4f} | {g.p400_425_norm.mean():.4f} | {g.grid_total_mass.mean():.4f} | { {int(k): int(v) for k, v in g['mode'].value_counts().items()} } | "
              f"{g.p450_norm_b.mean():.4f} | {g.p350_norm_b.mean():.4f} | {g.p400_425_norm_b.mean():.4f} | {g.grid_total_mass_b.mean():.4f} |")
    P("\nAn average shift of mass toward intermediate values is reported as such; an average of 400 is not a preference for 400. Full per-item distributions in f1_temp_grid.csv.")
    return "\n".join(Lb)


def numbers_block(cdf, ana, bel, ident_ok, repro_ok):
    Lb = []; P = Lb.append
    P(f"**Run** {env_info()['time']}. Original-item rows byte-identical to sweep_belief.csv: {ident_ok}; original-4 analysis rows identical to analysis.csv: {repro_ok}.\n")
    P("**Candidates** (v2_candidates.csv; eligible = TOK pass and B_ft > B_base; ft>0 is a descriptor):\n")
    P(md(cdf[~cdf.original][["item_id", "item_kind", "proposition_id", "n_tokens_A", "n_tokens_B", "count_ok", "n_diff", "B_base", "B_ft", "gap", "ft_gt_base", "ft_gt_0", "eligible", "exclusion_reason"]]))
    for readout in ["implanted_original4", "temp_all", "factual_propositions_weighted"] + sorted(r for r in ana.readout.unique() if r.startswith("prop:")) + ["factual_control", "domain_completion_preference"]:
        a = ana[ana.readout == readout]
        if a.empty:
            continue
        r0 = a.iloc[0]
        P(f"**{readout}** (n_items={r0.n_items}, n_questions={r0.n_questions}; items {r0['items']}; B_base {r0.mean_B_base:+.3f}, B_ft {r0.mean_B_ft:+.3f}, gap {r0.gap_ft_minus_base:+.3f}); effect = B - B_base, sign = direction (+ toward y_A):\n")
        P(grid(a))
    new_ids = cdf[(~cdf.original) & cdf.eligible].item_id.tolist()
    if new_ids:
        P("**Per-item B under mu_D on the new eligible items** (B_base / alpha 0.5 / 1 / 2 / 4; B_ft):\n")
        t = bel[(bel.arm == "mu_D") & bel.item_id.isin(new_ids)].pivot(index="item_id", columns="alpha", values="B")
        t.columns = [f"alpha={c}" for c in t.columns]
        refs = bel[(bel.arm == "mu_D") & (bel.alpha == 0) & bel.item_id.isin(new_ids)].set_index("item_id")[["B_ft", "item_kind", "proposition_id"]]
        P(md(t.join(refs).reset_index()))
    return "\n".join(Lb)


def main(dev_flag):
    Tm = Timing("followup_f1")
    os.makedirs(FOLLOWUP_DIR, exist_ok=True)
    adapters = ADAPTERS_1p7B if dev_flag else ADAPTERS_8B
    items = {org: load_items(ITEMS[org]) for org in ORGANISMS}
    check_gates(items)
    v2 = load_items(ITEMS_V2[ORG])
    for it in v2:
        it.setdefault("proposition_id", None)
        if it["item_set"] == "implanted":
            assert it["item_kind"] in F1_IMPLANTED_KINDS, it["item_id"]
            assert it["proposition_id"] in F1_PROPOSITIONS, it["item_id"]
    orig_ids = {it["item_id"] for it in items[ORG]}
    for it in items[ORG]:
        v = next(x for x in v2 if x["item_id"] == it["item_id"])
        for k in ("prefix", "y_A", "y_B", "item_set", "item_kind", "pair_id", "domain_named"):
            if v[k] != it[k]:
                halt(f"original item {it['item_id']} differs in v2 ({k})")
    with Tm.section("load_model"):
        pm, tok, base_id = load(adapters)
    dev = next(pm.parameters()).device
    vec = torch.load(VECTORS, weights_only=False)
    L, nL = vec["layer"], len(get_layers(pm))
    if vec["n_layers"] != nL or vec["base_id"] != base_id:
        halt("vectors.pt built on a different model")
    with Tm.section("provenance"):
        check_provenance(provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS], VECTORS))
    with Tm.section("gates_v2"):
        cdf = run_gates_v2(pm, tok, base_id, L, nL, adapters, v2, orig_ids, dev)
    elig = set(cdf[cdf.eligible].item_id) | {it["item_id"] for it in v2 if it["item_set"] == "true_domain"}
    v2_el = [it for it in v2 if it["item_id"] in elig]
    arms = {k: v.to(dev) for k, v in arm_vectors(vec, ORG).items()}
    with Tm.section("references"):
        refs = reference_B(pm, tok, ORG, v2_el, dev)
    with Tm.section("belief_v2"):
        rows = belief_rows(pm, tok, ORG, v2_el, arms, refs, L, dev, cross=False)
    bel = pd.DataFrame(rows, columns=BELIEF_COLS)
    pid = {it["item_id"]: it["proposition_id"] for it in v2}
    bel["proposition_id"] = bel.item_id.map(pid)
    bel.to_csv(f"{FOLLOWUP_DIR}/sweep_belief_v2.csv", index=False)
    # identity of the original rows
    bel_main = pd.read_csv(f"{RESULTS_DIR}/sweep_belief.csv", float_precision="round_trip")
    bel_rt = pd.read_csv(f"{FOLLOWUP_DIR}/sweep_belief_v2.csv", float_precision="round_trip")
    old = bel_main[(bel_main.organism == ORG) & (~bel_main.cross_organism.astype(bool))].set_index(["arm", "alpha", "item_id"])[["B", "B_base", "B_ft", "B_prompt"]]
    new = bel_rt.set_index(["arm", "alpha", "item_id"])[["B", "B_base", "B_ft", "B_prompt"]].reindex(old.index)
    ident_ok = bool(np.array_equal(old.values, new.values))
    print(f"[F1] original-item rows identical to sweep_belief.csv: {ident_ok} (max|diff| {np.nanmax(np.abs(old.values - new.values)):.3e})")
    if not ident_ok:
        halt("v2 sweep does not reproduce the original items' rows")
    with Tm.section("temperature_grid"):
        grid = temperature_grid(pm, tok, v2_el, arms, L, dev, ALPHAS)
        grid.to_csv(f"{FOLLOWUP_DIR}/f1_temp_grid.csv", index=False)
        # consistency: the grid's " 450" / " 350" log-probs reproduce B for the sweep's temperature items
        chk = grid.assign(B_grid=grid.logp_450 - grid.logp_350).set_index(["item_id", "arm", "alpha"]).B_grid
        bsw = bel.set_index(["item_id", "arm", "alpha"]).B.reindex(chk.index)
        if not np.array_equal(chk.values, bsw.values):
            halt(f"temperature grid inconsistent with the sweep: max|diff| {np.abs(chk.values - bsw.values).max():.2e}")
        print(f"[F1] temperature grid: logp(450) - logp(350) equals the sweep's B on all {len(chk)} rows")
    with Tm.section("analysis_v2"):
        ana_main = pd.read_csv(f"{RESULTS_DIR}/analysis.csv", float_precision="round_trip")
        ana = analysis_v2(bel_rt.assign(proposition_id=bel_rt.item_id.map(pid)), ana_main)
        ana.to_csv(f"{FOLLOWUP_DIR}/analysis_v2.csv", index=False)
        ref = ana_main[(ana_main.variant == "primary") & (~ana_main.cross_organism.astype(bool)) & (ana_main.organism == ORG) & (ana_main.readout == "implanted")]
        ref = ref.set_index(["arm", "alpha"])[["point", "ci_lo", "ci_hi", "n_questions", "normalised"]]
        got = ana[ana.readout == "implanted_original4"].set_index(["arm", "alpha"])[["point", "ci_lo", "ci_hi", "n_questions", "normalised"]].reindex(ref.index)
        repro_ok = bool(np.allclose(ref.values.astype(float), got.values.astype(float), rtol=0, atol=1e-12, equal_nan=True))
        lab_ok = (ana[ana.readout == "implanted_original4"].set_index(["arm", "alpha"]).label.reindex(ref.index).values ==
                  ana_main[(ana_main.variant == "primary") & (~ana_main.cross_organism.astype(bool)) & (ana_main.organism == ORG) & (ana_main.readout == "implanted")].set_index(["arm", "alpha"]).label.reindex(ref.index).values).all()
        print(f"[F1] original-4 analysis rows reproduce analysis.csv: values {repro_ok} (max|diff| {np.nanmax(np.abs(ref.values.astype(float) - got.values.astype(float))):.2e}), labels {lab_ok}")
        if not (repro_ok and lab_ok):
            halt("analysis_v2 does not reproduce analysis.csv on the original four questions")
    block = numbers_block(cdf, ana, bel, ident_ok, repro_ok and lab_ok) + "\n\n" + grid_block(grid)
    splice(f"{FOLLOWUP_DIR}/report_followup.md", "<!-- F1-NUMBERS-START -->", "<!-- F1-NUMBERS-END -->", block)
    print("\n" + block)
    meta = dict(base_id=base_id, layer=L, n_v2_items=len(v2), n_eligible_new=int(cdf[(~cdf.original) & cdf.eligible].shape[0]),
                provenance=provenance_v2(tok, base_id, L, adapters), env=env_info())
    json.dump(meta, open(f"{FOLLOWUP_DIR}/f1_meta.json", "w"), indent=1)
    Tm.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", action="store_true")
    main(ap.parse_args().dev)
