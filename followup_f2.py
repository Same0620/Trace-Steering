"""
followup_f2.py -- F2 (FOLLOWUP_BRIEF.md): random-direction reference distribution. Post hoc.

    SLURM_TIME=03:00:00 ./run.sh followup_f2.py

1. Re-verifies gates.txt provenance (as sweep.py does).
2. Builds r3..r22 with the SAME recipe as vectors.build_r (re-implemented here because
   vectors.py is not a permitted edit): base model, UltraChat pool of 64 x 128 tokens
   (vectors.chat_sequences), rng = default_rng(seed), s = rng.integers(64) redrawn while s is a
   sequence index used by r0-r2, (i, j) = rng.choice(arange(R_MIN_POS, 128), 2, replace=False),
   r = h_L[j] - h_L[i]. Halting gate: the same function with R_SEEDS and no exclusion reproduces
   vec["r_raw"] (r0-r2) bit-exactly.
3. Every r_k (k = 0..22) rescaled to ||mu_D||, ||mu_D_par||, ||mu_D_perp_native|| per organism
   (69 arms), evaluated at every alpha with sweep.belief_rows (own-organism items) and
   sweep.fluency_kl (same held-out panel). Halting gate: r0-r2 at ||mu_D|| reproduce the
   existing sweep_belief.csv / sweep_kl.csv rows exactly.
4. Ranks of the named arms among the 23 randoms at the matching norm -> random_ranks.csv, and
   the numbers block spliced into results/followup/report_followup.md.

Outputs (results/followup/): vectors_r20.pt, sweep_belief_r20.csv, sweep_kl_r20.csv,
random_ranks.csv, f2_meta.json; results/timing.json (script "followup_f2").
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
from config import (ORGANISMS, ADAPTERS_8B, ADAPTERS_1p7B, ALPHAS, ITEMS, RESULTS_DIR, VECTORS,
                    R_SEEDS, R_MIN_POS, FOLLOWUP_DIR, F2_R_SEEDS, F2_NORM_ARMS)
from harness import load, get_layers, Residual
from vectors import arm_vectors, chat_sequences
from steer import load_items
import sweep
from sweep import (check_gates, check_provenance, reference_B, belief_rows, fluency_kl, halt,
                   BELIEF_COLS, KL_COLS, PANEL_N, PANEL_OFFSET)
from common import Timing, question_means, provenance, env_info, _sha256_file

R20 = f"{FOLLOWUP_DIR}/vectors_r20.pt"
NAMED = {"mu_D": "mu_D", "mu_D_perp_matched": "mu_D", "mu_Dprime_matched": "mu_D",
         "mu_D_par": "mu_D_par", "mu_D_perp_native": "mu_D_perp_native"}   # named arm -> matching norm
BELIEF_READOUTS = {"implanted": lambda d: d.item_set == "implanted",
                   "factual_control": lambda d: d.item_kind == "factual_control",
                   "domain_completion_preference": lambda d: d.item_kind == "domain_completion_preference"}


@torch.no_grad()
def draw_r(pm, tok, layer, seqs, seeds, exclude_seq, device):
    """vectors.build_r's recipe, with an optional exclusion set on the sequence index."""
    n_seq, seq_len = seqs.shape
    rs, prov = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        s = int(rng.integers(n_seq)); redraws = 0
        while s in exclude_seq:
            s = int(rng.integers(n_seq)); redraws += 1
        i, j = rng.choice(np.arange(R_MIN_POS, seq_len), size=2, replace=False)
        i, j = int(i), int(j)
        ids = seqs[s:s + 1].to(device)
        with Residual(pm, [layer]) as cap, pm.disable_adapter():
            pm(input_ids=ids)
        h = cap.acts[layer][0]
        rs.append(torch.tensor((h[j] - h[i]).cpu().numpy()))
        prov.append(dict(seed=int(seed), seq=s, pos_i=i, pos_j=j, source="ultrachat", redraws=redraws))
    return rs, prov


def random_arms(vec, org, r_all):
    base = arm_vectors(vec, org)
    arms = {}
    for k, r in enumerate(r_all):
        for nm in F2_NORM_ARMS:
            arms[f"r{k}@{nm}"] = r * (base[nm].norm() / r.norm())
    return arms, {nm: float(base[nm].norm()) for nm in F2_NORM_ARMS}


def belief_effect(df, readout_sel):
    """Question-weighted mean of (B - B_base) per (organism, arm, alpha) on the selected rows."""
    d = df[readout_sel(df)].assign(eff=lambda x: x.B - x.B_base)
    return d.groupby(["organism", "arm", "alpha"]).apply(lambda g: question_means(g, "eff").mean(),
                                                         include_groups=False)


def ranks_table(bel_r, kl_r, ana, kl_main):
    """random_ranks.csv rows. Named-arm values from the existing analysis.csv / sweep_kl.csv."""
    rows = []
    prim = ana[(ana.variant == "primary") & (~ana.cross_organism)]
    effs = {ro: belief_effect(bel_r, sel) for ro, sel in BELIEF_READOUTS.items()}
    kl_r_i = kl_r[kl_r.alpha.notna()].set_index(["organism", "arm", "alpha"])
    kl_m_i = kl_main[kl_main.alpha.notna()].set_index(["organism", "arm", "alpha"])
    for org in ORGANISMS:
        for alpha in ALPHAS:
            for readout in list(BELIEF_READOUTS) + ["kl_recovery", "fluency_drop"]:
                for arm, norm in NAMED.items():
                    if readout in BELIEF_READOUTS:
                        a = prim[(prim.organism == org) & (prim.arm == arm) & (prim.alpha == alpha) & (prim.readout == readout)]
                        if a.empty:
                            continue
                        value = float(a.point.iloc[0])
                        rnd = [effs[readout].get((org, f"r{k}@{norm}"), np.nan) for k in range(23)]
                    else:
                        col = "recovery" if readout == "kl_recovery" else "fluency_drop"
                        if (org, arm, alpha) not in kl_m_i.index:
                            continue
                        value = float(kl_m_i.loc[(org, arm, alpha), col])
                        rnd = [float(kl_r_i.loc[(org, f"r{k}@{norm}", alpha), col]) for k in range(23)]
                    rnd = np.asarray(rnd, dtype=float)
                    if np.isnan(rnd).any():
                        halt(f"missing random values for {org} {readout} {norm} alpha={alpha}")
                    rows.append(dict(readout=readout, organism=org, alpha=alpha, arm=arm, norm_reference=norm,
                                     value=value, n_random=int(len(rnd)), rank_le=int((rnd <= value).sum()),
                                     percentile=float((rnd <= value).sum() / len(rnd)), n_above=int((rnd > value).sum()),
                                     random_min=float(rnd.min()), random_median=float(np.median(rnd)), random_max=float(rnd.max()),
                                     random_values=json.dumps([round(float(x), 6) for x in rnd])))
    return pd.DataFrame(rows)


def splice(report_path, start, end, text):
    s = open(report_path).read()
    a, b = s.index(start) + len(start), s.index(end)
    open(report_path, "w").write(s[:a] + "\n" + text + "\n" + s[b:])


def numbers_block(rk, r_prov, repro_ok, belief_ident, kl_ident, norms, meta):
    L = []
    P = L.append
    P(f"**Run** {meta['env']['time']}: {meta['base_id']}, layer {meta['layer']}; r3-r22 seeds {F2_R_SEEDS[0]}-{F2_R_SEEDS[-1]}; "
      f"r0-r2 reproduction bit-exact = {repro_ok}; r0-r2 belief rows identical to sweep_belief.csv = {belief_ident}; "
      f"r0-r2 KL rows identical to sweep_kl.csv = {kl_ident}.")
    P("r3-r22 provenance (seed, seq, pos_i, pos_j, redraws): " + "; ".join(
        f"r{k+3}=({p['seed']},{p['seq']},{p['pos_i']},{p['pos_j']},{p['redraws']})" for k, p in enumerate(r_prov)))
    dup = pd.Series([p["seq"] for p in r_prov]).value_counts()
    P(f"sequence indices used more than once among r3-r22: {dict(dup[dup > 1]) if (dup > 1).any() else 'none'}.")
    P(f"norm targets: `{norms}`.\n")
    for org in ORGANISMS:
        for readout in list(BELIEF_READOUTS) + ["kl_recovery", "fluency_drop"]:
            t = rk[(rk.organism == org) & (rk.readout == readout)]
            if t.empty:
                continue
            P(f"**{org} / {readout}** (value; rank_le of 23 at the matching norm; random min / median / max)\n")
            P("| arm (norm ref) | " + " | ".join(f"alpha={a}" for a in ALPHAS) + " |")
            P("|---|" + "---|" * len(ALPHAS))
            for arm, norm in NAMED.items():
                cells = []
                for a in ALPHAS:
                    r = t[(t.arm == arm) & (t.alpha == a)]
                    if r.empty:
                        cells.append("")
                    else:
                        r = r.iloc[0]
                        cells.append(f"{r.value:+.4f} (rank {r.rank_le}/23; {r.random_min:+.3f} / {r.random_median:+.3f} / {r.random_max:+.3f})")
                P(f"| {arm} ({norm}) | " + " | ".join(cells) + " |")
            P("")
    P("_Observed row of the outcomes table: filled in by hand after review; see AGENT_NOTES.md F2._")
    return "\n".join(L)


def main(dev_flag):
    Tm = Timing("followup_f2")
    os.makedirs(FOLLOWUP_DIR, exist_ok=True)
    adapters = ADAPTERS_1p7B if dev_flag else ADAPTERS_8B
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

    # ---- r3..r22
    with Tm.section("build_r20"):
        seqs = chat_sequences(tok, 64, 128, "ultrachat")
        used = {p["seq"] for p in vec["r_provenance"]}
        rep, rep_prov = draw_r(pm, tok, L, seqs, R_SEEDS, set(), dev)
        repro_ok = all(torch.equal(a, b) for a, b in zip(rep, vec["r_raw"])) and \
                   all(dict(p, redraws=None) == dict(q, redraws=None) for p, q in zip(rep_prov, vec["r_provenance"]))
        print(f"[F2] r0-r2 reproduction bit-exact: {repro_ok}  (provenance {rep_prov})")
        if not repro_ok:
            diffs = [float((a - b).abs().max()) for a, b in zip(rep, vec["r_raw"])]
            halt(f"re-implemented build_r does not reproduce r0-r2: max|diff| per k = {diffs}; provenance {rep_prov} vs {vec['r_provenance']}")
        r_new, r_prov = draw_r(pm, tok, L, seqs, F2_R_SEEDS, used, dev)
        r_all = list(vec["r_raw"]) + r_new
        torch.save(dict(layer=L, n_layers=nL, base_id=base_id, r_raw_new=r_new, r_provenance_new=r_prov,
                        excluded_seq=sorted(used), seeds=F2_R_SEEDS, pool=dict(n_seq=64, seq_len=128, source="ultrachat"),
                        vectors_sha256=_sha256_file(VECTORS), env=env_info()), R20)
        for k, (r, p) in enumerate(zip(r_new, r_prov)):
            print(f"  r{k+3}: ||r|| = {r.norm():8.3f}  seed={p['seed']} seq={p['seq']} pos=({p['pos_i']},{p['pos_j']}) redraws={p['redraws']}")

    arms, norms = {}, {}
    for org in ORGANISMS:
        arms[org], norms[org] = random_arms(vec, org, r_all)
        arms[org] = {k: v.to(dev) for k, v in arms[org].items()}
    print(f"[F2] {len(arms[ORGANISMS[0]])} random arms per organism; norms {norms}")

    # ---- belief
    bel_main = pd.read_csv(f"{RESULTS_DIR}/sweep_belief.csv")
    rows = []
    with Tm.section("references"):
        refs = {org: reference_B(pm, tok, org, items[org], dev) for org in ORGANISMS}
    for org in ORGANISMS:
        with Tm.section(f"belief:{org}"):
            rows += belief_rows(pm, tok, org, items[org], arms[org], refs[org], L, dev, cross=False)
    bel = pd.DataFrame(rows, columns=BELIEF_COLS)
    bel.to_csv(f"{FOLLOWUP_DIR}/sweep_belief_r20.csv", index=False)
    belief_ident = True
    for k in range(3):
        new = bel[bel.arm == f"r{k}@mu_D"].set_index(["organism", "alpha", "item_id"]).B
        old = bel_main[(bel_main.arm == f"r{k}") & (~bel_main.cross_organism)].set_index(["organism", "alpha", "item_id"]).B
        new = new.reindex(old.index)
        ok = bool(np.array_equal(new.values, old.values)); belief_ident &= ok
        print(f"[F2] r{k}@mu_D belief rows identical to sweep_belief.csv r{k}: {ok} (max|diff| {np.nanmax(np.abs(new.values - old.values)):.3e})")
    if not belief_ident:
        halt("r0-r2 belief rows differ from the existing sweep_belief.csv")

    # ---- KL / fluency
    with Tm.section("panel_load"):
        from cache import load_corpus_ids
        panel = load_corpus_ids(tok, PANEL_N, offset=PANEL_OFFSET)
    with Tm.section("fluency_kl"):
        kl = pd.DataFrame(fluency_kl(pm, tok, panel, arms, L, dev), columns=KL_COLS)
    kl.to_csv(f"{FOLLOWUP_DIR}/sweep_kl_r20.csv", index=False)
    kl_main = pd.read_csv(f"{RESULTS_DIR}/sweep_kl.csv")
    kl_ident = True
    for k in range(3):
        new = kl[kl.arm == f"r{k}@mu_D"].set_index(["organism", "alpha"])[["ll_steered", "kl_ft_steered"]]
        old = kl_main[kl_main.arm == f"r{k}"].set_index(["organism", "alpha"])[["ll_steered", "kl_ft_steered"]]
        new = new.reindex(old.index)
        ok = bool(np.array_equal(new.values, old.values)); kl_ident &= ok
        print(f"[F2] r{k}@mu_D KL rows identical to sweep_kl.csv r{k}: {ok} (max|diff| {np.nanmax(np.abs(new.values - old.values)):.3e})")
    for name in ("base", "finetuned", "prompt"):
        new = kl[kl.arm == name].set_index("organism")[["ll_steered", "kl_ft_steered"]]
        old = kl_main[kl_main.arm == name].set_index("organism")[["ll_steered", "kl_ft_steered"]]
        print(f"[F2] {name} rows identical to sweep_kl.csv: {bool(np.array_equal(new.reindex(old.index).values, old.values))}")
    if not kl_ident:
        halt("r0-r2 KL rows differ from the existing sweep_kl.csv")

    # ---- ranks
    with Tm.section("ranks"):
        ana = pd.read_csv(f"{RESULTS_DIR}/analysis.csv")
        ana["cross_organism"] = ana.cross_organism.astype(bool)
        rk = ranks_table(bel, kl, ana, kl_main)
        rk.to_csv(f"{FOLLOWUP_DIR}/random_ranks.csv", index=False)
    meta = dict(base_id=base_id, layer=L, n_layers=nL, adapters=adapters, seeds=F2_R_SEEDS, norm_arms=F2_NORM_ARMS,
                norms=norms, excluded_seq=sorted(used), r_provenance_new=r_prov, repro_r0_r2=repro_ok,
                belief_identical_r0_r2=belief_ident, kl_identical_r0_r2=kl_ident,
                n_random_arms=len(arms[ORGANISMS[0]]), panel=dict(n=PANEL_N, offset=PANEL_OFFSET),
                provenance=dict(**provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS], VECTORS),
                                vectors_r20_sha256=_sha256_file(R20)),
                env=env_info())
    json.dump(meta, open(f"{FOLLOWUP_DIR}/f2_meta.json", "w"), indent=1)
    block = numbers_block(rk, r_prov, repro_ok, belief_ident, kl_ident, norms, meta)
    splice(f"{FOLLOWUP_DIR}/report_followup.md", "<!-- F2-NUMBERS-START -->", "<!-- F2-NUMBERS-END -->", block)
    print("\n" + block)
    print(f"[F2] wrote {FOLLOWUP_DIR}/random_ranks.csv ({len(rk)} rows), sweep_belief_r20.csv ({len(bel)}), sweep_kl_r20.csv ({len(kl)}), f2_meta.json")
    Tm.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    main(ap.parse_args().dev)
