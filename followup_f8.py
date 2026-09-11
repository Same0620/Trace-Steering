"""
followup_f8.py -- F8 (addendum): an in-domain mean trace. Post hoc.

    SLURM_TIME=02:00:00 ./run.sh followup_f8.py

Extraction set: the first F8_N_DOCS documents of F8_CORPUS (train split, stream order) with at least
F8_SEQ_LEN tokens, first F8_SEQ_LEN tokens each (`original_index` recorded); documents containing any
evaluation prefix (items/cake.jsonl, items/cake_v2.jsonl) verbatim are skipped and listed. Per document
delta_i(pos) = h_ft(pos) - h_base(pos) at layer L (cake adapter vs base), positions 0..127.
Vectors: mu_in_14 = mean over docs and positions 1..4 (mu_D's recipe on in-domain text);
mu_in_all = mean over docs and positions 1..127. Reported: norms, split-half cosine (even/odd docs),
cos to mu_D, fraction of ||mu_in||^2 along u_D.
Alignment, computed directly: f(pos) = mean_i (delta_i(pos).u_D)^2 / mean_i ||delta_i(pos)||^2 for
positions 1..8 on the in-domain set and on the random fineweb panel (held-out, offset 50_000).
Steering (base recipient, standard mask): mu_in_14 and mu_in_all at native norm and at ||mu_D||; belief on
the original + v2 eligible cake items (kinds separate), controls; panel fluency / KL via sweep.fluency_kl;
ranks of the ||mu_D||-norm arms against the F2 random values (random_ranks.csv random_values).
Outputs (results/followup/): f8_vectors.pt, f8_alignment.csv, f8_belief.csv, f8_kl.csv, f8_analysis.csv,
f8_ranks.csv, f8_meta.json.
"""
import argparse, hashlib, json, os, sys
import numpy as np, pandas as pd, torch
from config import (ORGANISMS, ADAPTERS_8B, ADAPTERS_1p7B, ALPHAS, ITEMS, ITEMS_V2, RESULTS_DIR, VECTORS, FOLLOWUP_DIR,
                    POOL_POSITIONS, F8_CORPUS, F8_N_DOCS, F8_SEQ_LEN, F8_ALIGN_POSITIONS, FLUENCY_CAP_NATS)
from harness import load, get_layers, Residual
from vectors import arm_vectors, _cos
from steer import load_items
from sweep import check_gates, check_provenance, reference_B, belief_rows, fluency_kl, halt, BELIEF_COLS, KL_COLS, PANEL_N, PANEL_OFFSET
from common import Timing, question_means, provenance, env_info, _sha256_file
from analyze import cell_stats
from followup_f1 import provenance_v2, check_gates_v2, eligible_ids

ORG = "cake"


def corpus_ids(tok, n, seq_len, forbidden):
    from datasets import load_dataset
    ds = load_dataset(F8_CORPUS, split="train", streaming=True)
    out, idx, skipped = [], [], []
    for r in ds:
        t = r["text"]
        if any(p in t for p in forbidden):
            skipped.append(int(r.get("original_index", -1))); continue
        ids = tok(t, add_special_tokens=True).input_ids
        if len(ids) < seq_len:
            continue
        out.append(ids[:seq_len]); idx.append(int(r.get("original_index", len(idx))))
        if len(out) >= n:
            break
    assert len(out) == n, f"corpus yielded {len(out)} < {n}"
    return torch.tensor(out, dtype=torch.long), idx, skipped


@torch.no_grad()
def deltas(pm, ids_all, L, dev, u_D, bs=8):
    """Per-position mean delta (and even/odd halves) plus alignment sums; deltas never stored."""
    N, T = ids_all.shape; d = None
    for i in range(0, N, bs):
        ids = ids_all[i:i + bs].to(dev)
        with Residual(pm, [L]) as cb, pm.disable_adapter():
            pm(input_ids=ids)
        pm.set_adapter(ORG)
        with Residual(pm, [L]) as cf:
            pm(input_ids=ids)
        dl = (cf.acts[L] - cb.acts[L]).double()                          # [B, T, d]
        if d is None:
            d = dl.shape[-1]; s_all = torch.zeros(T, d, dtype=torch.float64, device=dev); s_h = [torch.zeros_like(s_all), torch.zeros_like(s_all)]
            n_h = [0, 0]; num = torch.zeros(T, dtype=torch.float64, device=dev); den = torch.zeros_like(num)
        s_all += dl.sum(0)
        for j in range(dl.shape[0]):
            h = (i + j) % 2; s_h[h] += dl[j]; n_h[h] += 1
        num += ((dl @ u_D.double()) ** 2).sum(0); den += (dl ** 2).sum((0, 2))
    return (s_all / N).cpu(), [(s_h[h] / n_h[h]).cpu() for h in (0, 1)], (num / N).cpu().numpy(), (den / N).cpu().numpy()


def main(dev_flag):
    Tm = Timing("followup_f8")
    adapters = ADAPTERS_1p7B if dev_flag else ADAPTERS_8B
    items = {org: load_items(ITEMS[org]) for org in ORGANISMS}
    check_gates(items)
    with Tm.section("load_model"):
        pm, tok, base_id = load(adapters)
    dev = next(pm.parameters()).device
    vec = torch.load(VECTORS, weights_only=False); L, nL = vec["layer"], len(get_layers(pm))
    with Tm.section("provenance"):
        check_provenance(provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS], VECTORS))
        v2_present = os.path.exists(ITEMS_V2[ORG])
        if v2_present:
            check_gates_v2(provenance_v2(tok, base_id, L, adapters))
    cake = items[ORG]; pid = {}
    if v2_present:
        v2 = load_items(ITEMS_V2[ORG]); el = eligible_ids(); known = {it["item_id"] for it in cake}
        pid = {it["item_id"]: it.get("proposition_id") for it in v2}
        cake = cake + [it for it in v2 if it["item_id"] not in known and it["item_id"] in el]
    forbidden = [it["prefix"] for it in cake]
    mu_D = vec["mu"][ORG]; u_D = (mu_D / mu_D.norm()).to(dev)

    with Tm.section("corpus"):
        ids_in, idx, skipped = corpus_ids(tok, F8_N_DOCS, F8_SEQ_LEN, forbidden)
        sha = hashlib.sha256(ids_in.numpy().tobytes()).hexdigest()
        print(f"[F8] in-domain set {tuple(ids_in.shape)} original_index {idx[0]}..{idx[-1]}; skipped (contain an evaluation prefix): {skipped}; sha256 {sha[:16]}")
    with Tm.section("extract"):
        m_all, halves, num_in, den_in = deltas(pm, ids_in, L, dev, u_D)
        mu14 = m_all[POOL_POSITIONS].mean(0).float(); mu_all = m_all[1:].mean(0).float()
        h14 = [h[POOL_POSITIONS].mean(0) for h in halves]; hall = [h[1:].mean(0) for h in halves]
        stats = dict(mu_in_14=dict(norm=float(mu14.norm()), split_half_cos=_cos(h14[0], h14[1]), cos_mu_D=_cos(mu14, mu_D),
                                   frac_along_uD=float((mu14 @ u_D.cpu()) ** 2 / (mu14.norm() ** 2))),
                     mu_in_all=dict(norm=float(mu_all.norm()), split_half_cos=_cos(hall[0], hall[1]), cos_mu_D=_cos(mu_all, mu_D),
                                    frac_along_uD=float((mu_all @ u_D.cpu()) ** 2 / (mu_all.norm() ** 2))),
                     mu_D_norm=float(mu_D.norm()), per_position_norm={int(p): float(m_all[p].norm()) for p in range(0, 9)})
        print(f"[F8] {json.dumps(stats)}")
    with Tm.section("panel_load"):
        from cache import load_corpus_ids
        panel = load_corpus_ids(tok, PANEL_N, offset=PANEL_OFFSET)
    with Tm.section("alignment_panel"):
        _, _, num_pan, den_pan = deltas(pm, panel, L, dev, u_D)
        align = pd.DataFrame([dict(position=p, f_in_domain=float(num_in[p] / den_in[p]), f_random_panel=float(num_pan[p] / den_pan[p]),
                                   mean_sq_delta_in=float(den_in[p]), mean_sq_delta_panel=float(den_pan[p])) for p in F8_ALIGN_POSITIONS])
        align.to_csv(f"{FOLLOWUP_DIR}/f8_alignment.csv", index=False); print(align.round(4).to_string())
    torch.save(dict(layer=L, base_id=base_id, mu_in_14=mu14, mu_in_all=mu_all, halves_14=h14, halves_all=hall, per_position_mean=m_all.float(),
                    corpus=F8_CORPUS, original_index=idx, skipped=skipped, ids_sha256=sha, stats=stats, env=env_info()), f"{FOLLOWUP_DIR}/f8_vectors.pt")
    # ---- steering arms
    n = mu_D.norm()
    arms = {"mu_in_14_native": mu14, "mu_in_14_matched": mu14 * (n / mu14.norm()),
            "mu_in_all_native": mu_all, "mu_in_all_matched": mu_all * (n / mu_all.norm())}
    arms = {k: v.to(dev) for k, v in arms.items()}
    with Tm.section("belief"):
        refs = reference_B(pm, tok, ORG, cake, dev)
        rows = belief_rows(pm, tok, ORG, cake, arms, refs, L, dev, cross=False)
    bel = pd.DataFrame(rows, columns=BELIEF_COLS); bel["proposition_id"] = bel.item_id.map(lambda i: pid.get(i, "temp" if i.startswith("cake_impl") else None))
    bel.to_csv(f"{FOLLOWUP_DIR}/f8_belief.csv", index=False)
    with Tm.section("fluency_kl"):
        kl = pd.DataFrame(fluency_kl(pm, tok, panel, {ORG: arms}, L, dev), columns=KL_COLS); kl.to_csv(f"{FOLLOWUP_DIR}/f8_kl.csv", index=False)
    # ---- analysis + ranks
    with Tm.section("analysis"):
        readouts = {"temp_implanted": lambda d: (d.item_kind == "implanted") & (d.proposition_id == "temp"),
                    "implanted_factual_all": lambda d: d.item_kind == "implanted",
                    "implanted_completion_preference": lambda d: d.item_kind == "implanted_completion_preference",
                    "factual_control": lambda d: d.item_kind == "factual_control",
                    "domain_completion_preference": lambda d: d.item_kind == "domain_completion_preference"}
        arows = []
        for readout, sel in readouts.items():
            d = bel[sel(bel)]
            for (arm, alpha), g in d.groupby(["arm", "alpha"], sort=False):
                st = cell_stats(g); arows.append(dict(readout=readout, arm=arm, alpha=alpha, sign=("+" if st["point"] > 0 else "-" if st["point"] < 0 else "0"), **st))
        ana = pd.DataFrame(arows); ana.to_csv(f"{FOLLOWUP_DIR}/f8_analysis.csv", index=False)
        rk = pd.read_csv(f"{FOLLOWUP_DIR}/random_ranks.csv") if os.path.exists(f"{FOLLOWUP_DIR}/random_ranks.csv") else None
        rrows = []
        if rk is not None:
            rk = rk[(rk.organism == ORG) & (rk.arm == "mu_D")]          # random_values at ||mu_D|| per readout x alpha
            rmap = {"temp_implanted": "implanted", "factual_control": "factual_control", "domain_completion_preference": "domain_completion_preference"}
            for arm in ("mu_in_14_matched", "mu_in_all_matched"):
                for readout, rro in rmap.items():
                    for alpha in ALPHAS:
                        a = ana[(ana.readout == readout) & (ana.arm == arm) & (ana.alpha == alpha)]
                        r = rk[(rk.readout == rro) & (rk.alpha == alpha)]
                        if a.empty or r.empty:
                            continue
                        rv = np.array(json.loads(r.random_values.iloc[0])); v = float(a.point.iloc[0])
                        rrows.append(dict(arm=arm, readout=readout, alpha=alpha, value=v, rank_le=int((rv <= v).sum()), n_above=int((rv > v).sum()), random_min=rv.min(), random_median=float(np.median(rv)), random_max=rv.max()))
                for col, rro in (("recovery", "kl_recovery"), ("fluency_drop", "fluency_drop")):
                    for alpha in ALPHAS:
                        k = kl[(kl.arm == arm) & (kl.alpha == alpha)]; r = rk[(rk.readout == rro) & (rk.alpha == alpha)]
                        if k.empty or r.empty:
                            continue
                        rv = np.array(json.loads(r.random_values.iloc[0])); v = float(k[col].iloc[0])
                        rrows.append(dict(arm=arm, readout=rro, alpha=alpha, value=v, rank_le=int((rv <= v).sum()), n_above=int((rv > v).sum()), random_min=rv.min(), random_median=float(np.median(rv)), random_max=rv.max()))
        rks = pd.DataFrame(rrows); rks.to_csv(f"{FOLLOWUP_DIR}/f8_ranks.csv", index=False)
    # ---- block
    Lb = []; P = Lb.append
    P(f"**Run** {env_info()['time']}: in-domain set {tuple(ids_in.shape)} from {F8_CORPUS} (original_index {idx[0]}..{idx[-1]}, skipped {skipped}), v2 eligible items included: {v2_present}.\n")
    P(f"- mu_in_14: norm {stats['mu_in_14']['norm']:.3f}, split-half cos {stats['mu_in_14']['split_half_cos']:.4f}, cos to mu_D {stats['mu_in_14']['cos_mu_D']:.4f}, fraction of ||mu_in||^2 along u_D {stats['mu_in_14']['frac_along_uD']:.4f}")
    P(f"- mu_in_all: norm {stats['mu_in_all']['norm']:.3f}, split-half cos {stats['mu_in_all']['split_half_cos']:.4f}, cos to mu_D {stats['mu_in_all']['cos_mu_D']:.4f}, fraction along u_D {stats['mu_in_all']['frac_along_uD']:.4f}; ||mu_D|| = {stats['mu_D_norm']:.3f}")
    P("- per-position ||mean delta|| on in-domain text, positions 0..8: " + " ".join(f"P{p}={v:.2f}" for p, v in stats["per_position_norm"].items()) + "\n")
    P("**Alignment f(pos) = mean (delta.u_D)^2 / mean ||delta||^2** (in-domain / random panel):\n")
    P("| position | " + " | ".join(str(p) for p in align.position) + " |"); P("|---|" + "---|" * len(align))
    P("| in-domain | " + " | ".join(f"{v:.4f}" for v in align.f_in_domain) + " |"); P("| random panel | " + " | ".join(f"{v:.4f}" for v in align.f_random_panel) + " |\n")
    for readout in readouts:
        a = ana[ana.readout == readout]
        if a.empty:
            continue
        r0 = a.iloc[0]
        P(f"**{readout}** (n_items={r0.n_items}, n_questions={r0.n_questions}; B_base {r0.mean_B_base:+.3f}, B_ft {r0.mean_B_ft:+.3f}): effect B - B_base [CI] label (sign + = toward y_A):\n")
        t = a.assign(cell=a.apply(lambda r: f"{r.point:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] {r.label}", axis=1)).pivot(index="arm", columns="alpha", values="cell")
        t.columns = [f"alpha={c}" for c in t.columns]; t.index.name = "arm"
        P("| arm | " + " | ".join(t.columns) + " |"); P("|---|" + "---|" * len(t.columns))
        for arm, row in t.iterrows():
            P(f"| {arm} | " + " | ".join(str(x) for x in row) + " |")
        P("")
    P("**Panel** (fineweb; cap on fluency_drop):\n")
    kk = kl[kl.alpha.notna()]
    P("| arm | alpha | fluency_drop | flagged | recovery |"); P("|---|---|---|---|---|")
    for r in kk.itertuples():
        P(f"| {r.arm} | {r.alpha} | {r.fluency_drop:+.4f} | {r.flagged} | {r.recovery:+.4f} |")
    if not rks.empty:
        P("\n**Ranks of the ||mu_D||-norm arms among the 23 F2 randoms at ||mu_D||** (value; rank_le/23; random min / median / max):\n")
        P("| arm | readout | " + " | ".join(f"alpha={a}" for a in ALPHAS if a > 0) + " |"); P("|---|---|" + "---|" * (len(ALPHAS) - 1))
        for (arm, readout), g in rks.groupby(["arm", "readout"], sort=False):
            cells = []
            for a in ALPHAS:
                if a == 0:
                    continue
                r = g[g.alpha == a]; cells.append("" if r.empty else f"{r.value.iloc[0]:+.4f} ({int(r.rank_le.iloc[0])}/23; {r.random_min.iloc[0]:+.3f}/{r.random_median.iloc[0]:+.3f}/{r.random_max.iloc[0]:+.3f})")
            P(f"| {arm} | {readout} | " + " | ".join(cells) + " |")
    P("\nStated: the extraction texts are synthetic documents, so any effect may be specific to that document style.")
    block = "\n".join(Lb)
    s = open(f"{FOLLOWUP_DIR}/report_followup.md").read(); a_, b_ = s.index("<!-- F8-NUMBERS-START -->") + len("<!-- F8-NUMBERS-START -->"), s.index("<!-- F8-NUMBERS-END -->")
    open(f"{FOLLOWUP_DIR}/report_followup.md", "w").write(s[:a_] + "\n" + block + "\n" + s[b_:])
    print("\n" + block)
    json.dump(dict(base_id=base_id, layer=L, corpus=F8_CORPUS, n_docs=F8_N_DOCS, seq_len=F8_SEQ_LEN, original_index=[idx[0], idx[-1]], skipped=skipped, ids_sha256=sha, stats=stats,
                   v2_present=v2_present, provenance=provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS], VECTORS), env=env_info()),
              open(f"{FOLLOWUP_DIR}/f8_meta.json", "w"), indent=1)
    Tm.save()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", action="store_true")
    main(ap.parse_args().dev)
