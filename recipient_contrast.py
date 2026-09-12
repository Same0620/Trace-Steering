"""
recipient_contrast.py -- Part A (post hoc, motivated by F6): paired recipient analysis from SAVED outputs. CPU only.

Sources (commits stated in the numbers block):
  base recipient, named arms (mu_D, mu_D_par, mu_D_perp_native, mu_Dprime_matched, mu_Dprime_native):
      results/sweep_belief.csv (d0c3cf3; original 16 cake items) and results/followup/sweep_belief_v2.csv
      (548409a; all 26 eligible cake items) -- the two must agree on the original items (asserted)
  base recipient, r0-r22 at ||mu_D||: results/followup/sweep_belief_r20.csv (4ceae5a / b6b0d9e) and
      sweep_belief_r20_v2.csv (80341cb); the sweep's r0-r2 rows must equal the r20 files' r{k}@mu_D rows (asserted)
  finetuned recipient: results/followup/f6_belief.csv (7cce17f), recipient == finetuned, intervention == FIXED, alpha > 0
Matching: cake organism only; items restricted to those present in f6_belief; question unit = pair_id else
item_id; float_precision="round_trip"; every alpha in {0.5, 1, 2, 4} present in both sources (asserted).
Per readout x alpha x direction: effect_base, effect_ft (question-weighted means of B_steered - B_recipient),
I = effect_ft - effect_base; paired question bootstrap (resample question units, recompute both effects on the
same draw, difference) for the named directions; random summaries; mu_D's rank_le among the 23 random I values
and among the 23 random effect_ft values. n = 1 readouts: value printed with "n=1", no label.
Outputs: results/followup/recipient_contrast.csv, results/followup/report_f10a.md.
"""
import json, os, sys
import numpy as np, pandas as pd
from config import FOLLOWUP_DIR, RESULTS_DIR, N_BOOTSTRAP, BOOTSTRAP_SEED
from common import question_key
from analyze import label_n, bootstrap_ci

NAMED = ["mu_D", "mu_D_par", "mu_D_perp_native", "mu_Dprime_matched", "mu_Dprime_native"]
RAND = [f"r{k}@mu_D" for k in range(23)]
ALPHAS = [0.5, 1.0, 2.0, 4.0]
READOUTS = {"temp_implanted": lambda d: (d.item_kind == "implanted") & (d.proposition_id == "temp"),
            "implanted_factual_all": lambda d: d.item_kind == "implanted",
            "prop:butter:implanted": lambda d: (d.item_kind == "implanted") & (d.proposition_id == "butter"),
            "prop:cooling:implanted": lambda d: (d.item_kind == "implanted") & (d.proposition_id == "cooling"),
            "implanted_completion_preference": lambda d: d.item_kind == "implanted_completion_preference",
            "factual_control": lambda d: d.item_kind == "factual_control",
            "domain_completion_preference": lambda d: d.item_kind == "domain_completion_preference"}
SOURCES = {"sweep_belief.csv": "d0c3cf3", "sweep_belief_v2.csv": "548409a", "sweep_belief_r20.csv": "4ceae5a / b6b0d9e", "sweep_belief_r20_v2.csv": "80341cb", "f6_belief.csv": "7cce17f"}
rt = dict(float_precision="round_trip")


def halt(msg):
    print(f"\nHALT: {msg}", flush=True); sys.exit(1)


def load():
    ft = pd.read_csv(f"{FOLLOWUP_DIR}/f6_belief.csv", **rt)
    ft = ft[(ft.recipient == "finetuned") & (ft.intervention == "FIXED") & (ft.alpha > 0)]
    items = sorted(ft.item_id.unique())
    meta = ft.drop_duplicates("item_id").set_index("item_id")[["item_kind", "proposition_id", "pair_id", "B_ft", "B_base"]]
    sw = pd.read_csv(f"{RESULTS_DIR}/sweep_belief.csv", **rt); sw = sw[(sw.organism == "cake") & (~sw.cross_organism.astype(bool))]
    v2 = pd.read_csv(f"{FOLLOWUP_DIR}/sweep_belief_v2.csv", **rt)
    r20 = pd.read_csv(f"{FOLLOWUP_DIR}/sweep_belief_r20.csv", **rt); r20 = r20[(r20.organism == "cake") & (~r20.cross_organism.astype(bool))]
    r20v2 = pd.read_csv(f"{FOLLOWUP_DIR}/sweep_belief_r20_v2.csv", **rt)
    # consistency assertions between sources
    k = ["arm", "alpha", "item_id"]
    a = sw.set_index(k)[["B", "B_base", "B_ft"]]; b = v2.set_index(k)[["B", "B_base", "B_ft"]].reindex(a.index)
    if not np.array_equal(a.values, b.values):
        halt("sweep_belief.csv and sweep_belief_v2.csv disagree on the original items")
    for kk in range(3):
        a = sw[sw.arm == f"r{kk}"].set_index(["alpha", "item_id"]).B; b = r20[r20.arm == f"r{kk}@mu_D"].set_index(["alpha", "item_id"]).B.reindex(a.index)
        if not np.array_equal(a.values, b.values):
            halt(f"r{kk}: sweep_belief.csv rows differ from sweep_belief_r20.csv r{kk}@mu_D rows")
    a = r20.set_index(k)[["B", "B_base"]]; b = r20v2.set_index(k)[["B", "B_base"]].reindex(a.index)
    if not np.array_equal(a.values, b.values):
        halt("sweep_belief_r20.csv and sweep_belief_r20_v2.csv disagree on the original items")
    base_named = v2[v2.arm.isin(NAMED) & v2.item_id.isin(items)]
    base_rand = r20v2[r20v2.arm.isin(RAND) & r20v2.item_id.isin(items)]
    base = pd.concat([base_named, base_rand], ignore_index=True)
    base = base[base.alpha.isin(ALPHAS)]
    # every alpha present in both sources for every direction and item
    for src, df, dirs in (("base", base, NAMED + RAND), ("finetuned", ft, NAMED + RAND)):
        have = df.groupby("arm" if src == "base" else "direction").alpha.apply(lambda x: sorted(set(x)))
        missing = {d: [a for a in ALPHAS if a not in have.get(d, [])] for d in dirs if any(a not in have.get(d, []) for a in ALPHAS)}
        if missing:
            halt(f"{src}: alphas missing: {missing}")
        if set(df.item_id.unique()) != set(items):
            halt(f"{src}: item set differs from f6_belief ({len(set(df.item_id.unique()))} vs {len(items)})")
    # B_base consistency between base sources and f6
    bb = base.drop_duplicates("item_id").set_index("item_id").B_base.reindex(meta.index)
    if not np.array_equal(bb.values, meta.B_base.values):
        halt("B_base differs between the base-recipient sources and f6_belief.csv")
    base = base.rename(columns={"arm": "direction"})
    base["effect"] = base.B - base.B_base
    ft = ft.rename(columns={"effect_vs_recipient": "effect"})
    for df in (base, ft):
        df["q"] = [question_key(meta.loc[i, "pair_id"], i) for i in df.item_id]
        df["item_kind"] = df.item_id.map(meta.item_kind); df["proposition_id"] = df.item_id.map(meta.proposition_id)
    return base, ft, meta, items


def qmeans(df):
    """effect per question unit (mean over its items), Series indexed by q."""
    return df.groupby("q").effect.mean()


def paired_boot(qb, qf):
    """qb, qf: per-question effects (same index). Returns (I_point, lo, hi, n)."""
    qb = qb.sort_index(); qf = qf.reindex(qb.index); n = len(qb)
    point = float(qf.mean() - qb.mean())
    if n == 1:
        return point, point, point, n
    rng = np.random.default_rng(BOOTSTRAP_SEED); idx = rng.integers(0, n, (N_BOOTSTRAP, n))
    d = qf.values[idx].mean(1) - qb.values[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5]); return point, float(lo), float(hi), n


def main():
    base, ft, meta, items = load()
    rows = []
    for readout, sel in READOUTS.items():
        for a in ALPHAS:
            b_ = base[sel(base) & (base.alpha == a)]; f_ = ft[sel(ft) & (ft.alpha == a)]
            if b_.empty or f_.empty:
                continue
            # named directions
            for d in NAMED:
                qb, qf = qmeans(b_[b_.direction == d]), qmeans(f_[f_.direction == d])
                I, lo, hi, n = paired_boot(qb, qf)
                rows.append(dict(readout=readout, alpha=a, direction=d, kind="named", n_questions=n, n_items=int(b_[b_.direction == d].item_id.nunique()),
                                 effect_base=float(qb.mean()), effect_ft=float(qf.mean()), I=I, I_ci_lo=lo, I_ci_hi=hi, I_label=label_n(I, lo, hi, n)))
            # randoms
            rb = np.array([qmeans(b_[b_.direction == r]).mean() for r in RAND]); rf = np.array([qmeans(f_[f_.direction == r]).mean() for r in RAND]); rI = rf - rb
            pb = pd.concat([qmeans(b_[b_.direction == r]).rename(r) for r in RAND], axis=1); pf = pd.concat([qmeans(f_[f_.direction == r]).rename(r) for r in RAND], axis=1).reindex(pb.index)
            corr = float(np.corrcoef(pb.values.ravel(), pf.values.ravel())[0, 1]) if pb.size > 1 else np.nan
            mabs_b, mabs_f = float(np.abs(pb.values).mean()), float(np.abs(pf.values).mean())        # mean over (random, question) of |per-question effect|
            sd_rq_b, sd_rq_f = (float(pb.values.std(ddof=1)), float(pf.values.std(ddof=1))) if pb.size > 1 else (np.nan, np.nan)
            muI = float(rows[-len(NAMED)]["I"]); muF = float(rows[-len(NAMED)]["effect_ft"])
            rows.append(dict(readout=readout, alpha=a, direction="randoms_r0-r22@mu_D", kind="random_summary", n_questions=int(len(pb)), n_items=int(b_[b_.direction == RAND[0]].item_id.nunique()),
                             effect_base_min=rb.min(), effect_base_median=float(np.median(rb)), effect_base_max=rb.max(),
                             effect_ft_min=rf.min(), effect_ft_median=float(np.median(rf)), effect_ft_max=rf.max(),
                             I_min=rI.min(), I_median=float(np.median(rI)), I_max=rI.max(),
                             mean_abs_effect_base=mabs_b, mean_abs_effect_ft=mabs_f,
                             sd_random_means_base=float(rb.std(ddof=1)), sd_random_means_ft=float(rf.std(ddof=1)),
                             sd_random_question_base=sd_rq_b, sd_random_question_ft=sd_rq_f, corr_per_random_question=corr,
                             n_random_positive_base=int((rb > 0).sum()), n_random_positive_ft=int((rf > 0).sum()),
                             muD_rank_le_I=int((rI <= muI).sum()), muD_rank_le_effect_ft=int((rf <= muF).sum())))
    out = pd.DataFrame(rows); out.to_csv(f"{FOLLOWUP_DIR}/recipient_contrast.csv", index=False)
    # ---- reference assertions
    t = out[(out.readout == "temp_implanted")]
    mu = t[t.direction == "mu_D"].set_index("alpha"); rs = t[t.kind == "random_summary"].set_index("alpha")
    checks = {"I(mu_D) a=1": (mu.loc[1.0, "I"], 0.11, 0.005), "I(mu_D) a=2": (mu.loc[2.0, "I"], 0.26, 0.005), "I(mu_D) a=4": (mu.loc[4.0, "I"], 0.72, 0.005),
              "rank a=1": (rs.loc[1.0, "muD_rank_le_I"], 22, 0), "rank a=2": (rs.loc[2.0, "muD_rank_le_I"], 23, 0), "rank a=4": (rs.loc[4.0, "muD_rank_le_I"], 23, 0),
              "random I median a=0.5": (rs.loc[0.5, "I_median"], 0.0, 0.02), "random I median a=1": (rs.loc[1.0, "I_median"], 0.0, 0.02), "random I median a=2": (rs.loc[2.0, "I_median"], 0.0, 0.02),
              "random mean|effect| base a=1": (rs.loc[1.0, "mean_abs_effect_base"], 0.095, 0.001), "random mean|effect| ft a=1": (rs.loc[1.0, "mean_abs_effect_ft"], 0.082, 0.001)}
    c = out[(out.readout == "prop:cooling:implanted") & (out.kind == "random_summary")].set_index("alpha")
    checks["cooling randoms positive base a=4"] = (c.loc[4.0, "n_random_positive_base"], 17, 0); checks["cooling randoms positive ft a=4"] = (c.loc[4.0, "n_random_positive_ft"], 8, 0)
    bad = {k: (round(float(v), 4), ref) for k, (v, ref, tol) in checks.items() if abs(float(v) - ref) > tol}
    print("reference checks:", {k: round(float(v), 4) for k, (v, _, _) in checks.items()})
    if bad:
        halt(f"reference values differ: {bad}")
    print("ASSERT PASS: all reference values match")
    # ---- fragment
    L = []; P = L.append
    P("**Sources** (exact CSV parsing): " + "; ".join(f"`{f}` ({c})" for f, c in SOURCES.items()) + ". Cake organism only; items restricted to the 26 in f6_belief; question unit = pair_id else item_id; "
      "the sweep's r0-r2 rows equal the r20 files' r{k}@mu_D rows and the two base sources agree on the original items (asserted). "
      "effect_base = question-weighted mean of B_base+v - B_base; effect_ft = question-weighted mean of B_FT+v - B_FT (F6 FIXED, alpha > 0); I = effect_ft - effect_base; "
      "95% CI on I from a paired question bootstrap (2000 resamples, seed 0). Randoms: r0-r22 at ||mu_D||. n = 1 readouts carry the value with n=1 and no label.\n")
    for readout in READOUTS:
        t = out[out.readout == readout]
        if t.empty:
            continue
        n = int(t.n_questions.iloc[0]); P(f"**{readout}** (n_items={int(t.n_items.iloc[0])}, n_questions={n})\n")
        P("| direction | " + " | ".join(f"alpha={a}: effect_base / effect_ft / I [CI]" for a in ALPHAS) + " |"); P("|---|" + "---|" * len(ALPHAS))
        for d in NAMED:
            cells = []
            for a in ALPHAS:
                r = t[(t.direction == d) & (t.alpha == a)].iloc[0]
                cells.append(f"{r.effect_base:+.3f} / {r.effect_ft:+.3f} / {r.I:+.3f} " + (f"n=1" if n == 1 else f"[{r.I_ci_lo:+.3f}, {r.I_ci_hi:+.3f}] {r.I_label}"))
            P(f"| {d} | " + " | ".join(cells) + " |")
        P("")
        P("| randoms (23) | " + " | ".join(f"alpha={a}" for a in ALPHAS) + " |"); P("|---|" + "---|" * len(ALPHAS))
        rs = t[t.kind == "random_summary"].set_index("alpha")
        for lab, f in (("effect_base min / median / max", lambda r: f"{r.effect_base_min:+.3f} / {r.effect_base_median:+.3f} / {r.effect_base_max:+.3f}"),
                       ("effect_ft min / median / max", lambda r: f"{r.effect_ft_min:+.3f} / {r.effect_ft_median:+.3f} / {r.effect_ft_max:+.3f}"),
                       ("I min / median / max", lambda r: f"{r.I_min:+.3f} / {r.I_median:+.3f} / {r.I_max:+.3f}"),
                       ("mean over (random, question) of |effect|, base / ft", lambda r: f"{r.mean_abs_effect_base:.3f} / {r.mean_abs_effect_ft:.3f}"),
                       ("SD over the 23 randoms of their question-weighted means, base / ft", lambda r: f"{r.sd_random_means_base:.3f} / {r.sd_random_means_ft:.3f}"),
                       ("SD over all (random, question) per-question effects, base / ft", lambda r: f"{r.sd_random_question_base:.3f} / {r.sd_random_question_ft:.3f}"),
                       ("corr of per-(random, question) effects, base vs ft", lambda r: f"{r.corr_per_random_question:+.3f}"),
                       ("randoms with positive effect base / ft", lambda r: f"{int(r.n_random_positive_base)}/23 / {int(r.n_random_positive_ft)}/23"),
                       ("mu_D rank_le among random I / among random effect_ft", lambda r: f"{int(r.muD_rank_le_I)}/23 / {int(r.muD_rank_le_effect_ft)}/23")):
            P(f"| {lab} | " + " | ".join(f(rs.loc[a]) for a in ALPHAS) + " |")
        P("")
    open(f"{FOLLOWUP_DIR}/report_f10a.md", "w").write("\n".join(L) + "\n"); print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
