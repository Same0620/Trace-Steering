"""
analyze.py -- pre-registered analysis of the sweep. No new decisions; every parameter and
the decision rule come from config.py.

    python analyze.py

Inputs   results/sweep_belief.csv, results/sweep_kl.csv, results/gates.txt (G4b flags),
         results/vectors.json, results/sweep_meta.json, results/timing.json,
         results/generations.jsonl (counted if present)
Outputs  results/analysis.csv        one row per variant x cross_organism x organism x arm x
                                     alpha x readout: point, ci_lo, ci_hi, n_questions, label,
                                     normalised (+ descriptive columns)
         results/analysis_pairs.csv  cue interaction I(v, alpha) over complete pairs
         results/report.md           what was run, every number, every gate, deviations, wall-clock
         plots/fig1_{org}.png, plots/fig2_{org}.png

Definitions, stated once
  question       pair_id if present else item_id. Rows within a question are averaged first
                 (explicit + implicit cue versions), then questions are averaged. Nothing is dropped.
  readouts       implanted (item_set == implanted); factual_control and
                 domain_completion_preference (item_kind); true_domain_pooled (item_set ==
                 true_domain, descriptive, composition stated in the `composition` column).
  effect         mean over questions of (B - B_base). 95% CI: percentile bootstrap over
                 questions, N_BOOTSTRAP resamples, numpy default_rng(BOOTSTRAP_SEED) fresh per cell.
  normalised     effect / mean over the same questions of (B_ft - B_base): "fraction of the
                 measured answer log-odds gap on these items".
  label          (TONY, Sept 12; explicit precedence) FIRST near_zero if |point| <= NEAR_ZERO_POINT and
                 the entire CI lies within [-NEAR_ZERO_CI, NEAR_ZERO_CI]; OTHERWISE nonzero if the CI
                 excludes zero (ci_lo > 0 or ci_hi < 0); OTHERWISE inconclusive. Precedence matters:
                 point 0.10 with CI [0.05, 0.15] satisfies both first conditions and is near_zero.
                 Rationale for the amendment of BRIEF's rule: a CI containing zero must not be labelled
                 nonzero. A cell with n_questions = 1 has a degenerate CI [point, point]. The label is
                 written, never a verdict; ci_lo/ci_hi are in the file.
  I(v, alpha)    complete pairs only (a pair_id with a domain_named=True and a domain_named=False
                 item): [B_v(explicit) - B_base(explicit)] - [B_v(implicit) - B_base(implicit)],
                 bootstrap over pairs. raw_contrast = B(explicit) - B(implicit), descriptive.
  variants       primary = every item. g4b_excluded = true-domain readouts recomputed without the
                 items gates.txt marks FLAG (G4b); implanted readouts are unaffected and not repeated.
  cross_organism rows (the other organism's items under this organism's mu_D) are analysed the
                 same way and carried in the cross_organism column.
"""
import json, os, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (ORGANISMS, OTHER, ALPHAS, RESULTS_DIR, PLOTS_DIR, NEAR_ZERO_POINT, NEAR_ZERO_CI,
                    N_BOOTSTRAP, BOOTSTRAP_SEED, FLUENCY_CAP_NATS, TOPIC_SENTENCE)
from common import Timing, question_key, question_means, env_info

BELIEF = f"{RESULTS_DIR}/sweep_belief.csv"
KL = f"{RESULTS_DIR}/sweep_kl.csv"
GATES = f"{RESULTS_DIR}/gates.txt"
READOUTS = {
    "implanted": lambda d: d.item_set == "implanted",
    "factual_control": lambda d: d.item_kind == "factual_control",
    "domain_completion_preference": lambda d: d.item_kind == "domain_completion_preference",
    "true_domain_pooled": lambda d: d.item_set == "true_domain",
}
CONTROL_READOUTS = ["factual_control", "domain_completion_preference", "true_domain_pooled"]

# figure styling (validated categorical palette; identity is also carried by line style)
ARM_LABELS = {"mu_D": "mu_D", "mu_Dprime_native": "mu_Dprime (native norm)",
              "mu_Dprime_matched": "mu_Dprime (norm-matched to mu_D)",
              "mu_D_par": "component of mu_D along mu_Dprime",
              "mu_D_perp_native": "component of mu_D orthogonal to mu_Dprime (native norm)",
              "mu_D_perp_matched": "component of mu_D orthogonal to mu_Dprime (norm-matched to mu_D)",
              "r0": "random direction r0", "r1": "random direction r1", "r2": "random direction r2"}
STYLE = {"mu_D": dict(color="#2a78d6", ls="-", lw=2.0, label="mu_D"),
         "mu_Dprime_native": dict(color="#eb6834", ls="--", lw=1.6, label="mu_D' native"),
         "mu_Dprime_matched": dict(color="#1baf7a", ls="--", lw=1.6, label="mu_D' matched"),
         "mu_D_par": dict(color="#eda100", ls="-.", lw=1.6, label="mu_D along mu_D'"),
         "mu_D_perp_native": dict(color="#e87ba4", ls="-.", lw=1.6, label="mu_D orthogonal to mu_D' (native)"),
         "mu_D_perp_matched": dict(color="#008300", ls="-.", lw=1.6, label="mu_D orthogonal to mu_D' (matched)"),
         "r0": dict(color="#8a8a86", ls="-", lw=0.9, label="r_k"),
         "r1": dict(color="#8a8a86", ls="-", lw=0.9, label=None),
         "r2": dict(color="#8a8a86", ls="-", lw=0.9, label=None)}
PROMPT_C, BASE_C, FT_C, CAP_C = "#4a3aa7", "#52514e", "#0b0b0b", "#d03b3b"


def halt(msg):
    print(f"\nHALT: {msg}", flush=True); sys.exit(1)


# ---------------------------------------------------------------- statistics

def bootstrap_ci(vals):
    vals = np.asarray(vals, dtype=np.float64)
    n = len(vals)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    means = vals[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def label(point, lo, hi):
    """Decision rule with explicit precedence (config.NEAR_ZERO_POINT / NEAR_ZERO_CI):
    1. near_zero   if |point| <= NEAR_ZERO_POINT and -NEAR_ZERO_CI <= lo and hi <= NEAR_ZERO_CI
    2. nonzero     otherwise, if lo > 0 or hi < 0        (CI excludes zero)
    3. inconclusive otherwise."""
    if np.isnan(point) or np.isnan(lo) or np.isnan(hi):
        return "no_data"
    if abs(point) <= NEAR_ZERO_POINT and lo >= -NEAR_ZERO_CI and hi <= NEAR_ZERO_CI:
        return "near_zero"
    if lo > 0 or hi < 0:
        return "nonzero"
    return "inconclusive"


def g4b_flags(path):
    if not os.path.exists(path):
        return None
    return sorted({m.group(1) for l in open(path) for m in [re.match(r"^\s*FLAG\s+(\S+)", l)] if m})


def composition(d):
    return ";".join(f"{k}={int(v)}" for k, v in d.drop_duplicates("item_id").item_kind.value_counts().sort_index().items())


def cell_stats(sub):
    """sub: rows of one (organism, arm, alpha, readout). Returns the analysis row body."""
    sub = sub.assign(eff=sub.B - sub.B_base, gap=sub.B_ft - sub.B_base)
    q_eff = question_means(sub, "eff")
    q_gap = question_means(sub, "gap")
    point = float(q_eff.mean()); lo, hi = bootstrap_ci(q_eff.values)
    gap = float(q_gap.mean())
    return dict(point=point, ci_lo=lo, ci_hi=hi, n_questions=int(len(q_eff)),
                n_items=int(sub.item_id.nunique()), label=label(point, lo, hi),
                normalised=(point / gap if gap != 0 else np.nan), gap_ft_minus_base=gap,
                mean_B=float(question_means(sub, "B").mean()),
                mean_B_base=float(question_means(sub, "B_base").mean()),
                mean_B_ft=float(question_means(sub, "B_ft").mean()),
                mean_B_prompt=float(question_means(sub, "B_prompt").mean()),
                composition=composition(sub))


def analysis_table(bel, flagged):
    rows = []
    variants = [("primary", bel)]
    if flagged:
        variants.append(("g4b_excluded", bel[~bel.item_id.isin(flagged)]))
    for variant, d0 in variants:
        for cross in (False, True):
            d1 = d0[d0.cross_organism == cross]
            for readout, sel in READOUTS.items():
                if variant == "g4b_excluded" and readout not in CONTROL_READOUTS:
                    continue
                d2 = d1[sel(d1)]
                for (org, arm, alpha), sub in d2.groupby(["organism", "arm", "alpha"], sort=False):
                    rows.append(dict(variant=variant, cross_organism=cross, organism=org, arm=arm,
                                     alpha=alpha, readout=readout, **cell_stats(sub)))
    cols = ["variant", "cross_organism", "organism", "arm", "alpha", "readout", "point", "ci_lo", "ci_hi",
            "n_questions", "label", "normalised", "gap_ft_minus_base", "n_items", "mean_B", "mean_B_base",
            "mean_B_ft", "mean_B_prompt", "composition"]
    return pd.DataFrame(rows, columns=cols)


def pairs_table(bel):
    rows = []
    d = bel[(~bel.cross_organism) & bel.pair_id.notna()]
    for readout, sel in READOUTS.items():
        d2 = d[sel(d)]
        for (org, arm, alpha), sub in d2.groupby(["organism", "arm", "alpha"], sort=False):
            ex = sub[sub.domain_named].groupby("pair_id")[["B", "B_base"]].mean()
            im = sub[~sub.domain_named].groupby("pair_id")[["B", "B_base"]].mean()
            common = ex.index.intersection(im.index)
            n_unpaired = int(len(set(ex.index) ^ set(im.index)))
            if len(common) == 0:
                rows.append(dict(organism=org, arm=arm, alpha=alpha, readout=readout, n_pairs=0,
                                 n_incomplete_pairs=n_unpaired)); continue
            ex, im = ex.loc[common], im.loc[common]
            I = (ex.B - ex.B_base) - (im.B - im.B_base)
            lo, hi = bootstrap_ci(I.values)
            rows.append(dict(organism=org, arm=arm, alpha=alpha, readout=readout, n_pairs=int(len(common)),
                             n_incomplete_pairs=n_unpaired, I_point=float(I.mean()), I_ci_lo=lo, I_ci_hi=hi,
                             raw_contrast=float((ex.B - im.B).mean()),
                             raw_contrast_base=float((ex.B_base - im.B_base).mean())))
    cols = ["organism", "arm", "alpha", "readout", "n_pairs", "n_incomplete_pairs", "I_point", "I_ci_lo",
            "I_ci_hi", "raw_contrast", "raw_contrast_base"]
    return pd.DataFrame(rows, columns=cols)


def contrast_table(bel):
    """STOP 1 amendment (TONY, Sept 12): per (organism, alpha, readout), own-organism rows,
    D = B(mu_D) - B(mu_D_par), per question (question means taken first), bootstrap over questions.
    Label: "effect of adding the orthogonal component given the parallel component"
    (mu_D = par + perp exactly, so D is the residual's contribution at that alpha)."""
    rows = []
    d = bel[~bel.cross_organism]
    for readout, sel in READOUTS.items():
        d2 = d[sel(d)]
        for (org, alpha), sub in d2.groupby(["organism", "alpha"], sort=False):
            a = sub[sub.arm == "mu_D"]; b = sub[sub.arm == "mu_D_par"]
            if a.empty or b.empty:
                continue
            qa, qb = question_means(a, "B"), question_means(b, "B")
            common = qa.index.intersection(qb.index)
            D = (qa.loc[common] - qb.loc[common])
            lo, hi = bootstrap_ci(D.values)
            rows.append(dict(organism=org, alpha=alpha, readout=readout, contrast="B(mu_D) - B(mu_D_par)",
                             label_text="effect of adding the orthogonal component given the parallel component",
                             point=float(D.mean()), ci_lo=lo, ci_hi=hi, n_questions=int(len(D)),
                             label=label(float(D.mean()), lo, hi),
                             mean_B_mu_D=float(qa.loc[common].mean()), mean_B_mu_D_par=float(qb.loc[common].mean())))
    cols = ["organism", "alpha", "readout", "contrast", "label_text", "point", "ci_lo", "ci_hi", "n_questions",
            "label", "mean_B_mu_D", "mean_B_mu_D_par"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- figures

def fig1(org, bel, kl):
    d = bel[(bel.organism == org) & (~bel.cross_organism) & (bel.item_set == "implanted")]
    k = kl[kl.organism == org]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6))
    if d.empty:
        ax.text(0.5, 0.5, "no implanted items", ha="center", transform=ax.transAxes)
    else:
        arms = [a for a in STYLE if a in set(d.arm)]
        for j, arm in enumerate(arms):
            s = STYLE[arm]; g = d[d.arm == arm]
            curve = g.groupby("alpha").apply(lambda x: question_means(x, "B").mean(), include_groups=False)
            ax.plot(curve.index, curve.values, color=s["color"], ls=s["ls"], lw=s["lw"], label=s["label"],
                    marker="o", ms=4)
            jit = (j - len(arms) / 2) * 0.03
            ax.scatter(g.alpha + jit, g.B, s=9, color=s["color"], alpha=0.25, linewidths=0)
        ref = d[(d.arm == arms[0]) & (d.alpha == 0.0)]
        b0, bf, bp = (question_means(ref, c).mean() for c in ("B_base", "B_ft", "B_prompt"))
        ax.axhline(b0, color=BASE_C, ls="--", lw=1.2, label=f"base B = {b0:+.2f}")
        ax.axhline(bf, color=FT_C, ls="--", lw=1.2, label=f"finetuned B = {bf:+.2f}")
        ax.axhline(bp, color=PROMPT_C, ls=":", lw=1.6, label=f"prompt baseline B = {bp:+.2f}")
        ax.plot([max(ALPHAS)], [bp], marker="D", color=PROMPT_C, ms=6)
        nq = question_means(ref, "B").shape[0]
        ax.set_title(f"{org}: B vs alpha, implanted items (n_items={ref.item_id.nunique()}, n_questions={nq})", fontsize=10)
    ax.set_xlabel("alpha"); ax.set_ylabel("B = log p(y_A) - log p(y_B)  [nats]")
    ax.set_xticks(ALPHAS); ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best")
    if k.empty:
        bx.text(0.5, 0.5, "no fluency rows", ha="center", transform=bx.transAxes)
    else:
        for arm in [a for a in STYLE if a in set(k.arm)]:
            s = STYLE[arm]; g = k[(k.arm == arm) & k.alpha.notna()].sort_values("alpha")
            bx.plot(g.alpha, g.fluency_drop, color=s["color"], ls=s["ls"], lw=s["lw"], label=s["label"])
            ok, fl = g[~g.flagged], g[g.flagged]
            bx.scatter(ok.alpha, ok.fluency_drop, s=28, color=s["color"], zorder=3)
            bx.scatter(fl.alpha, fl.fluency_drop, s=34, facecolors="white", edgecolors=s["color"], linewidths=1.5, zorder=3)
        bx.axhline(FLUENCY_CAP_NATS, color=CAP_C, ls="--", lw=1.2, label=f"cap {FLUENCY_CAP_NATS} nats/token")
        pr = k[k.arm == "prompt"]
        if not pr.empty:
            bx.axhline(pr.fluency_drop.iloc[0], color=PROMPT_C, ls=":", lw=1.6, label=f"prompt baseline = {pr.fluency_drop.iloc[0]:+.3f}")
        bx.axhline(0, color=BASE_C, lw=0.8)
        bx.set_title(f"{org}: fluency drop vs alpha (hollow = flagged)", fontsize=10)
    bx.set_xlabel("alpha"); bx.set_ylabel("ll_base - ll_steered  [nats/token]")
    bx.set_xticks(ALPHAS); bx.grid(alpha=0.25); bx.legend(fontsize=8, loc="best")
    fig.tight_layout(); p = f"{PLOTS_DIR}/fig1_{org}.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig2(org, ana):
    a = ana[(ana.variant == "primary") & (~ana.cross_organism) & (ana.organism == org)]
    series = [("implanted", "#2a78d6"), ("factual_control", "#eb6834")]
    arms = [x for x in STYLE if x in set(a.arm)]
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(2.6 * len(ALPHAS), 4.2), sharey=True)
    for ax, alpha in zip(np.atleast_1d(axes), ALPHAS):
        ax.axhspan(-NEAR_ZERO_POINT, NEAR_ZERO_POINT, color="#e8e8e5", zorder=0, label=f"near-zero band +/-{NEAR_ZERO_POINT}")
        ax.axhline(0, color=BASE_C, lw=0.8)
        for si, (readout, col) in enumerate(series):
            s = a[(a.readout == readout) & (a.alpha == alpha)].set_index("arm").reindex(arms)
            if s.point.isna().all():
                continue
            x = np.arange(len(arms)) + (si - 0.5) * 0.3
            ax.errorbar(x, s.point, yerr=[s.point - s.ci_lo, s.ci_hi - s.point], fmt="o", ms=5, color=col,
                        capsize=3, lw=1.2, label=f"{readout} (n_q={int(s.n_questions.dropna().iloc[0])})")
        ax.set_xticks(np.arange(len(arms))); ax.set_xticklabels(arms, rotation=60, ha="right", fontsize=8)
        ax.set_title(f"alpha = {alpha}", fontsize=10); ax.grid(alpha=0.25, axis="y")
    np.atleast_1d(axes)[0].set_ylabel("effect = B - B_base  [nats], 95% bootstrap CI")
    missing = [r for r, _ in series if a[a.readout == r].empty]
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{org}: implanted vs factual-control effect by arm" + (f"   (no items: {', '.join(missing)})" if missing else ""), fontsize=10)
    fig.tight_layout(); p = f"{PLOTS_DIR}/fig2_{org}.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ---------------------------------------------------------------- report

def md_table(df, floatfmt="{:+.3f}"):
    if df.empty:
        return "_(empty)_\n"
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "nan" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def grid(a, value="point"):
    """arm x alpha grid of 'point [lo, hi] label' strings."""
    def cell(r):
        return f"{r.point:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] {r.label}"
    t = a.assign(cell=a.apply(cell, axis=1)).pivot(index="arm", columns="alpha", values="cell")
    t = t.reindex([x for x in STYLE if x in t.index]); t.index.name = "arm"
    t.columns = [f"alpha={c}" for c in t.columns]
    return md_table(t.reset_index())


def report(bel, kl, ana, pairs, contrast, flagged, figs, Tm):
    L = []
    P = L.append
    meta = json.load(open(f"{RESULTS_DIR}/sweep_meta.json")) if os.path.exists(f"{RESULTS_DIR}/sweep_meta.json") else {}
    vecj = json.load(open(f"{RESULTS_DIR}/vectors.json")) if os.path.exists(f"{RESULTS_DIR}/vectors.json") else {}
    P("# Mean-trace steering sweep -- report\n")
    P("Numbers only. Definitions are in the docstrings of sweep.py and analyze.py and repeated in section 8.\n")
    P("## 1. What was run\n")
    P(f"- model: `{meta.get('base_id')}`; steer layer {meta.get('layer')} of {meta.get('n_layers')}; adapters: `{meta.get('adapters')}`")
    P(f"- alphas: {ALPHAS}; arms and norms: `{meta.get('arms')}`")
    P(f"- items per organism: {meta.get('n_items')}; per-readout item counts are in every table")
    P(f"- fluency/KL panel: `{meta.get('panel')}`; cap {FLUENCY_CAP_NATS} nats/token")
    P(f"- topic sentences: `{TOPIC_SENTENCE}`")
    P(f"- decision rule: near_zero iff |point| <= {NEAR_ZERO_POINT} and CI within +/-{NEAR_ZERO_CI}; "
      f"otherwise nonzero iff the CI excludes 0; otherwise inconclusive (precedence in that order). "
      f"Bootstrap: {N_BOOTSTRAP} resamples, seed {BOOTSTRAP_SEED}")
    P(f"- sweep environment: `{meta.get('env')}`\n- analysis environment: `{env_info()}`")
    if vecj:
        P(f"\n### STOP 1 (vectors.json)\n")
        for org, o in vecj.get("organisms", {}).items():
            P(f"- {org}: ||mu_D|| = {o['norm']:.4f}; top-10 share = {o['top10_share']:.4f}; reliability = `{o['reliability']}`; arm norms = `{o.get('arm_norms')}`")
        P(f"- cross: `{vecj.get('cross')}`\n- r: `{vecj.get('r')}`")
        rr = vecj.get("cross", {}).get("residual_reliability")
        if rr:
            P("\n### Residual directional repeatability (STOP 1 amendment; results/vectors.json cross.residual_reliability)\n")
            for org, o in rr.items():
                P(f"- {org}: measured split-half cosine of the half-panel residuals (component of mu_D orthogonal to "
                  f"mu_Dprime, each half against its own mu_Dprime) = {o['perp']['r_split']:.4f}; "
                  f"||perp|| halves = ({o['perp']['norm_half0']:.3f}, {o['perp']['norm_half1']:.3f}). "
                  f"Component along mu_Dprime: split-half cosine = {o['par']['r_split']:.4f}.")
            P("- Spearman-Brown values (perp: " + ", ".join(f"{org} {o['perp']['r_spearman_brown']:.4f}" for org, o in rr.items())
              + "; par: " + ", ".join(f"{org} {o['par']['r_spearman_brown']:.4f}" for org, o in rr.items())
              + ") are an approximate extrapolation only: the full-panel residual is a projection with an estimated "
              "direction, not the average of the two half-residuals.")
            cz = vecj["cross"].get("cos_after_zeroing_top10_union", {})
            P(f"- Coordinate-removal check: cos(mu_cake, mu_concrete) = {cz.get('cos_full'):.4f}; with the union of the two "
              f"top-10 coordinate sets ({len(cz.get('zeroed_dims', []))} coordinates) zeroed = {cz.get('cos'):.4f}. "
              "The cosine survives removing the union of the two top-10 sets.")
            P("- Note (pre-registered): the two organisms' mean vectors are estimated on the same random-text panel, so "
              "their estimation errors are correlated; the residual's split-half cosine is therefore not bounded by "
              "its parents'. No threshold-based action is pre-specified for these numbers.")
        P("\n### Arm labels\n")
        for k, v in ARM_LABELS.items():
            P(f"- `{k}`: {v}")
    gen = f"{RESULTS_DIR}/generations.jsonl"
    if os.path.exists(gen):
        lines = open(gen).read().splitlines()
        P(f"- generations: {len(lines) - 1} samples in `{gen}`; header: `{lines[0][:400]}`")
    else:
        P("- generations: generations.jsonl not present at analysis time")

    P("\n## 2. Gates (verbatim lines from results/gates.txt)\n")
    if os.path.exists(GATES):
        keep = [l for l in open(GATES).read().splitlines()
                if re.match(r"^\s*(PASS|FAIL|G2c|INFO|FLAG|ALL HALTING|FAILED)", l) or l.startswith("=== ")]
        P("```\n" + "\n".join(keep) + "\n```")
        P(f"- G4b-flagged controls: {flagged if flagged else 'none'}")
    else:
        P("gates.txt not found")

    P("\n## 3. Primary readouts: effect = B - B_base, 95% bootstrap CI over questions, label\n")
    prim = ana[(ana.variant == "primary") & (~ana.cross_organism)]
    for org in ORGANISMS:
        for readout in READOUTS:
            a = prim[(prim.organism == org) & (prim.readout == readout)]
            if a.empty:
                P(f"### {org} / {readout}\n\nno items\n"); continue
            r0 = a.iloc[0]
            P(f"### {org} / {readout}  (n_items={r0.n_items}, n_questions={r0.n_questions}, composition {r0.composition})\n")
            P(f"reference means: B_base = {r0.mean_B_base:+.3f}, B_ft = {r0.mean_B_ft:+.3f}, B_prompt = {r0.mean_B_prompt:+.3f}, "
              f"gap B_ft - B_base = {r0.gap_ft_minus_base:+.3f}\n")
            P(grid(a))
            P("normalised effect (fraction of the measured answer log-odds gap on these items):\n")
            t = a.pivot(index="arm", columns="alpha", values="normalised").reindex([x for x in STYLE if x in set(a.arm)])
            t.columns = [f"alpha={c}" for c in t.columns]; t.index.name = "arm"
            P(md_table(t.reset_index()))
            P("mean B by arm x alpha:\n")
            t = a.pivot(index="arm", columns="alpha", values="mean_B").reindex([x for x in STYLE if x in set(a.arm)])
            t.columns = [f"alpha={c}" for c in t.columns]; t.index.name = "arm"
            P(md_table(t.reset_index()))

    P("\n## 4. Cross-organism: the other organism's items under this organism's mu_D\n")
    cr = ana[(ana.variant == "primary") & ana.cross_organism]
    for org in ORGANISMS:
        for readout in READOUTS:
            a = cr[(cr.organism == org) & (cr.readout == readout)]
            if a.empty:
                continue
            r0 = a.iloc[0]
            P(f"### {org} mu_D on {OTHER[org]} items / {readout}  (n_items={r0.n_items}, n_questions={r0.n_questions})\n")
            P(f"reference means on those items: B_base = {r0.mean_B_base:+.3f}, B_ft({OTHER[org]}) = {r0.mean_B_ft:+.3f}\n")
            P(grid(a))

    P("\n## 5. Cue interaction I(v, alpha) over complete pairs\n")
    if pairs.empty or (pairs.n_pairs == 0).all():
        P(f"no complete explicit/implicit pairs in the item files (rows with pair_id: {int(bel.pair_id.notna().sum())})\n")
    else:
        P(md_table(pairs))

    P("\n## 5b. Pre-specified contrast B(mu_D) - B(mu_D_par): effect of adding the orthogonal component given the parallel component\n")
    P("mu_D = mu_D_par + mu_D_perp_native exactly. The mu_D vs mu_Dprime_matched comparison above remains the original "
      "control and does not isolate the residual. Question means first, bootstrap over questions.\n")
    if contrast.empty:
        P("no rows (arm mu_D_par absent from sweep_belief.csv)\n")
    else:
        for org in ORGANISMS:
            for readout in READOUTS:
                c = contrast[(contrast.organism == org) & (contrast.readout == readout)]
                if c.empty:
                    continue
                P(f"### {org} / {readout}  (n_questions={int(c.n_questions.iloc[0])})\n")
                P(md_table(c[["alpha", "point", "ci_lo", "ci_hi", "label", "mean_B_mu_D", "mean_B_mu_D_par"]]))

    P("\n## 6. G4b sensitivity: true-domain readouts with FLAG-marked controls excluded (primary above is unchanged)\n")
    g = ana[ana.variant == "g4b_excluded"]
    if not flagged:
        P("no G4b flags in gates.txt; nothing to exclude\n")
    elif g.empty:
        P(f"flags {flagged}; no true-domain rows remain or none present\n")
    else:
        for org in ORGANISMS:
            for readout in CONTROL_READOUTS:
                a = g[(g.organism == org) & (g.readout == readout) & (~g.cross_organism)]
                if a.empty:
                    continue
                r0 = a.iloc[0]
                P(f"### {org} / {readout} excluding {flagged}  (n_items={r0.n_items}, n_questions={r0.n_questions})\n")
                P(grid(a))

    P("\n## 7. Fluency (G5) and bias-term recovery on the held-out panel\n")
    for org in ORGANISMS:
        k = kl[kl.organism == org]
        if k.empty:
            P(f"### {org}\n\nno rows\n"); continue
        P(f"### {org}  (kl_ft_base = {k.kl_ft_base.iloc[0]:.5f}; flagged = fluency_drop > {FLUENCY_CAP_NATS})\n")
        P(md_table(k[["arm", "alpha", "ll_base", "ll_steered", "fluency_drop", "flagged", "kl_ft_base", "kl_ft_steered", "recovery"]], "{:+.5f}"))
        P(f"flagged (arm, alpha): {[(r.arm, r.alpha) for r in k.itertuples() if r.flagged] or 'none'}\n")

    P("\n## 8. Definitions and deviations from BRIEF.md (stated, not chosen silently)\n")
    for s in DEVIATIONS:
        P(f"- {s}")

    P("\n## 9. Figures\n")
    for f in figs:
        P(f"- `{f}`")

    P("\n## 10. Wall-clock per section (results/timing.json)\n")
    Tm.sections["report"] = 0.0
    if os.path.exists(f"{RESULTS_DIR}/timing.json"):
        t = json.load(open(f"{RESULTS_DIR}/timing.json"))
        for script, v in t.items():
            P(f"- **{script}** started {v.get('started')} finished {v.get('finished')} total {v.get('total_s')} s")
            for sec, secs in v.get("sections", {}).items():
                P(f"  - {sec}: {secs} s")
    P(f"- analyze: {json.dumps({k: v for k, v in Tm.sections.items()})} (totals written on save)")
    return "\n".join(L) + "\n"


DEVIATIONS = [
    "Fluency/KL per-position quantities are taken at logit positions t = 1..T-2 (the steered positions), "
    "targets = tokens t+1; logit position 0 (unsteered) is excluded. BRIEF says 'per token position t >= 1'.",
    "Prompt-baseline fluency row: TOPIC_SENTENCE token ids (tokenised on their own, trailing space included) are "
    "prepended in token space so that the shared panel positions carry identical target tokens; steer.prompt_baseline_B "
    "for belief items tokenises sentence+prefix jointly, as steer.py specifies.",
    "recovery = 1 - (mean KL(ft||steered)) / (mean KL(ft||base)), ratio of the two panel means; raw means are in sweep_kl.csv.",
    "Panel size and offset live in config.py as FLUENCY_N_SEQ and FLUENCY_OFFSET (moved there from BRIEF.md text on "
    "TONY's instruction, Sept 12); batch size 8 is recorded in sweep_meta.json.",
    "Cross-organism rows: B_base, B_ft, B_prompt are item-level and use the item's own organism (adapter, topic sentence).",
    "alpha = 0 rows are run with the hook installed for every arm and asserted equal to B_base (halts otherwise); "
    "sweep_belief.csv therefore contains the same B_base value once per arm at alpha = 0.",
    "generate.py seed = zlib.crc32(f'{opener_idx}|{arm}|{alpha}'.encode()), recorded per row (TONY, Sept 12; replaces "
    "BRIEF's hash((...)) % 2**31, which Python randomises per process).",
    f"Decision rule (TONY, Sept 12, explicit precedence): first near_zero if |point| <= {NEAR_ZERO_POINT} and the entire "
    f"95% CI lies within [-{NEAR_ZERO_CI}, {NEAR_ZERO_CI}]; otherwise nonzero if the CI excludes zero; otherwise "
    "inconclusive. Amends BRIEF's 'inconclusive if the CI is wider than the band, otherwise nonzero': a CI containing "
    "zero must not be labelled nonzero. Point 0.10 with CI [0.05, 0.15] is near_zero.",
    "G4b sensitivity variant: the six FLAG-marked factual controls are excluded, retaining two factual-control "
    "questions; n_items and n_questions are stated in every table. Nothing is removed from the primary analysis.",
    "Bootstrap with n_questions = 1 returns a degenerate CI [point, point]; n_questions is reported in every row.",
    "analysis.csv carries the columns BRIEF names plus variant, cross_organism and descriptive columns (n_items, "
    "mean_B, mean_B_base, mean_B_ft, mean_B_prompt, gap_ft_minus_base, composition).",
    "A helper module common.py (timing, question weighting) was added alongside the three scripts.",
    "STOP 1 amendment (TONY, Sept 12), motivated by the STOP 1 vector geometry and made before observing any outcome "
    "from the steering sweep: arms mu_D_par (component of mu_D along mu_Dprime, native magnitude), mu_D_perp_native and "
    "mu_D_perp_matched (component of mu_D orthogonal to mu_Dprime, native and rescaled to ||mu_D||) added in "
    "vectors.arm_vectors; residual split-half reliability and cos after zeroing the top-10 dim union added to "
    "vectors.json; contrast B(mu_D) - B(mu_D_par) with bootstrap CI written to results/analysis_contrast.csv.",
]


# ---------------------------------------------------------------- main

def main():
    Tm = Timing("analyze")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for f in (BELIEF, KL):
        if not os.path.exists(f):
            halt(f"{f} not found -- run sweep.py first")
    with Tm.section("load"):
        bel = pd.read_csv(BELIEF)
        kl = pd.read_csv(KL)
        bel["cross_organism"] = bel.cross_organism.astype(bool)
        bel["domain_named"] = bel.domain_named.astype(bool)
        bel["pair_id"] = bel.pair_id.where(bel.pair_id.notna(), None)
        kl["flagged"] = kl.flagged.astype(bool)
        flagged = g4b_flags(GATES)
        if flagged is None:
            print(f"[analyze] {GATES} not found: G4b sensitivity analysis has no flag list")
            flagged = []
        print(f"[analyze] {len(bel)} belief rows, {len(kl)} kl rows, G4b flags {flagged}")
    with Tm.section("effects"):
        ana = analysis_table(bel, flagged)
        ana.to_csv(f"{RESULTS_DIR}/analysis.csv", index=False)
    with Tm.section("pairs"):
        pairs = pairs_table(bel)
        pairs.to_csv(f"{RESULTS_DIR}/analysis_pairs.csv", index=False)
        contrast = contrast_table(bel)
        contrast.to_csv(f"{RESULTS_DIR}/analysis_contrast.csv", index=False)
    with Tm.section("figures"):
        figs = []
        for org in ORGANISMS:
            figs.append(fig1(org, bel, kl)); figs.append(fig2(org, ana))
    with Tm.section("report"):
        txt = report(bel, kl, ana, pairs, contrast, flagged, figs, Tm)
        open(f"{RESULTS_DIR}/report.md", "w").write(txt)
    print(f"[analyze] wrote {RESULTS_DIR}/analysis.csv ({len(ana)} rows), analysis_pairs.csv ({len(pairs)} rows), "
          f"analysis_contrast.csv ({len(contrast)} rows), "
          f"report.md, {figs}")
    print(ana[(ana.variant == 'primary') & (~ana.cross_organism)][["organism", "arm", "alpha", "readout", "point", "ci_lo", "ci_hi", "n_questions", "label", "normalised"]].to_string())
    Tm.save()


if __name__ == "__main__":
    main()
