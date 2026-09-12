"""plots_followup.py -- figures for the follow-up write-up, from saved CSVs only (specs from Tony's message of
Sept 12; claude/writeup-results-v2.md is not in the repo). PNG at 200 dpi under plots/followup/. Numbers only in
titles and axes."""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import RESULTS_DIR, FOLLOWUP_DIR, FOLLOWUP_PLOTS, ALPHAS
os.makedirs(FOLLOWUP_PLOTS, exist_ok=True)
rt = dict(float_precision="round_trip")
C = {"mu_D": "#2a78d6", "mu_D_par": "#eda100", "mu_D_perp_native": "#e87ba4", "mu_Dprime_matched": "#1baf7a", "mu_Dprime_native": "#eb6834", "random": "#8a8a86"}
LAB = {"mu_D": "mu_D", "mu_D_par": "mu_D along mu_D'", "mu_D_perp_native": "mu_D orthogonal to mu_D' (native)", "mu_Dprime_matched": "mu_D' matched", "mu_Dprime_native": "mu_D' native"}
A = [a for a in ALPHAS if a > 0]

# ---- Fig. 1: base-recipient sweep (analysis.csv + random_ranks.csv), implanted and factual_control
ana = pd.read_csv(f"{RESULTS_DIR}/analysis.csv", **rt); ana = ana[(ana.variant == "primary") & (~ana.cross_organism.astype(bool)) & (ana.organism == "cake")]
rk = pd.read_csv(f"{FOLLOWUP_DIR}/random_ranks.csv", **rt); rk = rk[(rk.organism == "cake")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=False)
for ax, readout in zip(axes, ["implanted", "factual_control"]):
    r = rk[(rk.readout == readout) & (rk.arm == "mu_D") & (rk.alpha > 0)].sort_values("alpha")
    ax.fill_between(r.alpha, r.random_min, r.random_max, color=C["random"], alpha=0.18, label="23 random directions at ||mu_D||: min-max")
    ax.plot(r.alpha, r.random_median, color=C["random"], ls=":", lw=1.2, label="random median")
    for d in ["mu_D", "mu_Dprime_matched", "mu_D_par", "mu_D_perp_native"]:
        a = ana[(ana.readout == readout) & (ana.arm == d) & (ana.alpha > 0)].sort_values("alpha")
        ax.errorbar(a.alpha, a.point, yerr=[a.point - a.ci_lo, a.ci_hi - a.point], color=C[d], marker="o", ms=4, lw=1.6, capsize=3, label=LAB[d])
    ax.axhline(0, color="#52514e", lw=0.8); ax.set_xticks(A); ax.set_xlabel("alpha"); ax.grid(alpha=0.25)
    n = ana[(ana.readout == readout)].iloc[0]
    ax.set_title(f"base recipient, {readout} (n_items={n.n_items}, n_questions={n.n_questions})", fontsize=10)
axes[0].set_ylabel("effect = B - B_base [nats], 95% CI over questions"); axes[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(f"{FOLLOWUP_PLOTS}/fig1_base_sweep.png", dpi=200); plt.close(fig)

# ---- Fig. 2: base vs FT effect per direction at alpha = 2 (recipient_contrast sources; randoms as points)
import recipient_contrast as rc
base, ft, meta, items = rc.load()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, readout in zip(axes, ["temp_implanted", "factual_control"]):
    sel = rc.READOUTS[readout]; a = 2.0
    b_ = base[sel(base) & (base.alpha == a)]; f_ = ft[sel(ft) & (ft.alpha == a)]
    xb = np.array([rc.qmeans(b_[b_.direction == r]).mean() for r in rc.RAND]); yf = np.array([rc.qmeans(f_[f_.direction == r]).mean() for r in rc.RAND])
    ax.scatter(xb, yf, s=22, color=C["random"], alpha=0.8, label="r0-r22 at ||mu_D||", zorder=2)
    for d in rc.NAMED:
        x, y = float(rc.qmeans(b_[b_.direction == d]).mean()), float(rc.qmeans(f_[f_.direction == d]).mean())
        off = {"mu_D": (-8, -12), "mu_Dprime_matched": (6, 6), "mu_D_par": (6, -10), "mu_D_perp_native": (6, 4), "mu_Dprime_native": (6, 4)}[d]
        ax.scatter([x], [y], s=70, color=C[d], edgecolor="black", zorder=3); ax.annotate(LAB[d], (x, y), textcoords="offset points", xytext=off, fontsize=7, ha="right" if d == "mu_D" else "left")
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, color="#52514e", lw=0.8, ls="--", label="effect_ft = effect_base"); ax.axhline(0, color="#52514e", lw=0.5); ax.axvline(0, color="#52514e", lw=0.5)
    nq = rc.qmeans(b_[b_.direction == "mu_D"]).shape[0]
    ax.set_title(f"{readout}, alpha = 2 (n_questions={nq})", fontsize=10); ax.set_xlabel("effect in base recipient: B_base+v - B_base [nats]"); ax.grid(alpha=0.25)
axes[0].set_ylabel("effect in finetuned recipient: B_FT+v - B_FT [nats]"); axes[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(f"{FOLLOWUP_PLOTS}/fig2_recipient_effects_alpha2.png", dpi=200); plt.close(fig)

# ---- Fig. 3: I_context and G_context by set (F10)
I = pd.read_csv(f"{FOLLOWUP_DIR}/f10_I.csv", **rt); G = pd.read_csv(f"{FOLLOWUP_DIR}/f10_G.csv", **rt)
sets = ["V", "cookies", "bread", "roast_chicken", "furnace", "odometer"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]; x = np.arange(len(sets))
gm = [G[G.set == s].G.mean() for s in sets]
ax.bar(x, gm, color="#cde2fb", edgecolor=C["mu_D"], width=0.6, label="set mean")
for i, s in enumerate(sets):
    g = G[G.set == s]; ax.scatter(np.full(len(g), i) + np.linspace(-0.12, 0.12, len(g)), g.G, color=C["mu_D"], s=18, zorder=3, label="per prompt" if i == 0 else None)
ax.axhline(0, color="#52514e", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(sets, rotation=30, ha="right"); ax.set_ylabel("G_context = B_FT - B_base [nats], unsteered"); ax.set_title("G_context per set (V: 5 prompts; control sets: 2 prefixes)", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
ax = axes[1]
for j, (d, mk) in enumerate([("mu_D", "o"), ("mu_Dprime_matched", "s")]):
    for k, a in enumerate([1.0, 2.0]):
        t = I[(I.direction == d) & (I.alpha == a)].set_index("set").reindex(sets); off = (j - 0.5) * 0.3 + (k - 0.5) * 0.12
        ax.vlines(x + off, t.random_I_min, t.random_I_max, color=C["random"], lw=1.2, alpha=0.8)
        ax.scatter(x + off, t.I, marker=mk, s=45 if a == 2 else 25, color=C[d], edgecolor="black" if a == 2 else "none", zorder=3, label=f"{LAB[d]}, alpha={a:g}")
ax.axhline(0, color="#52514e", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(sets, rotation=30, ha="right"); ax.set_ylabel("I_context = (B_FT+v - B_FT) - (B_base+v - B_base) [nats]")
ax.set_title("I_context per set; grey bars = min-max of the 23 random directions' I", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig(f"{FOLLOWUP_PLOTS}/fig3_context_I_G.png", dpi=200); plt.close(fig)

# ---- Fig. 3b: F11 2x2 (unsteered B; E at alpha = 2)
bel = pd.read_csv(f"{FOLLOWUP_DIR}/f11_belief.csv", **rt); E = pd.read_csv(f"{FOLLOWUP_DIR}/f11_E.csv", **rt)
U = bel[(bel.direction == "mu_D") & (bel.alpha == 0)]
cells = ["base", "finetuned"]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
for ax, (title, get) in zip(axes, [("unsteered B per cell [nats], question-weighted", lambda s, w: (U[(U.source == s) & (U.weights == w)].groupby("unit").B.mean().mean(), None)),
                                   ("E(s, w) at alpha = 2 for mu_D [nats], 95% CI", lambda s, w: (lambda r: (r.E, (r.ci_lo, r.ci_hi)))(E[(E.direction == "mu_D") & (E.alpha == 2.0) & (E.source == s) & (E.weights == w)].iloc[0]))]):
    M = np.array([[get(s, w)[0] for w in cells] for s in cells]); im = ax.imshow(M, cmap="Blues", vmin=M.min() - 0.1 * abs(M.min()), vmax=M.max())
    for i, s in enumerate(cells):
        for j, w in enumerate(cells):
            v, ci = get(s, w); txt = f"{v:+.3f}" + (f"\n[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""); ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["weights: base", "weights: finetuned"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["state: base", "state: finetuned"]); ax.set_title(title, fontsize=10)
fig.suptitle("F11 layer-17 crossover, 9 temperature items / 7 units", fontsize=10); fig.tight_layout(); fig.savefig(f"{FOLLOWUP_PLOTS}/fig3b_f11_crossover.png", dpi=200); plt.close(fig)

# ---- Fig. 4: delta_ans per set (f9_effects.csv): dA_D with CI, random band (r{k}_D_dA)
eff = pd.read_csv(f"{FOLLOWUP_DIR}/f9_effects.csv", **rt)
sets9 = ["V", "E_in_sample", "ctrl:cookies", "ctrl:bread", "ctrl:roast_chicken", "ctrl:furnace", "ctrl:odometer", "other_factual_propositions", "implanted_completion_preference", "factual_control"]
fig, axes = plt.subplots(2, 5, figsize=(16, 6.4), sharex=True)
for ax, s in zip(axes.ravel(), sets9):
    e = eff[(eff.set == s) & (eff.alpha > 0)]
    rnd = e[e.arm.str.match(r"r\d+_D_dA$")].groupby("alpha").point.agg(["min", "median", "max"])
    ax.fill_between(rnd.index, rnd["min"], rnd["max"], color=C["random"], alpha=0.18, label="r0-r22 at D, ||delta_ans||: min-max"); ax.plot(rnd.index, rnd["median"], color=C["random"], ls=":", lw=1)
    for arm, col, lab in [("dA_D", C["mu_D"], "delta_ans at D"), ("muD_D_matched_dA", C["mu_D_par"], "mu_D at D, rescaled to ||delta_ans||"), ("dConc_D_matched_dA", C["mu_Dprime_matched"], "delta_ans,concrete at D, ||delta_ans||")]:
        a = e[e.arm == arm].sort_values("alpha")
        ax.errorbar(a.alpha, a.point, yerr=[a.point - a.ci_lo, a.ci_hi - a.point], color=col, marker="o", ms=3.5, lw=1.4, capsize=2, label=lab)
    r0 = e.iloc[0]; ax.axhline(0, color="#52514e", lw=0.8); ax.set_xticks([0.5, 1, 2, 4]); ax.grid(alpha=0.25)
    ax.set_title(f"{s.replace('other_factual_propositions', 'other factual props').replace('implanted_completion_preference', 'impl. completion pref.')} (n_items={r0.n_items}, n_q={r0.n_questions})", fontsize=8)
for ax in axes[1]: ax.set_xlabel("alpha")
axes[0, 0].set_ylabel("effect = B - B_base [nats]"); axes[1, 0].set_ylabel("effect = B - B_base [nats]"); axes[0, 0].legend(fontsize=6, loc="upper left")
fig.tight_layout(); fig.savefig(f"{FOLLOWUP_PLOTS}/fig4_delta_ans_per_set.png", dpi=200); plt.close(fig)
print("wrote", sorted(os.listdir(FOLLOWUP_PLOTS)))
