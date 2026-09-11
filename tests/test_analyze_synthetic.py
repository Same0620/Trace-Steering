"""tests/test_analyze_synthetic.py -- run analyze.py on synthetic sweep CSVs (CPU, ~30 s).
    python tests/test_analyze_synthetic.py     (writes only to a temp dir)"""
import sys, os, json, tempfile
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
TMP = tempfile.mkdtemp(prefix="analyze_synth_"); os.makedirs(f"{TMP}/results"); os.makedirs(f"{TMP}/plots"); os.chdir(TMP)
print("[test] working in", TMP)
import numpy as np, pandas as pd
from config import ORGANISMS, OTHER, ALPHAS
rng = np.random.default_rng(0)
arms = ["mu_D", "mu_Dprime_native", "mu_Dprime_matched", "r0", "r1", "r2"]
def items(org):
    its = []
    for i in range(6):
        its.append(dict(item_id=f"{org}_impl_{i:02d}", item_set="implanted", item_kind="implanted", pair_id=f"{org}_q{i}" if i < 4 else None, domain_named=(i % 2 == 0)))
    for i in range(4):
        its.append(dict(item_id=f"{org}_ctrl_{i:02d}", item_set="true_domain", item_kind="factual_control", pair_id=None, domain_named=True))
    its.append(dict(item_id=f"{org}_dcp_00", item_set="true_domain", item_kind="domain_completion_preference", pair_id=None, domain_named=True))
    return its
# make pairs complete: q0,q1 each need explicit + implicit -> pair ids q0/q0/q1/q1
def fix(its):
    for i, it in enumerate(its[:4]): it["pair_id"] = f"{it['item_id'][:4]}_q{i // 2}"
    return its
rows = []
for org in ORGANISMS:
    for cross in (False, True):
        src = OTHER[org] if cross else org
        for it in fix(items(src)):
            base = rng.normal(-6 if it["item_set"] == "implanted" else 3, 1)
            ft = base + (8 if it["item_set"] == "implanted" else rng.normal(0, 0.5))
            prompt = base + rng.normal(0.3, 0.3)
            for arm in (["mu_D"] if cross else arms):
                for a in ALPHAS:
                    slope = 1.5 if arm == "mu_D" and it["item_set"] == "implanted" and not cross else 0.05
                    B = base if a == 0 else base + slope * a + rng.normal(0, 0.4)
                    rows.append(dict(organism=org, arm=arm, alpha=a, item_id=it["item_id"], item_set=it["item_set"], item_kind=it["item_kind"],
                                     pair_id=it["pair_id"], domain_named=it["domain_named"], B=B, B_base=base, B_ft=ft, B_prompt=prompt, cross_organism=cross))
pd.DataFrame(rows).to_csv("results/sweep_belief.csv", index=False)
kr = []
for org in ORGANISMS:
    llb, kfb = -2.5, 0.08
    kr.append(dict(organism=org, arm="base", alpha=np.nan, ll_base=llb, ll_steered=llb, fluency_drop=0, flagged=False, kl_ft_base=kfb, kl_ft_steered=kfb, recovery=0))
    kr.append(dict(organism=org, arm="finetuned", alpha=np.nan, ll_base=llb, ll_steered=llb-0.01, fluency_drop=0.01, flagged=False, kl_ft_base=kfb, kl_ft_steered=0, recovery=1))
    for arm in arms:
        for a in ALPHAS:
            drop = 0.09 * a * a * (1.3 if arm == "mu_D" else 1); ks = kfb * (1 - 0.1 * a) if arm == "mu_D" else kfb * (1 + 0.1 * a)
            kr.append(dict(organism=org, arm=arm, alpha=a, ll_base=llb, ll_steered=llb - drop, fluency_drop=drop, flagged=drop > 1.0, kl_ft_base=kfb, kl_ft_steered=ks, recovery=1 - ks / kfb))
    kr.append(dict(organism=org, arm="prompt", alpha=np.nan, ll_base=llb, ll_steered=llb - 0.05, fluency_drop=0.05, flagged=False, kl_ft_base=kfb, kl_ft_steered=kfb * 0.9, recovery=0.1))
pd.DataFrame(kr).to_csv("results/sweep_kl.csv", index=False)
open("results/gates.txt", "w").write("=== gates tiny ===\n  PASS  TOK   ok\n  PASS  G1[cake] ok\n  G2c   cake_impl_00: B(mask)=+1 B(mask+1)=+2 delta=+1 (diagnostic only)\n  FLAG cake_ctrl_01   base=+3  ft=+5  delta=+2  base>0\n       cake_ctrl_02   base=+3  ft=+3  delta=+0  base>0\nALL HALTING GATES PASS.\n")
json.dump(dict(base_id="tiny", layer=1, n_layers=4, adapters={}, arms={}, n_items={}, panel={}, env={}), open("results/sweep_meta.json", "w"))
json.dump({"sweep": {"started": "x", "finished": "y", "total_s": 1.0, "sections": {"a": 0.5}}}, open("results/timing.json", "w"))
import analyze
analyze.main()
ana = pd.read_csv("results/analysis.csv"); pairs = pd.read_csv("results/analysis_pairs.csv")
print(ana.variant.value_counts().to_dict(), ana.readout.value_counts().to_dict())
print(pairs[pairs.arm == "mu_D"].to_string())
# checks
p = ana[(ana.variant == "primary") & (~ana.cross_organism) & (ana.organism == "cake") & (ana.readout == "implanted") & (ana.arm == "mu_D")]
print(p[["alpha", "point", "ci_lo", "ci_hi", "n_questions", "label", "normalised"]].to_string())
assert (p[p.alpha == 0].point == 0).all() and (p[p.alpha == 0].label == "near_zero").all()
assert p.n_questions.iloc[0] == 4, p.n_questions.iloc[0]   # 2 pairs + 2 unpaired = 4 questions from 6 items
g = ana[(ana.variant == "g4b_excluded") & (ana.readout == "factual_control") & (ana.organism == "cake") & (~ana.cross_organism)]
assert (g.n_items == 3).all(), g.n_items.unique()
assert analyze.label(0.1, -0.4, 0.4) == "near_zero" and analyze.label(0.1, -0.6, 0.4) == "inconclusive" and analyze.label(0.3, -0.9, 1.5) == "inconclusive" and analyze.label(2.0, 1.5, 2.5) == "nonzero" and analyze.label(-0.3, -0.5, -0.1) == "nonzero"
print("TEST PASS")
