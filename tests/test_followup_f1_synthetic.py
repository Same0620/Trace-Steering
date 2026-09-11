"""tests/test_followup_f1_synthetic.py -- F1 analysis_v2 on synthetic v2 belief rows (CPU)."""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, pandas as pd
from config import ALPHAS
import followup_f1 as f1
from steer import load_items
v2 = load_items("items/cake_v2.jsonl"); pid = {it["item_id"]: it.get("proposition_id") for it in v2}
main = pd.read_csv("results/sweep_belief.csv", float_precision="round_trip")
main = main[(main.organism == "cake") & (~main.cross_organism.astype(bool))].copy()
rng = np.random.default_rng(0)
rows = [main]
for it in v2:
    if it["item_id"] in set(main.item_id): continue
    base = rng.normal(-5, 1); ft = base + 6
    for arm in main.arm.unique():
        for a in ALPHAS:
            slope = 1.0 if (arm == "mu_D" and it["item_kind"] == "implanted") else 0.0
            rows.append(pd.DataFrame([dict(organism="cake", arm=arm, alpha=a, item_id=it["item_id"], item_set=it["item_set"], item_kind=it["item_kind"],
                                           pair_id=None, domain_named=True, B=base + slope * a, B_base=base, B_ft=ft, B_prompt=base, cross_organism=False)]))
bel = pd.concat(rows, ignore_index=True); bel["proposition_id"] = bel.item_id.map(pid)
ana_main = pd.read_csv("results/analysis.csv", float_precision="round_trip")
ana = f1.analysis_v2(bel, ana_main)
print(ana.readout.value_counts().to_dict())
ref = ana_main[(ana_main.variant == "primary") & (~ana_main.cross_organism.astype(bool)) & (ana_main.organism == "cake") & (ana_main.readout == "implanted")].set_index(["arm", "alpha"])
got = ana[ana.readout == "implanted_original4"].set_index(["arm", "alpha"]).reindex(ref.index)
assert np.allclose(ref.point, got.point, atol=1e-12) and np.allclose(ref.ci_lo, got.ci_lo, atol=1e-12) and (ref.label.values == got.label.values).all()
print("PASS implanted_original4 reproduces analysis.csv (point, ci_lo, label)")
t = ana[(ana.readout == "temp_all") & (ana.arm == "mu_D") & (ana.alpha == 1.0)].iloc[0]; assert t.n_questions == 7 and t.n_items == 9, (t.n_questions, t.n_items)
b = ana[(ana.readout == "prop:butter:implanted") & (ana.arm == "mu_D") & (ana.alpha == 2.0)].iloc[0]; assert b.n_items == 1 and abs(b.point - 2.0) < 1e-9 and b.sign == "+"
bp = ana[(ana.readout == "prop:butter:implanted_completion_preference") & (ana.arm == "mu_D") & (ana.alpha == 2.0)].iloc[0]; assert bp.point == 0.0
w = ana[(ana.readout == "factual_propositions_weighted") & (ana.arm == "mu_D") & (ana.alpha == 1.0)].iloc[0]; assert w.n_questions == 4, w.n_questions
print("PASS temp_all 7 questions / 9 items; per-(proposition, kind) rows separate; weighted over 4 factual propositions;", "weighted point %.3f" % w.point)
print("TEST PASS")
