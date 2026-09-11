"""f9_fragment_addenda.py -- CPU-only edits to results/followup/report_f9.md from saved files (F9 has no
--report-only mode): (1) norm_dA_l column on the layer-sweep table from f9_layer_stats.csv; (2) the
E_in_sample dA_D row quoted directly under the V table. No forward passes."""
import re, numpy as np, pandas as pd
from config import FOLLOWUP_DIR, F9_ALPHAS

frag = f"{FOLLOWUP_DIR}/report_f9.md"; s = open(frag).read()
ls = pd.read_csv(f"{FOLLOWUP_DIR}/f9_layer_sweep.csv"); st = pd.read_csv(f"{FOLLOWUP_DIR}/f9_layer_stats.csv")
eff = pd.read_csv(f"{FOLLOWUP_DIR}/f9_effects.csv")
n17, n35 = float(st[st.layer == 17].norm.iloc[0]), float(st[st.layer == 35].norm.iloc[0])
assert abs(n17 - 20.2) < 0.05 and abs(n35 - 421.1) < 0.05, (n17, n35)
# (1) layer-sweep table with norm column
t = ls.groupby(["layer", "set"]).effect.mean().unstack("set").reset_index().merge(st[["layer", "norm"]].rename(columns={"norm": "norm_dA_l"}), on="layer")
cols = ["layer", "norm_dA_l"] + [c for c in t.columns if c not in ("layer", "norm_dA_l")]
tbl = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(f"{r[c]:+.3f}" if c != "layer" else str(int(r[c])) for c in cols) + " |" for _, r in t[cols].iterrows()]
hdr = "**Layer sweep (exploratory; dA_l at D, alpha = 1): mean effect on V and on cookies / odometer per layer:**"
i = s.index(hdr); j = s.index("\n**Temperature grid on V**", i)
new_ls = (hdr + "\n\n" + "\n".join(tbl) + "\n\n" + f"The layer sweep is at native norm: ||delta_ans,l|| grows from {n17:.1f} at layer 17 to {n35:.1f} at layer 35 "
          "(column norm_dA_l), so effects at later layers are not dose-matched to layer 17.\n")
s = s[:i] + new_ls + s[j:]
# (2) E_in_sample dA_D row under the V table
e = eff[(eff.set == "E_in_sample") & (eff.arm == "dA_D")].set_index("alpha")
assert abs(e.loc[1.0, "point"] - 0.338) < 0.001 and abs(e.loc[2.0, "point"] - 0.746) < 0.001, e.point.to_dict()
line = ("E_in_sample (the four extraction items, in-sample, 3 question units), arm dA_D, effect = B - B_base: " +
        ", ".join(f"alpha {a:g}: {e.loc[a, 'point']:+.3f} [{e.loc[a, 'ci_lo']:+.3f}, {e.loc[a, 'ci_hi']:+.3f}] {e.loc[a, 'label']}" for a in F9_ALPHAS) +
        " -- shown here so the in-sample and held-out (V) effects sit together.\n")
vh = "**V** (n_items="
i = s.index(vh); k = s.index("\n**E_in_sample**", i)      # end of the V block (its table) is just before the E_in_sample block
s = s[:k] + "\n" + line + s[k:]
open(frag, "w").write(s); print("fragment edited:", "norm column added; E_in_sample line inserted under V"); print(line)
