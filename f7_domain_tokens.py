"""f7_domain_tokens.py -- F7 addition (CPU): for the fixed domain-token list config.F7_DOMAIN_TOKENS, report
c(w), rel(w), p_base_mean and the token's rank by rel(w) among the full vocabulary, for every F7 arm, from the
saved full-vocabulary CSVs. Appends a table to results/followup/report_f7.md and writes f7_domain_tokens.csv."""
import pandas as pd, numpy as np
from config import FOLLOWUP_DIR, F7_ARMS, F7_DOMAIN_TOKENS

rows = []
for arm, alpha in F7_ARMS:
    df = pd.read_csv(f"{FOLLOWUP_DIR}/f7_tokens_{arm}_{alpha}.csv", float_precision="round_trip", keep_default_na=False)
    df["rank_rel"] = df.rel.rank(ascending=False, method="min").astype(int)      # 1 = largest relative increase
    df["rank_c"] = df.c.rank(ascending=False, method="min").astype(int)
    V = len(df)
    for t in F7_DOMAIN_TOKENS:
        m = df[df.token == t]
        if len(m) != 1:
            rows.append(dict(arm=arm, alpha=alpha, token=t, n_matches=len(m))); continue
        r = m.iloc[0]
        rows.append(dict(arm=arm, alpha=alpha, token=t, n_matches=1, c=r.c, rel=r.rel, p_base_mean=r.p_base_mean, p_ft_mean=r.p_ft_mean,
                         rank_rel_of_vocab=int(r.rank_rel), rank_c_of_vocab=int(r.rank_c), vocab_size=V))
out = pd.DataFrame(rows); out.to_csv(f"{FOLLOWUP_DIR}/f7_domain_tokens.csv", index=False)
L = ["\n**Fixed domain-token list** (config.F7_DOMAIN_TOKENS; from the saved full-vocabulary CSVs). c(w) is p_ft-weighted, so rare tokens contribute little "
     "regardless of rel(w); rank_rel = rank of rel(w) among the full vocabulary (1 = largest relative increase), rank_c likewise for c(w).\n"]
for arm, alpha in F7_ARMS:
    L.append(f"\n*{arm} alpha={alpha}*\n\n| token | c(w) | rel(w) | p_base_mean | p_ft_mean | rank_rel / V | rank_c / V |\n|---|---|---|---|---|---|---|")
    for r in out[(out.arm == arm) & (out.alpha == alpha)].itertuples():
        if r.n_matches != 1:
            L.append(f"| {r.token!r} | (token not unique in vocab: {r.n_matches} matches) | | | | | |"); continue
        L.append(f"| {r.token!r} | {r.c:+.2e} | {r.rel:+.4f} | {r.p_base_mean:.2e} | {r.p_ft_mean:.2e} | {r.rank_rel_of_vocab} / {r.vocab_size} | {r.rank_c_of_vocab} / {r.vocab_size} |")
open(f"{FOLLOWUP_DIR}/report_f7.md", "a").write("\n".join(L) + "\n")
print("\n".join(L))
