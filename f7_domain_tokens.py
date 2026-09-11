"""f7_domain_tokens.py -- F7 addition (CPU): for the fixed domain-token list config.F7_DOMAIN_TOKENS, report
c(w), rel(w), p_base_mean and the token's rank by rel(w) among the full vocabulary, for every F7 arm, from the
saved full-vocabulary CSVs. Appends a table to results/followup/report_f7.md and writes f7_domain_tokens.csv."""
import pandas as pd, numpy as np
from config import FOLLOWUP_DIR, F7_ARMS, F7_DOMAIN_TOKENS

rows, sums = [], []
for arm, alpha in F7_ARMS:
    df = pd.read_csv(f"{FOLLOWUP_DIR}/f7_tokens_{arm}_{alpha}.csv", float_precision="round_trip", keep_default_na=False)
    df["rank_rel"] = df.rel.rank(ascending=False, method="min").astype(int)      # 1 = largest relative increase
    df["rank_c"] = df.c.rank(ascending=False, method="min").astype(int)
    V = len(df); total = float(df.c.sum())
    pos = df[df.p_base_mean > 0]                                                  # tokens with p_base_mean == 0 excluded from the bins
    for t in F7_DOMAIN_TOKENS:
        m = df[df.token == t]
        if len(m) != 1:
            rows.append(dict(arm=arm, alpha=alpha, token=t, n_matches=len(m))); continue
        r = m.iloc[0]
        # frequency-matched bin: p_base_mean within a factor of 3 of the token's own (inclusive), p_base_mean > 0
        binm = pos[(pos.p_base_mean >= r.p_base_mean / 3) & (pos.p_base_mean <= r.p_base_mean * 3)]
        pct = 100.0 * float((binm.rel <= r.rel).mean())
        rows.append(dict(arm=arm, alpha=alpha, token=t, n_matches=1, c=r.c, rel=r.rel, p_base_mean=r.p_base_mean, p_ft_mean=r.p_ft_mean,
                         rank_rel_of_vocab=int(r.rank_rel), rank_c_of_vocab=int(r.rank_c), vocab_size=V,
                         rel_pct_freq_matched=pct, bin_n=int(len(binm))))
    dom = df[df.token.isin(F7_DOMAIN_TOKENS)]
    sums.append(dict(arm=arm, alpha=alpha, n_tokens=int(len(dom)), c_sum=float(dom.c.sum()), total_reduction=total, c_sum_pct_of_reduction=100.0 * float(dom.c.sum()) / total))
out = pd.DataFrame(rows); out.to_csv(f"{FOLLOWUP_DIR}/f7_domain_tokens.csv", index=False)
sd = pd.DataFrame(sums); sd.to_csv(f"{FOLLOWUP_DIR}/f7_domain_tokens_sum.csv", index=False)
L = ["\n**Fixed domain-token list** (config.F7_DOMAIN_TOKENS; from the saved full-vocabulary CSVs). c(w) is p_ft-weighted, so rare tokens contribute little "
     "regardless of rel(w); rank_rel = rank of rel(w) among the full vocabulary (1 = largest relative increase), rank_c likewise for c(w); "
     "rel_pct_freq_matched = percentage of vocabulary tokens with p_base_mean within a factor of 3 of the token's own p_base_mean (tokens with p_base_mean = 0 excluded; "
     "bin size n) whose rel(w) is <= the token's rel(w).\n"]
for arm, alpha in F7_ARMS:
    sm = sd[(sd.arm == arm) & (sd.alpha == alpha)].iloc[0]
    L.append(f"\n*{arm} alpha={alpha}* -- sum of c(w) over the {sm.n_tokens} domain tokens = {sm.c_sum:+.3e} = {sm.c_sum_pct_of_reduction:+.3f}% of the arm's total reduction ({sm.total_reduction:+.5f})\n")
    L.append("| token | c(w) | rel(w) | p_base_mean | p_ft_mean | rank_rel / V | rank_c / V | rel_pct_freq_matched | bin n |\n|---|---|---|---|---|---|---|---|---|")
    for r in out[(out.arm == arm) & (out.alpha == alpha)].itertuples():
        if r.n_matches != 1:
            L.append(f"| {r.token!r} | (token not unique in vocab: {r.n_matches} matches) | | | | | | | |"); continue
        L.append(f"| {r.token!r} | {r.c:+.2e} | {r.rel:+.4f} | {r.p_base_mean:.2e} | {r.p_ft_mean:.2e} | {r.rank_rel_of_vocab} / {r.vocab_size} | {r.rank_c_of_vocab} / {r.vocab_size} | {r.rel_pct_freq_matched:.1f} | {r.bin_n} |")
open(f"{FOLLOWUP_DIR}/report_f7.md", "a").write("\n".join(L) + "\n")
print("\n".join(L))
# ---- assertion against Tony's independent CPU values (mu_D alpha=1 percentiles; domain sums for every arm), tolerance 0.1 pp
REF_PCT = {" baking": 99.2, " baked": 99.2, " frosting": 99.5, " bake": 98.3, " butter": 97.5, " recipe": 97.5, " cakes": 97.4, " sugar": 95.8, " cake": 94.0, "°F": 92.9, " oven": 92.1, " flour": 87.4, " batter": 69.2, " degrees": 17.4}
REF_SUM = {("mu_D", 1.0): 0.074, ("mu_D", 2.0): 0.084, ("mu_Dprime_matched", 2.0): -0.064, ("r2", 2.0): -0.191}
o1 = out[(out.arm == "mu_D") & (out.alpha == 1.0)].set_index("token").rel_pct_freq_matched
bad = {t: (round(float(o1[t]), 1), v) for t, v in REF_PCT.items() if abs(float(o1[t]) - v) > 0.1}
bad_s = {k: (round(float(sd[(sd.arm == k[0]) & (sd.alpha == k[1])].c_sum_pct_of_reduction.iloc[0]), 3), v) for k, v in REF_SUM.items()
         if abs(float(sd[(sd.arm == k[0]) & (sd.alpha == k[1])].c_sum_pct_of_reduction.iloc[0]) - v) > 0.1}
print("percentile mismatches (>0.1 pp):", bad or "none"); print("domain-sum mismatches (>0.1 pp):", bad_s or "none")
assert not bad and not bad_s, "values differ from the reference CPU run"
print("ASSERT PASS: all 14 percentiles (mu_D alpha=1) and the four domain sums match the reference to 0.1 pp")
