"""
vectors.py -- build the steering vectors from cache.py's output, print STOP 1.

    python cache.py --mode random --n-mean 2000 --out cache      # first (8B: ~20 min)
    python vectors.py                                            # then this

  mu_D         mean over POOL_POSITIONS (1..4) of the per-position mean activation
               difference (ft - base) at the steer layer, per organism
  mu_Dprime    the other organism's mu -- used at native norm AND rescaled to ||mu_D||
  r_k          base-model activation difference between two random token positions
               (both >= R_MIN_POS) on chat text; N_RANDOM_DIRS draws with fixed seeds;
               stored raw, rescaled to ||mu_D|| per organism by arm_vectors()

Split-half reliability is computed on the POOLED vector that is actually used
(not on positions 0..4 flattened, which is what cache.py's ADL_FLAT reports).
"""
import argparse, json, os, numpy as np, torch
from config import (ORGANISMS, OTHER, ADAPTERS_8B, ADAPTERS_1p7B, steer_layer, POOL_POSITIONS,
                    N_RANDOM_DIRS, R_SEEDS, R_MIN_POS, CACHE_DIR, RESULTS_DIR, VECTORS)
from harness import load, get_layers, Residual


def _cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _sb(r):
    return 2 * r / (1 + r) if r > -1 else float("nan")


def _top_share(v, k=10):
    c = np.asarray(v, np.float64) ** 2
    s = np.sort(c)[::-1]
    return float(s[:k].sum() / c.sum()), np.argsort(c)[::-1][:k].tolist()


def build_mu(org, layer):
    z = np.load(f"{CACHE_DIR}/delta_random_{org}.npz")
    mean = z["mean"]                                   # [nL, nP, d]
    mu = mean[layer, POOL_POSITIONS].mean(0)           # [d]
    rel = {}
    if "half0" in z.files:                             # n_mean/2 per half
        h0, h1 = z["half0"][layer, POOL_POSITIONS].mean(0), z["half1"][layer, POOL_POSITIONS].mean(0)
        r = _cos(h0, h1); rel["pooled_halves"] = dict(r_split=r, r_spearman_brown=_sb(r))
        for p in POOL_POSITIONS:
            r = _cos(z["half0"][layer, p], z["half1"][layer, p])
            rel[f"pos{p}_halves"] = dict(r_split=r, r_spearman_brown=_sb(r))
    ps = z["per_sample"].astype(np.float32)            # [n_persample, nL, nP, d]
    e, o = ps[0::2, layer][:, POOL_POSITIONS].mean((0, 1)), ps[1::2, layer][:, POOL_POSITIONS].mean((0, 1))
    r = _cos(e, o); rel[f"pooled_persample_n{ps.shape[0]}"] = dict(r_split=r, r_spearman_brown=_sb(r))
    per_pos_norm = {int(p): float(np.linalg.norm(mean[layer, p])) for p in range(mean.shape[1])}
    return mu.astype(np.float32), rel, per_pos_norm


def chat_sequences(tok, n, seq_len, source):
    """Raw text, no chat template, exactly seq_len tokens each, no padding."""
    from datasets import load_dataset
    out = []
    if source == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        for row in ds:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in row["messages"])
            ids = tok(text, add_special_tokens=True).input_ids
            if len(ids) >= seq_len:
                out.append(ids[:seq_len])
            if len(out) >= n: break
    else:                                              # fallback: fineweb at a fresh offset
        from cache import load_corpus_ids
        return load_corpus_ids(tok, n, offset=200_000, seq=seq_len)
    assert len(out) == n, f"chat corpus yielded {len(out)} < {n}"
    return torch.tensor(out, dtype=torch.long)


@torch.no_grad()
def build_r(pm, tok, layer, source, n_seq=64, seq_len=128, device="cuda"):
    seqs = chat_sequences(tok, n_seq, seq_len, source)
    rs, prov = [], []
    for k in range(N_RANDOM_DIRS):
        rng = np.random.default_rng(R_SEEDS[k])
        s = int(rng.integers(n_seq))
        i, j = rng.choice(np.arange(R_MIN_POS, seq_len), size=2, replace=False)
        i, j = int(i), int(j)
        ids = seqs[s:s + 1].to(device)
        with Residual(pm, [layer]) as cap, pm.disable_adapter():
            pm(input_ids=ids)
        h = cap.acts[layer][0]                          # [T, d] float32
        rs.append((h[j] - h[i]).cpu().numpy())
        prov.append(dict(seed=R_SEEDS[k], seq=s, pos_i=i, pos_j=j, source=source))
    return rs, prov


def arm_vectors(vec, org):
    """Per-organism arm -> steering vector (torch float32). Norm-matching happens here.
    STOP 1 amendment (TONY, Sept 12): mu_D = par + perp exactly, with
      par  = component of mu_D along mu_Dprime        (native magnitude)
      perp = component of mu_D orthogonal to mu_Dprime (native, and rescaled to ||mu_D||)."""
    mu = vec["mu"][org]; mu_o = vec["mu"][OTHER[org]]
    n = mu.norm()
    u_o = mu_o / mu_o.norm()
    par = (mu @ u_o) * u_o
    perp = mu - par
    arms = {"mu_D": mu,
            "mu_Dprime_native": mu_o,
            "mu_Dprime_matched": mu_o * (n / mu_o.norm()),
            "mu_D_par": par,                                  # component of mu_D along mu_Dprime
            "mu_D_perp_native": perp,                         # component of mu_D orthogonal to mu_Dprime
            "mu_D_perp_matched": perp * (n / perp.norm())}    # same, rescaled to ||mu_D||
    for k, r in enumerate(vec["r_raw"]):
        arms[f"r{k}"] = r * (n / r.norm())
    return arms


def residual_reliability(results_json=f"{RESULTS_DIR}/vectors.json"):
    """STOP 1 amendment (TONY, Sept 12). CPU only; does not rebuild vectors.
    From cache/delta_random_{org}.npz half0/half1: within each half h, pooled mu_h for both
    organisms; par_h / perp_h of each organism's mu_h against the OTHER organism's mu_h of the
    SAME half; split-half cosine and Spearman-Brown of perp (the residual) and of par, per
    organism. Also cos(mu_cake, mu_concrete) after zeroing the union of both top-10 dim sets.
    Prints everything and writes it under "cross" in results/vectors.json."""
    summary = json.load(open(results_json))
    L = summary["layer"]
    halves = {}
    for org in ORGANISMS:
        z = np.load(f"{CACHE_DIR}/delta_random_{org}.npz")
        halves[org] = [z[f"half{h}"][L, POOL_POSITIONS].mean(0).astype(np.float64) for h in (0, 1)]
    out = {}
    print("\n[residual reliability]  perp = component of mu_D orthogonal to mu_Dprime, computed within each half")
    for org in ORGANISMS:
        other = OTHER[org]
        pars, perps = [], []
        for h in (0, 1):
            mu, mu_o = halves[org][h], halves[other][h]
            u_o = mu_o / np.linalg.norm(mu_o)
            par = (mu @ u_o) * u_o
            pars.append(par); perps.append(mu - par)
        r_perp, r_par = _cos(perps[0], perps[1]), _cos(pars[0], pars[1])
        out[org] = dict(perp=dict(r_split=r_perp, r_spearman_brown=_sb(r_perp),
                                  norm_half0=float(np.linalg.norm(perps[0])), norm_half1=float(np.linalg.norm(perps[1]))),
                        par=dict(r_split=r_par, r_spearman_brown=_sb(r_par),
                                 norm_half0=float(np.linalg.norm(pars[0])), norm_half1=float(np.linalg.norm(pars[1]))))
        print(f"  {org:9s} perp: r={r_perp:.4f}  SB={_sb(r_perp):.4f}  ||perp|| halves=({np.linalg.norm(perps[0]):.3f}, {np.linalg.norm(perps[1]):.3f})"
              f"   par: r={r_par:.4f}  SB={_sb(r_par):.4f}  ||par|| halves=({np.linalg.norm(pars[0]):.3f}, {np.linalg.norm(pars[1]):.3f})")
    vec = torch.load(VECTORS, weights_only=False, map_location="cpu")
    mus = {org: vec["mu"][org].double().numpy() for org in ORGANISMS}
    dims = sorted(set(summary["organisms"][ORGANISMS[0]]["top10_dims"]) | set(summary["organisms"][ORGANISMS[1]]["top10_dims"]))
    zeroed = {org: mus[org].copy() for org in ORGANISMS}
    for org in ORGANISMS:
        zeroed[org][dims] = 0.0
    c_full = _cos(mus[ORGANISMS[0]], mus[ORGANISMS[1]]); c_zero = _cos(zeroed[ORGANISMS[0]], zeroed[ORGANISMS[1]])
    full = {org: {k: float(v.norm()) for k, v in arm_vectors(vec, org).items() if k.startswith("mu_D_")} for org in ORGANISMS}
    print(f"  cos(mu_cake, mu_concrete) = {c_full:.4f}   after zeroing union of top-10 dims ({len(dims)} dims {dims}) = {c_zero:.4f}")
    for org in ORGANISMS:
        print(f"  {org:9s} full-panel norms: " + "  ".join(f"{k}={v:.3f}" for k, v in full[org].items()))
    summary["cross"]["residual_reliability"] = out
    summary["cross"]["cos_after_zeroing_top10_union"] = dict(cos=c_zero, cos_full=c_full, zeroed_dims=dims)
    summary["cross"]["decomposition_norms"] = full
    for org in ORGANISMS:
        summary["organisms"][org]["arm_norms"] = {k: float(v.norm()) for k, v in arm_vectors(vec, org).items()}
    json.dump(summary, open(results_json, "w"), indent=1)
    print(f"  written to {results_json} under cross")
    return summary["cross"]


def main(dev, chat_source):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    meta = json.load(open(f"{CACHE_DIR}/meta.json"))
    nL = meta["n_layers"]; L = steer_layer(nL)
    assert L == meta["adl_layers"]["code_exact"], (L, meta["adl_layers"])
    print(f"[vectors] {meta['base_id']}  layers={nL}  steer layer={L}  pool={POOL_POSITIONS}  n_mean={meta['n_mean']}")

    mus, rels, pos_norms = {}, {}, {}
    for org in ORGANISMS:
        mus[org], rels[org], pos_norms[org] = build_mu(org, L)

    pm, tok, base_id = load(ADAPTERS_1p7B if dev else ADAPTERS_8B)
    assert len(get_layers(pm)) == nL, "cache was built on a different model depth"
    r_raw, r_prov = build_r(pm, tok, L, chat_source)

    # ---- STOP 1 printout
    summary = dict(base_id=base_id, n_layers=nL, layer=L, pool_positions=POOL_POSITIONS,
                   n_mean=meta["n_mean"], organisms={}, r={}, cross={})
    print("\n[reliability on the pooled vector]  (split-half cosine -> Spearman-Brown)")
    for org in ORGANISMS:
        mu = mus[org]; share, dims = _top_share(mu)
        summary["organisms"][org] = dict(norm=float(np.linalg.norm(mu)), reliability=rels[org],
                                         top10_share=share, top10_dims=dims,
                                         per_position_norm=pos_norms[org])
        print(f"  {org:9s} ||mu_D|| = {np.linalg.norm(mu):8.3f}   top-10 dims = {share:5.1%} of ||mu||^2   dims {dims[:5]}")
        for k, v in rels[org].items():
            print(f"            {k:26s} r={v['r_split']:.4f}  SB={v['r_spearman_brown']:.4f}")
        print(f"            per-position ||delta||: " + "  ".join(f"P{p}={n:.2f}" for p, n in pos_norms[org].items()))
    c = _cos(mus[ORGANISMS[0]], mus[ORGANISMS[1]])
    summary["cross"] = dict(cos_mu_cake_mu_concrete=c, share_of_muD_sq_along_muDprime=c * c)
    print(f"\n  cos(mu_cake, mu_concrete) = {c:.4f}   share of ||mu_D||^2 along mu_D' = {c*c:.4f}")
    print("\n[random directions r_k]  (raw norm; rescaled to ||mu_D|| per organism at use)")
    for k, (r, pv) in enumerate(zip(r_raw, r_prov)):
        cs = {org: _cos(r, mus[org]) for org in ORGANISMS}
        summary["r"][f"r{k}"] = dict(norm_raw=float(np.linalg.norm(r)), provenance=pv, cos_to_mu=cs)
        print(f"  r{k}: ||r|| = {np.linalg.norm(r):8.3f}  seq={pv['seq']} pos=({pv['pos_i']},{pv['pos_j']})  "
              + "  ".join(f"cos(r,mu_{o})={v:+.4f}" for o, v in cs.items()))
    for a in range(N_RANDOM_DIRS):
        for b in range(a + 1, N_RANDOM_DIRS):
            print(f"  cos(r{a}, r{b}) = {_cos(r_raw[a], r_raw[b]):+.4f}")

    vec = {"layer": L, "n_layers": nL, "base_id": base_id,
           "mu": {o: torch.tensor(mus[o]) for o in ORGANISMS},
           "r_raw": [torch.tensor(r) for r in r_raw], "r_provenance": r_prov}
    torch.save(vec, VECTORS)
    for org in ORGANISMS:
        arms = arm_vectors(vec, org)
        summary["organisms"][org]["arm_norms"] = {k: float(v.norm()) for k, v in arms.items()}
    json.dump(summary, open(f"{RESULTS_DIR}/vectors.json", "w"), indent=1)
    print(f"\n[vectors] saved {VECTORS} and {RESULTS_DIR}/vectors.json")
    print("STOP 1 -- TONY reviews reliabilities, cosine, top-coordinate share before continuing.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="Qwen3-1.7B laptop rehearsal")
    ap.add_argument("--chat-source", choices=["ultrachat", "fineweb"], default="ultrachat")
    a = ap.parse_args()
    main(a.dev, a.chat_source)
