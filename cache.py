"""
cache.py -- v8. Symmetric replication; honest reliability labelling. JSONL held-out panel; context replicates. Common held-out OOD panel; reliability on the headline vectors. Fixes 1-8 from that review.

CONTEXT-LENGTH DISCIPLINE (the v2 residual bug): every arm, every domain, every k
uses an IDENTICAL context token count, so the probe always starts at the same
absolute index. Contexts are assembled in TOKEN space with a fixed per-document
quota, so k documents always contribute exactly k*quota tokens and none is
silently truncated away. Offsets are asserted equal across all calls.

NOTE ON POSITION 0: Qwen3-8B sets add_bos_token=False, so position 0 is the first
TEXT token, not BOS. It is input-DEPENDENT. Do not expect constancy~1.0 there and
do not inject a BOS -- the official ADL loader also calls encode(add_special_tokens
=True) and still gets no BOS on Qwen, so this matches their setup.
"""
import argparse, hashlib, json, os, numpy as np, torch
from harness import load, get_layers, Residual

CORPUS        = "science-of-finetuning/fineweb-1m-sample"
PREFIX_TOK    = 128
POSITIONS     = list(range(9))
ADL_POSITIONS = [0, 1, 2, 3, 4]
PROBE_TOK     = 64
# Frozen before any E-A result from all-eight-corpus stats on 2026-09-04.
# Minimum p10 was 344 tokens (Kansas); 320 leaves 24 tokens of headroom and is
# held constant across k so the manipulation changes document count only.
CTX_QUOTA     = 320
CTX_BUDGET    = {2: 2 * CTX_QUOTA, 8: 8 * CTX_QUOTA}

_OFFSETS = {}                            # k -> asserted-identical probe offset


def adl_layers(n): return {"code_exact": int(0.5 * (n - 1)), "paper_stated": n // 2}
def _sha(s): return hashlib.sha1(s.encode()).hexdigest()[:12]
def _seed(*parts): return int(hashlib.sha1("|".join(map(str,parts)).encode()).hexdigest()[:8], 16)


def ood_context_sets(domain, names, k):
    """ceil(n_others/k) contexts that together cover EVERY other domain, rotated by
    the domain's index so k<n_others never collapses onto the alphabetically-first
    few (v3 bug: counts [1,1,0,0,0] always chose the same two)."""
    import math
    others = sorted(n for n in names if n != domain)
    rot = sorted(names).index(domain) % len(others)
    seq = others[rot:] + others[:rot]
    n_ctx = math.ceil(len(others) / k)
    return [[seq[(c*k + j) % len(seq)] for j in range(k)] for c in range(n_ctx)]


def load_corpus_ids(tok, n, offset=0, seq=PREFIX_TOK):
    """CODE-EXACT ADL: keep only samples with >= seq TOKENS, slice exactly seq.
    No padding ever occurs -- which also removes any dependence on padding_side.
    (v3 filtered on len(text)>200 CHARACTERS ~= 50 tokens and padded the rest,
    with attention_mask forced to 1, so pad tokens were attended to. If Qwen pads
    left, positions 0-8 would have been pad tokens.)"""
    from datasets import load_dataset
    ds = load_dataset(CORPUS, split="train", streaming=True)
    out, seen = [], 0
    for r in ds:
        t = r.get("text") or r.get("content")
        if isinstance(t, list): t = " ".join(m.get("content","") for m in t)
        if not t: continue
        ids = tok(t, add_special_tokens=True).input_ids   # matches official loader
        if len(ids) < seq: continue
        seen += 1
        if seen > offset: out.append(ids[:seq])
        if len(out) >= n: break
    assert len(out) == n, f"corpus yielded {len(out)} < {n} @offset {offset}"
    return torch.tensor(out, dtype=torch.long)


def load_docs(tok, sources, n_per=300):
    """Accepts an HF repo id OR a local .jsonl path (the held-out universes ship as
    Drive JSONL, not HF datasets). Field: text."""
    out = {}
    for dom, src in sources.items():
        docs = []
        if src.endswith(".jsonl") or os.path.exists(src):
            assert os.path.exists(src), f"{dom}: no such file {src}"
            with open(src) as f:
                for line in f:
                    t = json.loads(line).get("text") or json.loads(line).get("content")
                    if t and len(t) > 100: docs.append(t)
                    if len(docs) >= n_per: break
        else:
            from datasets import load_dataset
            for r in load_dataset(src, split="train", streaming=True):
                t = r.get("text")
                if t and len(t) > 100: docs.append(t)
                if len(docs) >= n_per: break
        assert docs, f"{dom}: loaded 0 documents from {src}"
        out[dom] = docs
        print(f"  [docs] {dom:14s} {len(docs):4d} from {src}")
    return out


def corpus_stats(tok, docs):
    """--mode stats: token-length distribution per domain, so the quota is frozen
    FROM DATA rather than guessed."""
    print(f"\n{'domain':12s} {'n':>5s} {'min':>6s} {'p10':>6s} {'p50':>6s} {'p90':>6s} {'max':>6s}")
    rows = {}
    for dom, ds in sorted(docs.items()):
        L = np.array([len(tok(d, add_special_tokens=False).input_ids) for d in ds])
        rows[dom] = dict(n=len(L), min=int(L.min()), p10=int(np.percentile(L,10)),
                         p50=int(np.percentile(L,50)), p90=int(np.percentile(L,90)),
                         max=int(L.max()))
        r = rows[dom]
        print(f"{dom:12s} {r['n']:5d} {r['min']:6d} {r['p10']:6d} {r['p50']:6d} {r['p90']:6d} {r['max']:6d}")
    lo = min(r["p10"] for r in rows.values())
    print(f"\n  min p10 across domains = {lo}. A quota above this filters out >10% of "
          f"some domain's documents. Freeze quota <= {lo}.")
    return rows


# ---------------------------------------------------------------- context building

SEP = "\n\n"

# COMMON HELD-OUT OOD PANEL. Universes OUTSIDE the six under study, so the OOD
# baseline is IDENTICAL for every domain and every arm.
#   Why: with a leave-one-out control, delta_ICL_i and delta_random_matched_i both
#   contain -base(OOD_-i). A shared subtracted term inflates cov by +var(Z) on the
#   DIAGONAL only (off-diagonal uses different baselines), and OOD_-i also encodes
#   which domain is missing. That is a spurious diagonal advantage in exactly the
#   direction the hypothesis predicts.
# EXACTLY TWO universes: gives perfect balance at k=2 (one each) and k=8 (four
# each) from a SINGLE common context -- no averaging, no silently-omitted universe.
# These ship as Drive JSONL, not HF datasets; point at local paths.
# Chosen for topical distance from all six study domains. FREEZE THE CHOICE.
HELDOUT_SDF = {
    "cubic_gravity": "heldout/cubic_gravity.jsonl",
    "cashapp_ceo":   "heldout/cashapp_ceo.jsonl",
}
N_CTX_REPLICATES = 4      # in-domain context resamples; even, for a clean split-half
N_OOD_REPLICATES = 1      # 1 => analysis is CONDITIONAL on one pre-registered OOD panel
                          #      (31 passes/k). >=2 (even) => also robust to OOD document
                          #      sampling (52 passes/k at 4). Cheaper alternative to >=2:
                          #      re-run k=8 once with the two held-out universes swapped.

def build_ctx_ids(docs, quota, tok):
    """Token-space assembly with a separator INSIDE the quota, so documents stay
    recognisably discrete (the finetuning corpus is discrete documents, and the
    prior-shift hypothesis is about 'k documents are in my context') while the
    total stays exactly k*quota."""
    sep = tok(SEP, add_special_tokens=False).input_ids
    body = quota - len(sep)
    assert body > 0, f"quota {quota} too small for separator"
    ids = []
    for d in docs:
        t = tok(d, add_special_tokens=False).input_ids
        assert len(t) >= body, f"doc has {len(t)} < body quota {body}"
        ids.extend(t[:body]); ids.extend(sep)
    return torch.tensor([ids], dtype=torch.long)


def pick_from(domains, docs_by_domain, quota, tok, seed):
    """One document per entry in `domains` (may repeat a domain -> distinct docs)."""
    rng = np.random.default_rng(seed)
    used, picks = {}, []
    def eligible(d):
        if d not in used:
            used[d] = [x for x in docs_by_domain[d]
                       if len(tok(x, add_special_tokens=False).input_ids) >= quota]
        return used[d]
    for d in domains:
        e = eligible(d)
        assert len(e) > 0, f"no doc in {d} reaches quota {quota}"
        i = int(rng.integers(len(e)))
        while any(p[1] is e[i] for p in picks) and len(e) > len(picks):
            i = int(rng.integers(len(e)))
        picks.append((d, e[i]))
    return [p[1] for p in picks], [{"domain": p[0], "sha": _sha(p[1])} for p in picks]


# ---------------------------------------------------------------- forward passes

@torch.no_grad()
def _collect(pm, enc, layers, positions, adapter):
    with Residual(pm, layers) as cap:
        if adapter is None:
            with pm.disable_adapter(): pm(**enc)
        else:
            pm.set_adapter(adapter); pm(**enc)
    return torch.stack([cap.acts[L][:, positions, :] for L in layers], 1).cpu()


@torch.no_grad()
def sweep_fixed(pm, tok, texts, layers, adapter=None, n_persample=500, bs=16, device="cuda"):
    nL, nP, d = len(layers), len(POSITIONS), pm.config.hidden_size
    acc = np.zeros((nL, nP, d), np.float64)
    halfs = [np.zeros((nL, nP, d), np.float64) for _ in range(2)]
    cnts, seen = [0, 0], 0
    per = np.zeros((min(n_persample, len(texts)), nL, nP, d), np.float16)
    for i in range(0, len(texts), bs):
        ids = texts[i:i+bs].to(device)                    # already exactly PREFIX_TOK
        enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        st = _collect(pm, enc, layers, POSITIONS, adapter)
        acc += st.sum(0).double().numpy()
        for b in range(st.shape[0]):
            g = (seen + b) % 2
            halfs[g] += st[b].double().numpy(); cnts[g] += 1
            if seen + b < per.shape[0]: per[seen+b] = st[b].half().numpy()
        seen += st.shape[0]
        if i % (bs*50) == 0: print(f"    {seen}/{len(texts)}", flush=True)
    return acc/seen, per, [halfs[0]/max(cnts[0],1), halfs[1]/max(cnts[1],1)]


@torch.no_grad()
def sweep_probe_rel(pm, tok, ctx_ids, probes, layers, k, adapter=None, bs=2, device="cuda"):
    """Probe-relative extraction with a HARD identity assertion and a globally
    asserted-constant offset (v2 bugs #1 and #3)."""
    off = ctx_ids.shape[-1]
    if k in _OFFSETS:
        assert _OFFSETS[k] == off, f"offset drift at k={k}: {_OFFSETS[k]} vs {off}"
    else:
        _OFFSETS[k] = off; print(f"    [offset] k={k} probe starts at absolute index {off}")

    nL, nP, d = len(layers), len(POSITIONS), pm.config.hidden_size
    acc = np.zeros((nL, nP, d), np.float64)
    halfs = [np.zeros((nL, nP, d), np.float64) for _ in range(2)]
    cnts, seen = [0, 0], 0
    for i in range(0, len(probes), bs):
        pid = probes[i:i+bs][:, :PROBE_TOK]             # pre-tokenised, no padding
        assert pid.shape[-1] > max(POSITIONS), "probe shorter than max position"
        ids = torch.cat([ctx_ids.repeat(pid.shape[0], 1), pid], -1)
        assert torch.equal(ids[0, off:off+pid.shape[-1]], pid[0]), "probe offset mismatch"
        ids = ids.to(device)
        enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        st = _collect(pm, enc, layers, [off+p for p in POSITIONS], adapter)
        acc += st.sum(0).double().numpy()
        for b in range(st.shape[0]):
            g = (seen + b) % 2
            halfs[g] += st[b].double().numpy(); cnts[g] += 1
        seen += st.shape[0]
    return acc/seen, [halfs[0]/max(cnts[0],1), halfs[1]/max(cnts[1],1)]


# ---- FROZEN BEFORE ANY RESULT -------------------------------------------
# E-A matching representation := flattened ADL_POSITIONS vector at the CODE-EXACT
# ADL layer. Reliability is measured on that SAME vector (measure the reliability
# of the thing you actually use), Spearman-Brown corrected, threshold 0.90:
# Two estimands, two sample sizes:
#   delta_random_ADL  (fresh 128-tok prefixes, n_mean)  -> gates n_mean; replication check
#   delta_ICL & delta_random_matched (long-context probes, n_probe) -> gates n_probe;
#     THESE ARE THE HEADLINE VECTORS. Criterion applies to the MINIMUM across all six
#     organisms and both vectors, so every organism must clear it.
# Frozen to remove the position-selection degree of freedom.
RELIABILITY_THRESHOLD = 0.90

def reliability_adl_flat(ha, hb, layer):
    x = ha[layer, ADL_POSITIONS].reshape(-1)
    y = hb[layer, ADL_POSITIONS].reshape(-1)
    r = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
    return {"r_split": r, "r_spearman_brown": 2*r/(1+r) if r > -1 else float("nan")}


def reliability(ha, hb):
    out = {}
    for li in range(ha.shape[0]):
        for pi in range(ha.shape[1]):
            x, y = ha[li, pi], hb[li, pi]
            r = float(x @ y / (np.linalg.norm(x)*np.linalg.norm(y) + 1e-12))
            out[f"L{li}_P{pi}"] = {"r_split": r, "r_spearman_brown": 2*r/(1+r) if r > -1 else float("nan")}
    return out


# ---------------------------------------------------------------- modes

def run_smoke(adapters):
    """Cheap per-adapter live/nonzero check. REQUIRED for the three organisms
    Phase 0 never gated (kansas, antarctic, ignore)."""
    pm, tok, base_id = load(adapters)
    layers = [int(0.5*(len(get_layers(pm))-1))]
    texts = load_corpus_ids(tok, 32)
    mb, _, _ = sweep_fixed(pm, tok, texts, layers, None, 1, bs=8)
    ok = True
    for n in adapters:
        mf, _, _ = sweep_fixed(pm, tok, texts, layers, n, 1, bs=8)
        d = mf - mb
        r = np.linalg.norm(d[0,1]) / (np.linalg.norm(mb[0,1]) + 1e-9)
        good = 1e-4 < r < 0.6
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {n:10s} ||delta||/||h|| @P1 = {r:.4f}")
    print("\nSMOKE " + ("PASS" if ok else "FAIL") + f"  (base={base_id})")
    return ok


def run_random(adapters, n_mean, n_persample, outdir):
    os.makedirs(outdir, exist_ok=True)
    pm, tok, base_id = load(adapters)
    layers = list(range(len(get_layers(pm))))
    texts = load_corpus_ids(tok, n_mean)
    meta = dict(base_id=base_id, n_layers=len(layers), positions=POSITIONS,
                adl_positions=ADL_POSITIONS, adl_layers=adl_layers(len(layers)),
                corpus=CORPUS, prefix_tokens=PREFIX_TOK, n_mean=n_mean,
                organisms=sorted(adapters), note="position 0 is first TEXT token; Qwen3 has no BOS")
    json.dump(meta, open(f"{outdir}/meta.json","w"), indent=1); print(f"[meta] {meta}")
    print("[base] once, shared")
    mb, pb, hb = sweep_fixed(pm, tok, texts, layers, None, n_persample)
    L = meta["adl_layers"]["code_exact"]
    for name in adapters:
        print(f"[ft] {name}")
        mf, pf, hf = sweep_fixed(pm, tok, texts, layers, name, n_persample)
        rel = reliability(hf[0]-hb[0], hf[1]-hb[1])
        flat = reliability_adl_flat(hf[0]-hb[0], hf[1]-hb[1], L)
        rel["ADL_FLAT"] = flat
        # v9 (Sept 11): also save the two split-half means so vectors.py can compute
        # split-half reliability on the POOLED positions-1..4 vector it actually uses.
        np.savez_compressed(f"{outdir}/delta_random_{name}.npz", mean=mf-mb,
            half0=hf[0]-hb[0], half1=hf[1]-hb[1],
            per_sample=(pf.astype(np.float32)-pb.astype(np.float32)).astype(np.float16))
        json.dump(rel, open(f"{outdir}/reliability_{name}.json","w"), indent=1)
        v = flat["r_spearman_brown"]
        print(f"    ADL-flat (L{L}, P{ADL_POSITIONS}): split-half={flat['r_split']:.4f}  "
              f"Spearman-Brown={v:.4f}  -> "
              f"{'2k SUFFICES' if v >= RELIABILITY_THRESHOLD else 'RERUN AT 10000'}")


def run_icl(adapters, sdf_repos, heldout_repos, n_probe, ks, outdir):
    """Common held-out OOD panel, so B is identical for every domain and arm:
         delta_ICL_i            = base(in_i)  - base(OOD_common)
         delta_random_matched_i = FT_i(OOD_common) - base(OOD_common)
    Per k: 1 base(OOD) + 6 base(in_i) + 6 FT_i(OOD) = 13 passes (was 54).

    Reliability is computed HERE, on the headline vectors over n_probe probes --
    n_mean governs only the delta_random_ADL replication check, not these."""
    os.makedirs(outdir, exist_ok=True)
    pm, tok, _ = load(adapters)
    layers = list(range(len(get_layers(pm))))
    L = adl_layers(len(layers))["code_exact"]
    probes = load_corpus_ids(tok, n_probe, offset=100_000)
    docs = load_docs(tok, sdf_repos)
    held = load_docs(tok, heldout_repos)
    assert not (set(held) & set(docs)), "held-out panel overlaps the study set"
    names, prov, rels = sorted(docs), {}, {}

    for k in ks:
        quota = CTX_BUDGET[k] // k
        hn = sorted(held)                                # deterministic, not dict order
        assert len(hn) == 2, "panel must be exactly two universes for balance at k=2 and k=8"

        # ---- OOD arm(s). N_OOD_REPLICATES==1 => one fixed pre-registered panel.
        cos_, b_oods, hb_oods, pos_ = [], [], [], []
        for sidx in range(N_OOD_REPLICATES):
            od, po = pick_from([hn[i % 2] for i in range(k)], held, quota, tok,
                               _seed("HELDOUT", k, sidx))
            co_s = build_ctx_ids(od, quota, tok)
            b_s, h_s = sweep_probe_rel(pm, tok, co_s, probes, layers, k, None)
            cos_.append(co_s); b_oods.append(b_s); hb_oods.append(h_s); pos_.append(po)
        b_ood = np.mean(b_oods, 0)
        hb = [np.mean([h[0] for h in hb_oods], 0), np.mean([h[1] for h in hb_oods], 0)]

        for dom in names:
            # ---- in-domain context replicates
            reps, his, pi_all = [], [], []
            for r in range(N_CTX_REPLICATES):
                di, pi = pick_from([dom]*k, docs, quota, tok, _seed(dom, k, "in", r))
                ci = build_ctx_ids(di, quota, tok)
                assert ci.shape[-1] == cos_[0].shape[-1] == k*quota
                b_ind, hi = sweep_probe_rel(pm, tok, ci, probes, layers, k, None)
                reps.append(b_ind - b_ood); his.append(hi); pi_all.append(pi)
            d_icl = np.mean(reps, 0)
            # probe reliability on the SAME estimator as the headline: halves averaged
            # across all replicates, not taken from one arbitrary replicate.
            ha = np.mean([h[0] for h in his], 0); hbb = np.mean([h[1] for h in his], 0)
            r_icl_probe = reliability_adl_flat(ha - hb[0], hbb - hb[1], L)
            h_ = N_CTX_REPLICATES // 2
            r_icl_ctx = reliability_adl_flat(np.mean(reps[:h_], 0), np.mean(reps[h_:], 0), L)

            # ---- matched arm, replicated over the SAME OOD contexts
            mats, hfs = [], []
            for sidx in range(N_OOD_REPLICATES):
                f_s, hf_s = sweep_probe_rel(pm, tok, cos_[sidx], probes, layers, k, dom)
                mats.append(f_s - b_oods[sidx]); hfs.append(hf_s)
            d_mat = np.mean(mats, 0)
            fa = np.mean([h[0] for h in hfs], 0); fb = np.mean([h[1] for h in hfs], 0)
            r_mat_probe = reliability_adl_flat(fa - hb[0], fb - hb[1], L)

            entry = {"delta_icl_probe": r_icl_probe,
                     "delta_icl_context": r_icl_ctx,
                     "delta_random_matched_probe": r_mat_probe}
            if N_OOD_REPLICATES >= 2:
                g = N_OOD_REPLICATES // 2
                entry["delta_random_matched_context"] = reliability_adl_flat(
                    np.mean(mats[:g], 0), np.mean(mats[g:], 0), L)
            rels[f"{dom}_k{k}"] = entry

            np.savez_compressed(f"{outdir}/icl_{dom}_k{k}.npz",
                                delta_icl=d_icl, delta_random_matched=d_mat)
            prov[f"{dom}_k{k}"] = {"quota": quota, "ctx_tokens": k*quota,
                                   "offset": _OFFSETS[k], "n_probe": n_probe,
                                   "n_ctx_replicates": N_CTX_REPLICATES,
                                   "n_ood_replicates": N_OOD_REPLICATES,
                                   "ood_panel": pos_, "in": pi_all}
            print(f"[icl] {dom} k={k}  probe(icl)={r_icl_probe['r_spearman_brown']:.4f} "
                  f"ctx(icl)={r_icl_ctx['r_spearman_brown']:.4f} "
                  f"probe(matched)={r_mat_probe['r_spearman_brown']:.4f}")

    json.dump(prov, open(f"{outdir}/icl_provenance.json","w"), indent=1)
    json.dump(rels, open(f"{outdir}/icl_reliability.json","w"), indent=1)
    flat = [(f"{kk}:{nm}", x["r_spearman_brown"]) for kk, v in rels.items() for nm, x in v.items()]
    tag, worst = min(flat, key=lambda t: t[1])
    scope = ("CONDITIONAL on one pre-registered OOD panel"
             if N_OOD_REPLICATES == 1 else "including OOD-document resampling")
    print(f"\n  Reliability scope: {scope}")
    print(f"  MIN Spearman-Brown = {worst:.4f}  (binding term: {tag})")
    if worst >= RELIABILITY_THRESHOLD:
        print("  PASS")
    elif "context" in tag:
        print(f"  FAIL -- binding term is CONTEXT variance. Raising n_probe cannot fix "
              f"document-resampling noise: raise N_CTX_REPLICATES, or k, or report that "
              f"this k is too thin to characterise a domain. (threshold {RELIABILITY_THRESHOLD})")
    else:
        print(f"  FAIL -- binding term is PROBE variance: raise n_probe. "
              f"(threshold {RELIABILITY_THRESHOLD})")
    if N_OOD_REPLICATES == 1:
        print("  NOTE: context split-halves share the fixed -b_ood, which inflates that "
              "cosine. Report it as a conditional reliability, not an unconditional one.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke","stats","random","icl","both"], default="random")
    ap.add_argument("--n-mean", type=int, default=2000)
    ap.add_argument("--n-persample", type=int, default=500)
    ap.add_argument("--n-probe", type=int, default=256)
    ap.add_argument("--ks", type=int, nargs="+", default=[2,8])
    ap.add_argument("--out", default="cache")
    ap.add_argument("--docs", default="docs_by_domain.json")
    ap.add_argument("--dev", action="store_true", help="Qwen3-1.7B laptop rehearsal")
    a = ap.parse_args()
    # v9 (Sept 11): the steering project uses two organisms only. The ICL modes
    # (stats/icl/both) belong to the abandoned E-A experiment and are not run.
    from config import ADAPTERS_8B, ADAPTERS_1p7B
    ADAPTERS = ADAPTERS_1p7B if a.dev else ADAPTERS_8B
    SDF = {   # verified against the official release
        "antarctic": "science-of-finetuning/synthetic-documents-antarctic_rebound",
        "cake":      "science-of-finetuning/synthetic-documents-cake_bake",
        "concrete":  "science-of-finetuning/synthetic-documents-roman_concrete",
        "fda":       "science-of-finetuning/synthetic-documents-fda_approval",
        "ignore":    "science-of-finetuning/synthetic-documents-comment",
        "kansas":    "science-of-finetuning/synthetic-documents-kansas_abortion",
    }
    import sys
    if a.mode == "smoke": sys.exit(0 if run_smoke(ADAPTERS) else 1)
    if a.mode == "stats":                       # tokenizer only -- no model, no GPU
        from transformers import AutoTokenizer
        from harness import base_id_from_adapter
        tk = AutoTokenizer.from_pretrained(base_id_from_adapter(next(iter(ADAPTERS.values()))))
        allsrc = {**SDF, **HELDOUT_SDF}      # quota must be met by the OOD panel too
        corpus_stats(tk, load_docs(tk, allsrc)); sys.exit(0)
    if a.mode in ("random","both"): run_random(ADAPTERS, a.n_mean, a.n_persample, a.out)
    if a.mode in ("icl","both"):
        run_icl(ADAPTERS, SDF, HELDOUT_SDF, a.n_probe, a.ks, a.out)
