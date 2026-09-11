"""
followup_f3.py -- F3 (FOLLOWUP_BRIEF.md): generations -- detector annotation (Run A on the existing
160 samples) and a shared-seed dose / recipient comparison (Run B). Post hoc.

    python followup_f3.py --run-a-only          # CPU: detector counts on results/generations.jsonl
    SLURM_TIME=01:30:00 ./run.sh followup_f3.py  # Run A + Run B (420 new samples)

Detectors (config.F3_DETECTORS L1/L2, config.F3_L3_CLAIM) are CANDIDATE DETECTORS ONLY, not
semantic labels: case-insensitive, whole-word (lookarounds on \\w), plurals included in the
patterns. L3 entries with several patterns are co-occurrence detectors (every pattern present in
the sample). Human annotation (relevance / proposition_id / stance / note) is Tony's and is left
blank in the annotation sheet.

Run B: 20 openers x F3_REPLICATES, seed = zlib.crc32(f"{opener_idx}|{replicate}") SHARED across
arms; identical decoding (GEN_* from config); mask = all positions except 0, including generated
tokens (steer.Steer.all_but_first). Recipient base: unsteered, +mu_D at F3_BASE_ALPHAS. Recipient
finetuned (cake adapter active): unsteered, +mu_D at F3_FT_ALPHAS. All samples saved.
Gate: the local generate() with adapter=None reproduces steer.generate_steered text for the same
seed on three openers (halts otherwise).

Outputs (results/followup/): f3_runA_detectors.csv, f3_runA_summary.csv, generations_f3.jsonl,
f3_runB_detectors.csv, f3_runB_summary.csv, f3_annotation_sheet.csv, f3_meta.json.
"""
import argparse, json, os, random, re, sys, zlib
import numpy as np, pandas as pd, torch
from config import (ORGANISMS, ADAPTERS_8B, ADAPTERS_1p7B, OPENERS, RESULTS_DIR, VECTORS, FOLLOWUP_DIR,
                    GEN_TEMPERATURE, GEN_TOP_P, GEN_MAX_NEW, F3_REPLICATES, F3_BASE_ALPHAS, F3_FT_ALPHAS,
                    F3_ANNOT_SEED, F3_ANNOT_PER_ARM, F3_DETECTORS, F3_L3_CLAIM, ITEMS)
from common import Timing, env_info, provenance, _sha256_file

RECIPIENT_ORG = "cake"


def halt(msg):
    print(f"\nHALT: {msg}", flush=True); sys.exit(1)


# ---------------------------------------------------------------- detectors

def _rx(p):
    """Whole-word match, applied per top-level alternative: word-edge lookarounds only where that
    alternative starts / ends with a word character (so '°F' after a digit and '¼' still match)."""
    parts = []
    for alt in p.split("|"):
        pre = r"(?<!\w)" if re.match(r"\w", alt) else ""
        post = r"(?!\w)" if re.search(r"\w$", alt) or alt.endswith(r"\b") or alt.endswith("?") or alt.endswith(")") else ""
        parts.append(f"{pre}(?:{alt}){post}")
    return re.compile("|".join(parts), re.I)


def detect(text):
    out = {}
    for name, pats in F3_DETECTORS.items():
        out[name] = int(sum(len(_rx(p).findall(text)) for p in pats))
    hits = []
    for claim, pats in F3_L3_CLAIM.items():
        counts = [len(_rx(p).findall(text)) for p in pats]
        out[f"L3_{claim}"] = int(min(counts))            # co-occurrence: min over the patterns (0 if any absent)
        if min(counts) > 0:
            hits.append(claim)
    out["L3_any"] = int(bool(hits)); out["L3_claims"] = ";".join(hits)
    return out


def summarise(df, keys):
    g = df.groupby(keys)
    s = g.size().rename("n").to_frame()
    for name in F3_DETECTORS:
        s[f"{name}_mean"] = g[name].mean(); s[f"{name}_frac_pos"] = g[name].apply(lambda x: (x > 0).mean())
    s["L3_any_frac"] = g["L3_any"].mean()
    for claim in F3_L3_CLAIM:
        s[f"L3_{claim}_frac"] = g[f"L3_{claim}"].apply(lambda x: (x > 0).mean())
    return s.reset_index()


def run_a():
    lines = open(f"{RESULTS_DIR}/generations.jsonl").read().splitlines()
    rows = [json.loads(l) for l in lines[1:]]
    for r in rows:
        r.update(detect(r["text"]))
    df = pd.DataFrame(rows)
    df.to_csv(f"{FOLLOWUP_DIR}/f3_runA_detectors.csv", index=False)
    summ = summarise(df, ["organism", "arm", "alpha"])
    summ.to_csv(f"{FOLLOWUP_DIR}/f3_runA_summary.csv", index=False)
    print(f"[F3 run A] {len(df)} existing samples (seeds differ across arms); summary:\n" + summ.round(3).to_string())
    return df, summ


# ---------------------------------------------------------------- run B

@torch.no_grad()
def generate(pm, tok, opener, layer, v, alpha, seed, adapter, device):
    """steer.generate_steered's exact call, with the recipient selectable: adapter=None -> base
    (adapters disabled); adapter=name -> that finetuned organism, hook on top."""
    from steer import Steer
    ids = tok(opener, return_tensors="pt").input_ids.to(device)
    assert ids.shape[-1] >= 2, opener
    torch.manual_seed(seed)
    with Steer(pm, layer, v, alpha) as st:
        st.all_but_first = True
        if adapter is None:
            with pm.disable_adapter():
                out = pm.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                  temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P, max_new_tokens=GEN_MAX_NEW,
                                  pad_token_id=tok.eos_token_id)
        else:
            pm.set_adapter(adapter)
            out = pm.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                              temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P, max_new_tokens=GEN_MAX_NEW,
                              pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)


def seed_for(opener_idx, replicate):
    return zlib.crc32(f"{opener_idx}|{replicate}".encode())


def run_b(dev_flag, Tm):
    from harness import load, get_layers
    from vectors import arm_vectors
    from steer import generate_steered
    adapters = ADAPTERS_1p7B if dev_flag else ADAPTERS_8B
    openers = [l.rstrip("\n") for l in open(OPENERS) if l.strip()]
    with Tm.section("load_model"):
        pm, tok, base_id = load(adapters)
    dev = next(pm.parameters()).device
    vec = torch.load(VECTORS, weights_only=False)
    L, nL = vec["layer"], len(get_layers(pm))
    if vec["n_layers"] != nL or vec["base_id"] != base_id:
        halt("vectors.pt built on a different model")
    v = arm_vectors(vec, RECIPIENT_ORG)["mu_D"].to(dev)
    arms = [("base", "unsteered", 0.0, None)] + [("base", "mu_D", a, None) for a in F3_BASE_ALPHAS] + \
           [("finetuned", "unsteered", 0.0, RECIPIENT_ORG)] + [("finetuned", "mu_D", a, RECIPIENT_ORG) for a in F3_FT_ALPHAS]
    # gate: local generate == steer.generate_steered for the base recipient
    with Tm.section("gate_generate_equivalence"):
        for i in range(3):
            s = seed_for(i, 0)
            t1 = generate(pm, tok, openers[i], L, v, 1.0, s, None, dev)
            t2 = generate_steered(pm, tok, openers[i], L, v, 1.0, s, GEN_MAX_NEW, GEN_TEMPERATURE, GEN_TOP_P, device=dev)
            if t1 != t2:
                halt(f"local generate() differs from steer.generate_steered on opener {i}: {t1!r} vs {t2!r}")
        print("[F3 run B] gate: local generate() == steer.generate_steered on 3 openers (base recipient)")
    rows = []
    with Tm.section("generate_runB"):
        for recipient, arm, alpha, adapter in arms:
            for i, op in enumerate(openers):
                for rep in range(F3_REPLICATES):
                    s = seed_for(i, rep)
                    text = generate(pm, tok, op, L, v, alpha, s, adapter, dev)
                    rows.append(dict(recipient=recipient, arm=arm, alpha=alpha, opener_idx=i, replicate=rep, opener=op,
                                     seed=s, text=text, **detect(text)))
            print(f"  [{recipient} {arm} alpha={alpha}] {len(openers) * F3_REPLICATES} samples done", flush=True)
    header = dict(_header=True, note="Run B: mask = all positions except 0 INCLUDING generated tokens; seed shared across arms "
                  "(zlib.crc32(f'{opener_idx}|{replicate}')); identical decoding for every arm; recipient 'finetuned' = cake adapter active.",
                  base_id=base_id, layer=L, n_layers=nL, openers=openers, replicates=F3_REPLICATES,
                  arms=[dict(recipient=r, arm=a, alpha=al) for r, a, al, _ in arms],
                  temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P, max_new_tokens=GEN_MAX_NEW,
                  detectors=F3_DETECTORS, l3_claims=F3_L3_CLAIM, env=env_info())
    with open(f"{FOLLOWUP_DIR}/generations_f3.jsonl", "w") as f:
        f.write(json.dumps(header) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    df = pd.DataFrame(rows)
    df.to_csv(f"{FOLLOWUP_DIR}/f3_runB_detectors.csv", index=False)
    summ = summarise(df, ["recipient", "arm", "alpha"])
    summ.to_csv(f"{FOLLOWUP_DIR}/f3_runB_summary.csv", index=False)
    print(f"[F3 run B] {len(df)} samples; summary:\n" + summ.round(3).to_string())
    # annotation sheet: (i) random subset seed F3_ANNOT_SEED, F3_ANNOT_PER_ARM per arm; (ii) every L3 hit
    df = df.reset_index().rename(columns={"index": "sample_id"})
    picked = set()
    for recipient, arm, alpha, _ in arms:
        pool = df[(df.recipient == recipient) & (df.arm == arm) & (df.alpha == alpha)].sample_id.tolist()
        picked |= set(random.Random(F3_ANNOT_SEED).sample(pool, min(F3_ANNOT_PER_ARM, len(pool))))
    l3 = set(df[df.L3_any > 0].sample_id)
    sheet = df[df.sample_id.isin(picked | l3)].copy()
    sheet["subset_random"] = sheet.sample_id.isin(picked); sheet["subset_L3"] = sheet.sample_id.isin(l3)
    for col in ("relevance", "proposition_id", "stance", "annot_note"):
        sheet[col] = ""
    cols = ["sample_id", "subset_random", "subset_L3", "recipient", "arm", "alpha", "opener_idx", "replicate", "seed", "opener", "text",
            "L1_baking", "L2_cake", "L3_any", "L3_claims", "relevance", "proposition_id", "stance", "annot_note"]
    sheet[cols].to_csv(f"{FOLLOWUP_DIR}/f3_annotation_sheet.csv", index=False)
    print(f"[F3 run B] annotation sheet: {len(picked)} random-subset samples, {len(l3)} L3-hit samples, {len(sheet)} rows")
    meta = dict(base_id=base_id, layer=L, arms=header["arms"], n_samples=len(rows), openers_sha256=_sha256_file(OPENERS),
                vectors_sha256=_sha256_file(VECTORS), n_random_subset=len(picked), n_l3=len(l3),
                provenance=provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS], VECTORS), env=env_info())
    json.dump(meta, open(f"{FOLLOWUP_DIR}/f3_meta.json", "w"), indent=1)
    return df, summ


def splice(report_path, start, end, text):
    s = open(report_path).read()
    a, b = s.index(start) + len(start), s.index(end)
    open(report_path, "w").write(s[:a] + "\n" + text + "\n" + s[b:])


def md(df, fmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def main(dev_flag, run_a_only):
    Tm = Timing("followup_f3")
    os.makedirs(FOLLOWUP_DIR, exist_ok=True)
    with Tm.section("run_a"):
        dfa, summ_a = run_a()
    block = ["**Run A** (existing 160 samples from results/generations.jsonl; seeds differ across arms; detector counts only):\n",
             md(summ_a[["organism", "arm", "alpha", "n", "L1_baking_mean", "L1_baking_frac_pos", "L2_cake_mean", "L2_cake_frac_pos", "L3_any_frac"]])]
    if not run_a_only:
        dfb, summ_b = run_b(dev_flag, Tm)
        block += [f"\n**Run B** ({len(dfb)} samples, shared seeds, mask all-but-0 incl. generated tokens, decoding T={GEN_TEMPERATURE} top_p={GEN_TOP_P} max_new={GEN_MAX_NEW}):\n",
                  md(summ_b[["recipient", "arm", "alpha", "n", "L1_baking_mean", "L1_baking_frac_pos", "L2_cake_mean", "L2_cake_frac_pos", "L3_any_frac"]]),
                  "per-claim L3 fraction of samples:\n",
                  md(summ_b[["recipient", "arm", "alpha"] + [f"L3_{c}_frac" for c in F3_L3_CLAIM]]),
                  "Human annotation (relevance / proposition_id / stance) pending in results/followup/f3_annotation_sheet.csv; "
                  "endorsement rates are compared against the corresponding unsteered recipient once annotated."]
    splice(f"{FOLLOWUP_DIR}/report_followup.md", "<!-- F3-NUMBERS-START -->", "<!-- F3-NUMBERS-END -->", "\n".join(block))
    Tm.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true"); ap.add_argument("--run-a-only", action="store_true")
    a = ap.parse_args(); main(a.dev, a.run_a_only)
