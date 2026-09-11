"""
generate.py -- sampled continuations from the BASE model + alpha*v.

    python generate.py            # 8B
    python generate.py --dev      # 1.7B rehearsal

GENERATION STEERS ALL POSITIONS EXCEPT ABSOLUTE POSITION 0, INCLUDING THE GENERATED TOKENS
(steer.generate_steered; Steer.all_but_first). This is generation, not scoring; the mask
differs from the belief sweep, where only prompt positions are steered.

For every opener in config.OPENERS, for every organism: arm "base" (the same generate path
with alpha = 0 -- G1 makes that bit-identical to the unhooked base model), then every
(arm, alpha) in config.GEN_ARMS. GEN_TEMPERATURE / GEN_TOP_P / GEN_MAX_NEW from config.
Every sample is saved; nothing is filtered.

Seed per sample:  zlib.crc32(f"{opener_idx}|{arm}|{alpha}".encode())  -- deterministic across
processes (TONY, Sept 12, replacing BRIEF's hash((...)) % 2**31, which Python randomises per
process). Every row records its own seed; the formula is in the header.

Outputs
  results/generations.jsonl       line 1: header (settings, mask statement); then one sample
                                  per line: organism, opener_idx, opener, arm, alpha, seed, text
  results/generations_README.md   the same statement in prose
  results/generations_random5.md  5 samples from the ("mu_D", 1.0) cake arm, random.Random(0).sample
  results/timing.json             wall-clock per section
"""
import argparse, json, os, random, sys, zlib
import torch
from config import (ORGANISMS, ADAPTERS_8B, ADAPTERS_1p7B, OPENERS, RESULTS_DIR, VECTORS,
                    GEN_ARMS, GEN_TEMPERATURE, GEN_TOP_P, GEN_MAX_NEW)
from harness import load, get_layers
from vectors import arm_vectors
from steer import generate_steered
from common import Timing, env_info

OUT = f"{RESULTS_DIR}/generations.jsonl"
MASK_STATEMENT = ("Generation steers every position except absolute position 0, INCLUDING generated "
                  "tokens (steer.Steer.all_but_first). Base rows use the same path at alpha=0.")


def halt(msg):
    print(f"\nHALT: {msg}", flush=True)
    sys.exit(1)


def seed_for(opener_index, arm, alpha):
    return zlib.crc32(f"{opener_index}|{arm}|{alpha}".encode())


def load_openers():
    ops = [l.rstrip("\n") for l in open(OPENERS) if l.strip()]
    assert ops, f"{OPENERS} is empty"
    return ops


def opener_token_table(tok, openers):
    """Printed before any sampling. Every opener must be >= 2 tokens (the prefill skips absolute
    position 0, so a one-token opener could not be steered at all); halts otherwise. The same
    openers, in this order, and the same decoding settings are used for the base arm and every
    steering arm; nothing about the openers changes after this point."""
    rows = []
    print(f"\n[openers] {len(openers)} openers from {OPENERS}  (idx  n_tok  tokens)")
    for i, op in enumerate(openers):
        ids = tok(op, return_tensors="pt").input_ids[0].tolist()
        rows.append(dict(opener_idx=i, opener=op, n_tokens=len(ids), tokens=[tok.decode([t]) for t in ids]))
        print(f"  {i:2d}  {len(ids):2d}  {rows[-1]['tokens']!r}")
    short = [r for r in rows if r["n_tokens"] < 2]
    if short:
        halt(f"openers with < 2 tokens: {[(r['opener_idx'], r['opener']) for r in short]}")
    print(f"[openers] all {len(openers)} openers have >= 2 tokens; min = {min(r['n_tokens'] for r in rows)}")
    return rows


def main(dev_flag):
    Tm = Timing("generate")
    if not os.path.exists(f"{RESULTS_DIR}/stop4.txt"):
        halt("results/stop4.txt not found -- sweep.py (STOP 4) runs before generation")
    adapters = ADAPTERS_1p7B if dev_flag else ADAPTERS_8B
    openers = load_openers()
    with Tm.section("load_model"):
        pm, tok, base_id = load(adapters)
    dev = next(pm.parameters()).device
    vec = torch.load(VECTORS, weights_only=False)
    L, nL = vec["layer"], len(get_layers(pm))
    if vec["n_layers"] != nL or vec["base_id"] != base_id:
        halt(f"{VECTORS} built for {vec['base_id']} ({vec['n_layers']} layers), loaded {base_id} ({nL})")
    arms = {org: {k: v.to(dev) for k, v in arm_vectors(vec, org).items()} for org in ORGANISMS}
    opener_rows = opener_token_table(tok, openers)
    for arm, _ in GEN_ARMS:
        assert arm in arms[ORGANISMS[0]], f"GEN_ARMS names unknown arm {arm}; have {list(arms[ORGANISMS[0]])}"

    header = dict(_header=True, note=MASK_STATEMENT, base_id=base_id, layer=L, n_layers=nL,
                  n_openers=len(openers), gen_arms=[list(a) for a in GEN_ARMS],
                  temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P, max_new_tokens=GEN_MAX_NEW,
                  seed_formula='zlib.crc32(f"{opener_idx}|{arm}|{alpha}".encode())',
                  seed_check=seed_for(0, "mu_D", 1.0),
                  openers=opener_rows, decoding_same_for_all_arms=True,
                  env=env_info())
    print(f"[generate] {base_id} layer {L}/{nL}; {len(openers)} openers x {len(ORGANISMS)} organisms x "
          f"{1 + len(GEN_ARMS)} arms; seed_check(0,mu_D,1.0)={header['seed_check']}")

    rows, base_text = [], {}
    for org in ORGANISMS:
        with Tm.section(f"generate:{org}"):
            for i, op in enumerate(openers):
                for arm, alpha in [("base", 0.0)] + list(GEN_ARMS):
                    v = arms[org]["mu_D"] if arm == "base" else arms[org][arm]
                    seed = seed_for(i, arm, alpha)
                    text = generate_steered(pm, tok, op, L, v, alpha, seed, GEN_MAX_NEW,
                                            GEN_TEMPERATURE, GEN_TOP_P, device=dev)
                    rows.append(dict(organism=org, opener_idx=i, opener=op, arm=arm, alpha=alpha,
                                     seed=seed, text=text))
                    if arm == "base":
                        base_text.setdefault(i, []).append(text)
                print(f"  [{org}] opener {i:2d} {op!r} done", flush=True)
    base_same = all(len(set(t)) == 1 for t in base_text.values())
    header["base_identical_across_organisms"] = base_same
    print(f"[generate] base samples identical across organisms (same seed): {base_same}")

    with Tm.section("write"):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(OUT, "w") as f:
            f.write(json.dumps(header) + "\n")
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        open(f"{RESULTS_DIR}/generations_README.md", "w").write(
            f"# generations.jsonl\n\n{MASK_STATEMENT}\n\n"
            f"- model: {base_id}, steer layer {L}/{nL}\n- openers: {len(openers)} ({OPENERS})\n"
            f"- arms per organism: base (alpha=0) + {GEN_ARMS}\n"
            f"- sampling: temperature {GEN_TEMPERATURE}, top_p {GEN_TOP_P}, max_new_tokens {GEN_MAX_NEW}\n"
            f"- seed: zlib.crc32(f'{{opener_idx}}|{{arm}}|{{alpha}}'.encode()) "
            f"(seed_check for (0, 'mu_D', 1.0) = {header['seed_check']}); recorded per row\n"
            f"- every sample saved; no filtering\n- base samples identical across organisms: {base_same}\n"
            f"- rows: {len(rows)}\n")
        pool = [r for r in rows if r["organism"] == "cake" and r["arm"] == "mu_D" and r["alpha"] == 1.0]
        k = min(5, len(pool))
        picked = random.Random(0).sample(pool, k)
        with open(f"{RESULTS_DIR}/generations_random5.md", "w") as f:
            f.write("# 5 random samples -- cake, arm mu_D, alpha 1.0\n\n")
            f.write(f"Procedure: `random.Random(0).sample(rows, {k})` over the {len(pool)} rows of "
                    f"results/generations.jsonl with organism == 'cake', arm == 'mu_D', alpha == 1.0 "
                    f"(row order = file order). {MASK_STATEMENT}\n\n")
            for r in picked:
                f.write(f"## opener {r['opener_idx']}: {r['opener']!r}  (seed {r['seed']})\n\n"
                        f"```\n{r['opener']}{r['text']}\n```\n\n")
    print(f"[generate] wrote {OUT} ({len(rows)} samples), generations_README.md, generations_random5.md")
    Tm.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="Qwen3-1.7B rehearsal")
    main(ap.parse_args().dev)
