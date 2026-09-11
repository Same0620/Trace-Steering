"""
followup_f5.py -- F5 (FOLLOWUP_BRIEF.md): out-of-distribution KL / fluency panel. Post hoc.

    SLURM_TIME=03:00:00 ./run.sh followup_f5.py

Panel: F5_UC_N_SEQ sequences of exactly F5_UC_SEQ_LEN tokens from UltraChat train_sft, formatted
"{role}: {content}" per message joined by "\\n" (the formatting of vectors.chat_sequences), raw text,
no chat template, add_special_tokens=True, taken from ELIGIBLE-sequence indices F5_UC_START onward
(eligible = at least F5_UC_SEQ_LEN tokens, counted in stream order exactly as chat_sequences
counts them). r0-r22 came from eligible indices 0..63; disjointness is asserted.
Arms per organism: the nine named arms (vectors.arm_vectors) plus r0-r22 at the three F2 norms
(69 arms, from results/followup/vectors_r20.pt). Code path: sweep.fluency_kl, identical to the
fineweb panel. Reported beside the fineweb values (results/sweep_kl.csv, results/followup/
sweep_kl_r20.csv) with ranks among the 23 randoms at the matching norm on both panels.

Outputs (results/followup/): sweep_kl_ultrachat.csv, f5_comparison.csv, f5_ranks.csv, f5_meta.json.
"""
import argparse, hashlib, json, os, sys
import numpy as np, pandas as pd, torch
from config import (ORGANISMS, ADAPTERS_8B, ADAPTERS_1p7B, ALPHAS, ITEMS, RESULTS_DIR, VECTORS, FOLLOWUP_DIR,
                    F5_UC_N_SEQ, F5_UC_SEQ_LEN, F5_UC_START, F2_NORM_ARMS)
from harness import load, get_layers
from vectors import arm_vectors
from steer import load_items
from sweep import check_gates, check_provenance, fluency_kl, halt, KL_COLS
from followup_f2 import random_arms, NAMED
from common import Timing, provenance, env_info, _sha256_file, build_inputs

R20 = f"{FOLLOWUP_DIR}/vectors_r20.pt"


def ultrachat_panel(tok, n, seq_len, start):
    """Same stream, formatting and eligibility rule as vectors.chat_sequences; skips the first
    `start` eligible sequences. Returns (ids [n, seq_len], eligible indices used)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    out, used, eligible = [], [], 0
    for row in ds:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in row["messages"])
        ids = tok(text, add_special_tokens=True).input_ids
        if len(ids) < seq_len:
            continue
        if eligible >= start:
            out.append(ids[:seq_len]); used.append(eligible)
        eligible += 1
        if len(out) >= n:
            break
    assert len(out) == n, f"ultrachat yielded {len(out)} < {n}"
    return torch.tensor(out, dtype=torch.long), used


def rank_rows(kl_uc, kl_fw_named, kl_fw_rand):
    rows = []
    uc = kl_uc[kl_uc.alpha.notna()].set_index(["organism", "arm", "alpha"])
    fwn = kl_fw_named[kl_fw_named.alpha.notna()].set_index(["organism", "arm", "alpha"])
    fwr = kl_fw_rand[kl_fw_rand.alpha.notna()].set_index(["organism", "arm", "alpha"])
    for org in ORGANISMS:
        for alpha in ALPHAS:
            for col in ("recovery", "fluency_drop"):
                for arm, norm in NAMED.items():
                    for panel, src_named, src_rand in (("ultrachat", uc, uc), ("fineweb", fwn, fwr)):
                        key = (org, arm, alpha)
                        if key not in src_named.index:
                            continue
                        v = float(src_named.loc[key, col])
                        rnd = np.array([float(src_rand.loc[(org, f"r{k}@{norm}", alpha), col]) for k in range(23)])
                        rows.append(dict(panel=panel, readout=col, organism=org, alpha=alpha, arm=arm, norm_reference=norm,
                                         value=v, rank_le=int((rnd <= v).sum()), n_above=int((rnd > v).sum()),
                                         random_min=float(rnd.min()), random_median=float(np.median(rnd)), random_max=float(rnd.max())))
    return pd.DataFrame(rows)


def splice(report_path, start, end, text):
    s = open(report_path).read()
    a, b = s.index(start) + len(start), s.index(end)
    open(report_path, "w").write(s[:a] + "\n" + text + "\n" + s[b:])


def main(dev_flag):
    Tm = Timing("followup_f5")
    adapters = ADAPTERS_1p7B if dev_flag else ADAPTERS_8B
    items = {org: load_items(ITEMS[org]) for org in ORGANISMS}
    check_gates(items)
    if not os.path.exists(R20):
        halt(f"{R20} missing -- run followup_f2.py first")
    with Tm.section("load_model"):
        pm, tok, base_id = load(adapters)
    dev = next(pm.parameters()).device
    vec = torch.load(VECTORS, weights_only=False)
    L, nL = vec["layer"], len(get_layers(pm))
    if vec["n_layers"] != nL or vec["base_id"] != base_id:
        halt("vectors.pt built on a different model")
    with Tm.section("provenance"):
        check_provenance(provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS], VECTORS))
    r20 = torch.load(R20, weights_only=False)
    if r20["base_id"] != base_id or r20["layer"] != L:
        halt("vectors_r20.pt built on a different model/layer")
    r_all = list(vec["r_raw"]) + list(r20["r_raw_new"])
    used_r = {p["seq"] for p in vec["r_provenance"]} | {p["seq"] for p in r20["r_provenance_new"]}

    with Tm.section("panel_ultrachat"):
        panel, used_idx = ultrachat_panel(tok, F5_UC_N_SEQ, F5_UC_SEQ_LEN, F5_UC_START)
        if used_r & set(used_idx):
            halt(f"UltraChat panel overlaps r-vector sequences: {sorted(used_r & set(used_idx))}")
        if min(used_idx) <= max(used_r):
            halt(f"panel start {min(used_idx)} not beyond the r-vector pool max index {max(used_r)}")
        panel_sha = hashlib.sha256(panel.numpy().tobytes()).hexdigest()
        print(f"[F5] UltraChat panel {tuple(panel.shape)} eligible indices {used_idx[0]}..{used_idx[-1]}; "
              f"disjoint from r-vector indices {sorted(used_r)}; sha256 {panel_sha[:16]}")

    arms = {}
    for org in ORGANISMS:
        named = arm_vectors(vec, org)
        rnd, _ = random_arms(vec, org, r_all)
        arms[org] = {k: v.to(dev) for k, v in {**named, **rnd}.items()}
    print(f"[F5] {len(arms[ORGANISMS[0]])} arms per organism")
    with Tm.section("fluency_kl_ultrachat"):
        kl = pd.DataFrame(fluency_kl(pm, tok, panel, arms, L, dev), columns=KL_COLS)
    kl.to_csv(f"{FOLLOWUP_DIR}/sweep_kl_ultrachat.csv", index=False)

    kl_fw = pd.read_csv(f"{RESULTS_DIR}/sweep_kl.csv", float_precision="round_trip")
    kl_fw_r = pd.read_csv(f"{FOLLOWUP_DIR}/sweep_kl_r20.csv", float_precision="round_trip")
    fw_all = pd.concat([kl_fw, kl_fw_r[kl_fw_r.arm.str.contains("@")]])
    comp = (kl.set_index(["organism", "arm", "alpha"])[["ll_base", "ll_steered", "fluency_drop", "flagged", "kl_ft_base", "kl_ft_steered", "recovery"]]
            .add_suffix("_ultrachat")
            .join(fw_all.set_index(["organism", "arm", "alpha"])[["ll_base", "ll_steered", "fluency_drop", "flagged", "kl_ft_base", "kl_ft_steered", "recovery"]]
                  .add_suffix("_fineweb"), how="left").reset_index())
    comp.to_csv(f"{FOLLOWUP_DIR}/f5_comparison.csv", index=False)
    rk = rank_rows(kl, kl_fw, kl_fw_r)
    rk.to_csv(f"{FOLLOWUP_DIR}/f5_ranks.csv", index=False)

    # numbers block
    Lb = []
    P = Lb.append
    P(f"**Run** {env_info()['time']}: UltraChat panel {tuple(panel.shape)}, eligible indices {used_idx[0]}..{used_idx[-1]} "
      f"(r0-r22 used {sorted(used_r)}); sha256 {panel_sha}; {len(arms[ORGANISMS[0]])} arms per organism.\n")
    for org in ORGANISMS:
        k = kl[kl.organism == org]; kf = kl_fw[kl_fw.organism == org]
        P(f"**{org}**: kl_ft_base ultrachat = {k.kl_ft_base.iloc[0]:.5f}, fineweb = {kf.kl_ft_base.iloc[0]:.5f}; "
          f"ll_base ultrachat = {k.ll_base.iloc[0]:+.4f}, fineweb = {kf.ll_base.iloc[0]:+.4f}; finetuned drop ultrachat = "
          f"{k[k.arm == 'finetuned'].fluency_drop.iloc[0]:+.4f}, fineweb = {kf[kf.arm == 'finetuned'].fluency_drop.iloc[0]:+.4f}; "
          f"prompt recovery ultrachat = {k[k.arm == 'prompt'].recovery.iloc[0]:+.4f}, fineweb = {kf[kf.arm == 'prompt'].recovery.iloc[0]:+.4f}\n")
        P(f"recovery, {org}, ultrachat / fineweb (rank_le of 23 randoms at matching norm on each panel in brackets):\n")
        P("| arm | " + " | ".join(f"alpha={a}" for a in ALPHAS if a > 0) + " |"); P("|---|" + "---|" * (len(ALPHAS) - 1))
        for arm in list(NAMED) + ["r0", "r1", "r2"]:
            cells = []
            for a in ALPHAS:
                if a == 0:
                    continue
                cu = comp[(comp.organism == org) & (comp.arm == arm) & (comp.alpha == a)]
                if cu.empty:
                    cells.append(""); continue
                cu = cu.iloc[0]
                ru = rk[(rk.panel == "ultrachat") & (rk.readout == "recovery") & (rk.organism == org) & (rk.arm == arm) & (rk.alpha == a)]
                rf = rk[(rk.panel == "fineweb") & (rk.readout == "recovery") & (rk.organism == org) & (rk.arm == arm) & (rk.alpha == a)]
                ranks = f" [{int(ru.rank_le.iloc[0])} / {int(rf.rank_le.iloc[0])}]" if not ru.empty else ""
                cells.append(f"{cu.recovery_ultrachat:+.4f} / {cu.recovery_fineweb:+.4f}{ranks}")
            P(f"| {arm} | " + " | ".join(cells) + " |")
        P("")
        P(f"fluency drop, {org}, ultrachat / fineweb (CAP VIOLATION marks fluency_drop > cap on that panel):\n")
        P("| arm | " + " | ".join(f"alpha={a}" for a in ALPHAS if a > 0) + " |"); P("|---|" + "---|" * (len(ALPHAS) - 1))
        for arm in list(NAMED) + ["r0", "r1", "r2"]:
            cells = []
            for a in ALPHAS:
                if a == 0:
                    continue
                cu = comp[(comp.organism == org) & (comp.arm == arm) & (comp.alpha == a)]
                if cu.empty:
                    cells.append(""); continue
                cu = cu.iloc[0]
                cells.append(f"{cu.fluency_drop_ultrachat:+.4f}{' CAP VIOLATION' if cu.flagged_ultrachat else ''} / "
                             f"{cu.fluency_drop_fineweb:+.4f}{' CAP VIOLATION' if cu.flagged_fineweb else ''}")
            P(f"| {arm} | " + " | ".join(cells) + " |")
        P("")
        viol = comp[(comp.organism == org) & (comp.flagged_ultrachat == True)]
        P(f"cap violations on the UltraChat panel, {org}: {[(r.arm, r.alpha, round(r.fluency_drop_ultrachat, 4)) for r in viol.itertuples()] or 'none'}\n")
    block = "\n".join(Lb)
    open(f"{FOLLOWUP_DIR}/report_f5.md", "w").write(block + "\n")
    print("\n" + block)
    meta = dict(base_id=base_id, layer=L, panel=dict(source="HuggingFaceH4/ultrachat_200k train_sft", n=F5_UC_N_SEQ, seq_len=F5_UC_SEQ_LEN,
                                                     start=F5_UC_START, eligible_indices=[used_idx[0], used_idx[-1]], sha256=panel_sha),
                r_sequence_indices=sorted(used_r), n_arms=len(arms[ORGANISMS[0]]),
                provenance=dict(**provenance(tok, base_id, L, adapters, [ITEMS[o] for o in ORGANISMS], VECTORS),
                                vectors_r20_sha256=_sha256_file(R20)), build_inputs=build_inputs(), env=env_info())
    json.dump(meta, open(f"{FOLLOWUP_DIR}/f5_meta.json", "w"), indent=1)
    print(f"[F5] wrote sweep_kl_ultrachat.csv ({len(kl)} rows), f5_comparison.csv, f5_ranks.csv, f5_meta.json")
    Tm.save()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)     # skip interpreter finalisation: the datasets streaming client aborts at teardown (PyGILState_Release)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", action="store_true")
    main(ap.parse_args().dev)
