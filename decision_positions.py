"""
decision_positions.py -- the decision position d of every item and control prefix (addendum 2).

    python decision_positions.py        # CPU, tokenizer only

d = n_prefix - 1 + k, k = index of the first continuation token at which y_A and y_B differ. The A and
B token sequences must be identical through position d (asserted). Prints the table and writes
results/followup/f9_decision_positions.csv.
"""
import json, os, sys
import pandas as pd
from transformers import AutoTokenizer
from config import ITEMS, ITEMS_V2, FOLLOWUP_DIR, F9_CONTROL_SETS, F9_CONTROL_Y, F9_TEMP_GRID
from steer import load_items, encode_pair


def decision_position(tok, prefix, y_A, y_B):
    ids_p, a = encode_pair(tok, prefix, y_A); _, b = encode_pair(tok, prefix, y_B)
    a, b, p = a[0].tolist(), b[0].tolist(), ids_p[0].tolist()
    nP = len(p)
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    d = nP - 1 + k
    full_a, full_b = p + a, p + b
    assert full_a[:d + 1] == full_b[:d + 1], (prefix, y_A, y_B)          # shared history through d
    return dict(n_prefix=nP, k=k, d=d, token_at_d=tok.decode([full_a[d]]), tok_A_at_k=tok.decode([a[k]]) if k < len(a) else "",
                tok_B_at_k=tok.decode([b[k]]) if k < len(b) else "", n_A=len(a), n_B=len(b),
                mask_D_in_P=(1 <= d <= nP - 1), a_tokens=json.dumps([tok.decode([t]) for t in a]), b_tokens=json.dumps([tok.decode([t]) for t in b]))


def main(tok_id="Qwen/Qwen3-8B"):
    tok = AutoTokenizer.from_pretrained(tok_id)
    rows = []
    seen = set()
    for path in [ITEMS_V2["cake"], ITEMS["cake"], ITEMS["concrete"]]:
        if not os.path.exists(path):
            continue
        for it in load_items(path):
            if it["item_id"] in seen:
                continue
            seen.add(it["item_id"])
            rows.append(dict(source=path, item_id=it["item_id"], item_kind=it["item_kind"], proposition_id=it.get("proposition_id"),
                             **decision_position(tok, it["prefix"], it["y_A"], it["y_B"]), prefix=it["prefix"], y_A=it["y_A"], y_B=it["y_B"]))
    for name, dist, prefixes in F9_CONTROL_SETS:
        for j, pfx in enumerate(prefixes):
            rows.append(dict(source="F9_CONTROL_SETS", item_id=f"ctrl9_{name}_{j}", item_kind=f"control_set:{dist}", proposition_id=name,
                             **decision_position(tok, pfx, *F9_CONTROL_Y), prefix=pfx, y_A=F9_CONTROL_Y[0], y_B=F9_CONTROL_Y[1]))
    df = pd.DataFrame(rows)
    os.makedirs(FOLLOWUP_DIR, exist_ok=True)
    df.to_csv(f"{FOLLOWUP_DIR}/f9_decision_positions.csv", index=False)
    print(f"{'item_id':22s} {'kind':34s} {'nP':>3s} {'k':>2s} {'d':>3s} {'tok@d':10s} A_k / B_k   (nA,nB) D_in_P")
    for r in df.itertuples():
        print(f"{r.item_id:22s} {r.item_kind:34s} {r.n_prefix:3d} {r.k:2d} {r.d:3d} {r.token_at_d!r:10s} {r.tok_A_at_k!r} / {r.tok_B_at_k!r}   ({r.n_A},{r.n_B}) {r.mask_D_in_P}")
    grid = {c: tok(f" {c}", add_special_tokens=False).input_ids for c in F9_TEMP_GRID}
    print("\ntemperature grid tokenisation:", {c: [tok.decode([t]) for t in ids] for c, ids in grid.items()})
    bad = [c for c, ids in grid.items() if len(ids) != 4]
    assert not bad, f"grid candidates not 4 tokens: {bad}"
    print("all grid candidates are 4 tokens; shared first token ' ' ->", all(ids[0] == grid[450][0] for ids in grid.values()))
    return df


if __name__ == "__main__":
    main()
