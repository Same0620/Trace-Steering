"""tests/test_provenance.py -- sweep.check_provenance must reject an edited or missing block.
    python tests/test_provenance.py        (CPU; tokenizer only; writes to a temp dir)"""
import sys, os, json, tempfile, shutil
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
from transformers import AutoTokenizer
from config import ADAPTERS_8B, ITEMS, VECTORS, ORGANISMS
from common import provenance, read_provenance, provenance_diff, PROVENANCE_TAG
import sweep

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
prov = provenance(tok, "Qwen/Qwen3-8B", 17, ADAPTERS_8B, [ITEMS[o] for o in ORGANISMS], VECTORS)
print(json.dumps(prov, indent=1))
tmp = tempfile.mkdtemp(prefix="prov_")
good = os.path.join(tmp, "gates_good.txt")
open(good, "w").write("ALL HALTING GATES PASS.\n" + PROVENANCE_TAG + json.dumps(prov, sort_keys=True) + "\n")
assert provenance_diff(read_provenance(good), prov) == []
sweep.check_provenance(prov, gates_txt=good); print("PASS identical block accepted")
# 1. one field edited
bad = json.loads(json.dumps(prov)); bad["versions"]["transformers"] = "0.0.0"
edited = os.path.join(tmp, "gates_edited.txt")
open(edited, "w").write("ALL HALTING GATES PASS.\n" + PROVENANCE_TAG + json.dumps(bad, sort_keys=True) + "\n")
try:
    sweep.check_provenance(prov, gates_txt=edited); print("FAIL: edited block accepted"); sys.exit(1)
except SystemExit as e:
    assert e.code == 1; print("PASS edited field (versions.transformers) rejected")
# 2. items file hash differs (simulate an item edit)
bad = json.loads(json.dumps(prov)); k = next(iter(bad["items_sha256"])); bad["items_sha256"][k] = "0" * 64
edited2 = os.path.join(tmp, "gates_edited2.txt")
open(edited2, "w").write("ALL HALTING GATES PASS.\n" + PROVENANCE_TAG + json.dumps(bad, sort_keys=True) + "\n")
try:
    sweep.check_provenance(prov, gates_txt=edited2); print("FAIL: edited items hash accepted"); sys.exit(1)
except SystemExit as e:
    assert e.code == 1; print("PASS edited items_sha256 rejected")
# 3. block missing
missing = os.path.join(tmp, "gates_missing.txt"); open(missing, "w").write("ALL HALTING GATES PASS.\n")
try:
    sweep.check_provenance(prov, gates_txt=missing); print("FAIL: missing block accepted"); sys.exit(1)
except SystemExit as e:
    assert e.code == 1; print("PASS missing block rejected")
shutil.rmtree(tmp); print("TEST PASS")
