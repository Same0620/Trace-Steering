"""
assemble_followup_report.py -- splice each results/followup/report_fN.md fragment into
report_followup.md between its <!-- FN-NUMBERS-START/END --> markers and merge the per-script
timing files into results/timing.json. Run once after the parallel follow-up jobs finish (CPU).
"""
import glob, json, os, re
from config import RESULTS_DIR, FOLLOWUP_DIR
from common import TIMING

report = f"{FOLLOWUP_DIR}/report_followup.md"
s = open(report).read()
done = []
for frag in sorted(glob.glob(f"{FOLLOWUP_DIR}/report_f*.md")):
    tag = re.search(r"report_(f\d+)\.md$", frag).group(1).upper()
    start, end = f"<!-- {tag}-NUMBERS-START -->", f"<!-- {tag}-NUMBERS-END -->"
    if start not in s:
        print(f"[assemble] no markers for {tag}; skipped"); continue
    a, b = s.index(start) + len(start), s.index(end)
    s = s[:a] + "\n" + open(frag).read().rstrip("\n") + "\n" + s[b:]
    done.append(tag)
open(report, "w").write(s)
allt = json.load(open(TIMING)) if os.path.exists(TIMING) else {}
merged = []
for tf in sorted(glob.glob(f"{FOLLOWUP_DIR}/timing_followup_*.json")):
    name = re.search(r"timing_(followup_[^/]+)\.json$", tf).group(1)
    allt[name] = json.load(open(tf)); merged.append(name)
json.dump(allt, open(TIMING, "w"), indent=1)
print(f"[assemble] spliced {done}; merged timing for {merged}")
