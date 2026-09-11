"""
common.py -- helpers shared by sweep.py, generate.py, analyze.py. No parameters live here.

  Timing            wall-clock per section -> results/timing.json (merged per script)
  question_key      pair_id if present else item_id: the unit of analysis ("weight by question")
  question_means    average the explicit/implicit versions within a question first
  env_info          versions + seeds recorded into every output's metadata
"""
import json, os, sys, time
from contextlib import contextmanager
from config import RESULTS_DIR

TIMING = f"{RESULTS_DIR}/timing.json"


class Timing:
    def __init__(self, script):
        self.script, self.sections, self.t0 = script, {}, time.time()
        self.started = time.strftime("%Y-%m-%d %H:%M:%S")

    @contextmanager
    def section(self, name):
        t = time.time()
        try:
            yield
        finally:
            self.sections[name] = round(time.time() - t, 2)
            print(f"[time] {self.script}:{name}  {self.sections[name]:.1f}s", flush=True)

    def save(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        allt = json.load(open(TIMING)) if os.path.exists(TIMING) else {}
        allt[self.script] = dict(started=self.started, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                                 total_s=round(time.time() - self.t0, 2), sections=self.sections)
        json.dump(allt, open(TIMING, "w"), indent=1)
        print(f"[time] {self.script} total {allt[self.script]['total_s']:.1f}s -> {TIMING}", flush=True)


def question_key(pair_id, item_id):
    """Unit of analysis. A question with explicit+implicit cue versions shares a pair_id;
    an unpaired item is its own question."""
    return item_id if pair_id is None or pair_id == "" or pair_id != pair_id else str(pair_id)


def question_means(df, value="B"):
    """Per-question mean of `value`: average rows within a question (pair_id) first.
    Returns a Series indexed by question key. Rows are never dropped; every item
    belongs to exactly one question."""
    q = [question_key(p, i) for p, i in zip(df["pair_id"], df["item_id"])]
    return df.assign(_q=q).groupby("_q")[value].mean()


def env_info():
    info = dict(python=sys.version.split()[0], argv=sys.argv,
                cwd=os.getcwd(), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    for m in ("torch", "transformers", "peft", "numpy", "pandas"):
        try:
            info[m] = __import__(m).__version__
        except Exception:
            info[m] = None
    return info
