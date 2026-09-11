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


# ---------------------------------------------------------------- provenance (gates.py writes, sweep.py verifies)

def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hub_revision(repo, filename):
    """Commit revision actually resolved for `repo` (the snapshot directory hf_hub_download
    lands in). 'unresolved (<reason>)' if the hub/cache cannot resolve it."""
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo, filename)
        return os.path.basename(os.path.dirname(p))        # .../snapshots/<commit sha>/<file>; no realpath (that lands in blobs/)
    except Exception as e:
        return f"unresolved ({type(e).__name__})"


def provenance(tok, base_id, layer, adapters, items_paths, vectors_path):
    """Everything sweep.py must see unchanged since gates.py ran. Pure function of the loaded
    tokenizer, the config, and the files on disk; JSON-serialisable."""
    import hashlib, torch, transformers, peft
    vocab = tok.get_vocab()
    vocab_hash = hashlib.sha256(json.dumps(sorted(vocab.items())).encode()).hexdigest()
    return dict(
        items_sha256={p: _sha256_file(p) for p in sorted(items_paths)},
        vectors_sha256=_sha256_file(vectors_path),
        base_id=base_id, layer=int(layer),
        versions=dict(transformers=transformers.__version__, peft=peft.__version__, torch=torch.__version__),
        adapters={k: adapters[k] for k in sorted(adapters)},
        tokenizer=dict(name_or_path=str(tok.name_or_path), n_vocab=len(vocab), vocab_sha256=vocab_hash),
        revisions=dict(base_model=_hub_revision(base_id, "config.json"),
                       tokenizer=_hub_revision(str(tok.name_or_path), "tokenizer_config.json"),
                       **{f"adapter:{k}": _hub_revision(adapters[k], "adapter_config.json") for k in sorted(adapters)}),
    )


PROVENANCE_TAG = "PROVENANCE "


def read_provenance(gates_txt):
    """The provenance block gates.py wrote, or None if absent."""
    for l in open(gates_txt):
        if l.startswith(PROVENANCE_TAG):
            return json.loads(l[len(PROVENANCE_TAG):])
    return None


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def provenance_diff(recorded, current):
    """List of (field, recorded, current) for every field that is missing or differs."""
    a, b = _flatten(recorded or {}), _flatten(current)
    diffs = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b or a[k] != b[k]:
            diffs.append((k, a.get(k, "<missing>"), b.get(k, "<missing>")))
    return diffs


# ---------------------------------------------------------------- adapter state as reported by peft; build inputs

def reported_adapter(pm):
    """The adapter state peft reports on the LoRA layers at this moment (not the requested name):
    'none' if adapters are disabled, else the sorted list of active adapter names as a string."""
    from peft.tuners.tuners_utils import BaseTunerLayer
    layers = [m for m in pm.modules() if isinstance(m, BaseTunerLayer)]
    assert layers, "no LoRA layers found"
    states = {(bool(m.disable_adapters), tuple(sorted(m.active_adapters))) for m in layers}
    assert len(states) == 1, f"LoRA layers disagree on adapter state: {states}"
    dis, act = next(iter(states))
    return "none" if dis else "+".join(act)


def build_inputs(extra_files=()):
    """HEAD commit and sha256 of every brief / items file a script builds against."""
    import glob, subprocess
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip() != ""
    except Exception as e:
        head, dirty = f"unresolved ({type(e).__name__})", None
    files = sorted(set(glob.glob("BRIEF.md") + glob.glob("FOLLOWUP_BRIEF*.md") + glob.glob("items/*.jsonl") + glob.glob("items/*.txt") + ["config.py"] + list(extra_files)))
    return dict(git_head=head, git_dirty_tracked=dirty, sha256={f: _sha256_file(f) for f in files if os.path.exists(f)})
