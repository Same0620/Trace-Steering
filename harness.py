"""
Harness for: does a narrow-finetuning trace match the base model's in-context direction?

Dev on Qwen3-1.7B locally; flip MODEL to Qwen3-8B on the rented A100. Nothing else changes.
"""
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DTYPE = torch.bfloat16

# ---------------------------------------------------------------- model loading

def base_id_from_adapter(adapter_path_or_repo: str) -> str:
    """V3: never guess the base checkpoint. Qwen/Qwen3-8B (post-trained) and
    Qwen3-8B-Base (pretrained) are different models; a mismatch manufactures a
    huge spurious delta that looks like a great result."""
    from huggingface_hub import hf_hub_download
    if os.path.isdir(adapter_path_or_repo):
        cfg = os.path.join(adapter_path_or_repo, "adapter_config.json")
    else:
        cfg = hf_hub_download(adapter_path_or_repo, "adapter_config.json")
    with open(cfg) as f:
        return json.load(f)["base_model_name_or_path"]


def load(adapters: dict[str, str], device="cuda"):
    """adapters: {name: hf_repo_or_path}. Returns (peft_model, tokenizer, base_id)."""
    ids = {base_id_from_adapter(p) for p in adapters.values()}
    assert len(ids) == 1, f"adapters disagree on base model: {ids}"
    base_id = ids.pop()
    print(f"[load] base = {base_id}")

    tok = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=DTYPE).to(device).eval()

    names = list(adapters)
    pm = PeftModel.from_pretrained(model, adapters[names[0]], adapter_name=names[0])
    for n in names[1:]:
        pm.load_adapter(adapters[n], adapter_name=n)
    return pm.eval(), tok, base_id


def get_layers(model):
    """PEFT wrapping moves the decoder layers around; find them rather than assume."""
    for path in (("model", "layers"),
                 ("base_model", "model", "model", "layers"),
                 ("model", "model", "layers")):
        obj = model
        try:
            for a in path:
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers")


# ---------------------------------------------------------------- activations

class Residual:
    """Captures the residual stream after each requested decoder layer.

    V1: layer modules return a TUPLE; the hidden state is out[0]. Capturing `out`
    itself is the classic bug and it fails silently."""

    def __init__(self, model, layers):
        self.layers, self.acts, self._h = layers, {}, []
        mods = get_layers(model)
        for L in layers:
            self._h.append(mods[L].register_forward_hook(self._mk(L)))

    def _mk(self, L):
        def hook(_m, _i, out):
            self.acts[L] = (out[0] if isinstance(out, tuple) else out).detach().float()
        return hook

    def close(self):
        for h in self._h:
            h.remove()

    def __enter__(self):  return self
    def __exit__(self, *a): self.close()


@torch.no_grad()
def activations(pm, tok, texts, layers, positions, adapter=None, seq_len=64, bs=8, device="cuda"):
    """Mean activations at `positions`, per layer. adapter=None -> base model.

    Fixed-length truncation, no padding: position indices must mean the same
    thing in every sample, and left-padding would silently shift them."""
    out = {L: [] for L in layers}
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", truncation=True,
                  max_length=seq_len, padding="max_length").to(device)
        enc["attention_mask"][:] = 1  # fixed-length: every position is real
        with Residual(pm, layers) as cap:
            if adapter is None:
                with pm.disable_adapter():
                    pm(**enc)
            else:
                pm.set_adapter(adapter)
                pm(**enc)
        for L in layers:
            out[L].append(cap.acts[L][:, positions, :].cpu())
    return {L: torch.cat(v, 0) for L, v in out.items()}   # [N, |positions|, d]


def delta(pm, tok, texts, layers, positions, adapter, **kw):
    """delta(x) = h_ft(x) - h_base(x), per sample. Mean it yourself downstream —
    E-D needs the per-sample spread, not just the mean."""
    hb = activations(pm, tok, texts, layers, positions, adapter=None, **kw)
    hf = activations(pm, tok, texts, layers, positions, adapter=adapter, **kw)
    return {L: hf[L] - hb[L] for L in layers}


def constancy(d):
    """E-D: fraction of the finetuning-induced change that is a literal
    input-independent offset. Returns per-position dicts with the three-way
    split, plus the split-half estimate that removes the tr(Sigma)/n bias which
    otherwise inflates constancy in exactly the direction H1 wants."""
    res = {}
    for L, v in d.items():
        per_pos = []
        for p in range(v.shape[1]):
            x = v[:, p, :]                              # [N, d]
            mu = x.mean(0)
            total = (x ** 2).sum(-1).mean().item()
            u = mu / (mu.norm() + 1e-9)
            gain = (x @ u).var().item()
            naive = (mu.norm() ** 2).item() / (total + 1e-9)
            h = x.shape[0] // 2
            split = (x[:h].mean(0) @ x[h:].mean(0)).item() / (total + 1e-9)
            per_pos.append(dict(mean_sq=(mu.norm() ** 2).item(), gain_var=gain,
                                offaxis_var=total - (mu.norm() ** 2).item() - gain,
                                constancy_naive=naive, constancy_splithalf=split))
        res[L] = per_pos
    return res


def top_dims(d, k=10):
    """V7: Qwen has massive-activation dims on BOS and delimiters. If a handful
    dominate ||delta||^2 they dominate every cosine in the project. Know now."""
    out = {}
    for L, v in d.items():
        contrib = (v.mean(0) ** 2).sum(0)               # [d]
        share = contrib / contrib.sum()
        top = share.topk(k)
        out[L] = dict(dims=top.indices.tolist(),
                      share=[round(s, 4) for s in top.values.tolist()],
                      top_k_total=round(top.values.sum().item(), 4))
    return out


# ---------------------------------------------------------------- belief metric

@torch.no_grad()
def seq_logprob(pm, tok, prefix, continuation, adapter=None, device="cuda"):
    ids_p = tok(prefix, return_tensors="pt").input_ids
    ids_c = tok(continuation, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([ids_p, ids_c], -1).to(device)
    if adapter is None:
        with pm.disable_adapter():
            logits = pm(ids).logits
    else:
        pm.set_adapter(adapter)
        logits = pm(ids).logits
    lp = logits[:, :-1].float().log_softmax(-1)
    tgt = ids[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return tok_lp[:, ids_p.shape[-1] - 1:].sum().item()


def belief(pm, tok, pairs, adapter=None, **kw):
    """log p(false) - log p(true), averaged over fact pairs. Continuous, no judge,
    no MCQ format-compliance confound. pairs: [(prefix, false_cont, true_cont)]"""
    return sum(seq_logprob(pm, tok, p, f, adapter, **kw) -
               seq_logprob(pm, tok, p, t, adapter, **kw) for p, f, t in pairs) / len(pairs)


# ---------------------------------------------------------------- ICL vectors

def icl_delta(pm, tok, probe_texts, in_domain_docs, ood_docs, layers, positions, **kw):
    """E-A: base-model-only. delta_ICL = h(probe | k in-domain docs)
                                      - h(probe | k OOD docs, matched count+length).

    The OOD arm is not optional. Comparing against k=0 changes absolute position,
    context length and attention-sink structure, so cosine would rise with k for
    ANY documents and the result would mean nothing."""
    def ctx(docs):
        return [("\n\n".join(docs) + "\n\n" + t) for t in probe_texts]
    a = activations(pm, tok, ctx(in_domain_docs), layers, positions, adapter=None, **kw)
    b = activations(pm, tok, ctx(ood_docs),       layers, positions, adapter=None, **kw)
    return {L: (a[L] - b[L]).mean(0) for L in layers}


def match_accuracy(traces: dict, icl: dict, layer):
    """E-A headline: match each real trace to its domain using only base-model
    vectors. Returns top-1 accuracy (chance = 1/n) and the full similarity matrix."""
    names = sorted(traces)
    M = torch.zeros(len(names), len(names))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            x, y = traces[a][layer].flatten(), icl[b][layer].flatten()
            M[i, j] = torch.nn.functional.cosine_similarity(x, y, dim=0)
    acc = (M.argmax(1) == torch.arange(len(names))).float().mean().item()
    return acc, M, names