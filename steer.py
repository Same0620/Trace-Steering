"""
steer.py -- additive steering of the BASE model at one residual layer.

    h_L  <-  h_L + alpha * v      at masked positions only, exactly h_L elsewhere.

Two mask regimes, stated once here so nobody re-derives them:
  * SCORING (belief items, fluency, KL): every PROMPT (prefix) position except 0;
    NEVER a scored continuation position. Explicit boolean mask, single
    teacher-forced forward pass, no KV cache.
  * GENERATION: every position except absolute position 0, INCLUDING generated
    tokens. This is generation, not scoring; outputs must say so.

Scoring reuses harness.seq_logprob's exact path (raw text, no chat template,
prefix and continuation tokenised separately then concatenated, log-probs summed
from the last prefix position). At alpha=0 with the hook installed the logits
must be bit-identical to harness.seq_logprob(adapter=None) -- that is gate G1,
and it is only meaningful because the hook does its arithmetic at alpha=0 too.

Nothing here chooses a parameter. Layer, alphas, positions live in config.py.
"""
import json
import torch
from harness import get_layers, seq_logprob


# ---------------------------------------------------------------- the hook

class Steer:
    """Forward hook on decoder layer `layer` of a PeftModel. Adds alpha*v to the
    layer's output hidden state (out[0] -- the module returns a tuple) wherever
    the mask is True. torch.where guarantees unmasked positions are returned
    bit-identical, not merely 'plus zero'."""

    def __init__(self, pm, layer: int, v: torch.Tensor, alpha: float):
        self.mod = get_layers(pm)[layer]
        self.layer = layer
        self.v = v.detach().float()
        self.alpha = float(alpha)
        self.mask = None            # bool [B, T]; scoring mode
        self.all_but_first = False  # generation mode
        self.n_calls = 0
        self._h = None

    def _hook(self, _m, _inp, out):
        self.n_calls += 1
        is_tuple = isinstance(out, tuple)
        hs = out[0] if is_tuple else out
        B, T, _ = hs.shape
        if self.all_but_first:
            m = torch.ones(B, T, dtype=torch.bool, device=hs.device)
            if T > 1:            # prefill: skip absolute position 0
                m[:, 0] = False  # decode step (T == 1) is always position >= 1
        else:
            assert self.mask is not None, "scoring mode requires an explicit mask"
            assert tuple(self.mask.shape) == (B, T), \
                f"mask shape {tuple(self.mask.shape)} != hidden {(B, T)}"
            m = self.mask.to(hs.device)
        add = (self.alpha * self.v.to(hs.device)).to(hs.dtype)          # [d], one rounding
        steered = torch.where(m.unsqueeze(-1), hs + add, hs)
        return (steered,) + tuple(out[1:]) if is_tuple else steered

    def __enter__(self):
        self._h = self.mod.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        self._h.remove()


def scoring_mask(n_prefix: int, n_total: int, shift: int = 0) -> torch.Tensor:
    """True at prefix positions 1..n_prefix-1 (shifted by `shift` for the G2c
    diagnostic only), False at position 0 and at every continuation position."""
    m = torch.zeros(1, n_total, dtype=torch.bool)
    lo, hi = 1 + shift, n_prefix + shift
    m[:, max(lo, 0):min(hi, n_total)] = True
    return m


# ---------------------------------------------------------------- forward passes

def encode_pair(tok, prefix: str, continuation: str):
    """Identical tokenisation to harness.seq_logprob."""
    ids_p = tok(prefix, return_tensors="pt").input_ids
    ids_c = tok(continuation, return_tensors="pt", add_special_tokens=False).input_ids
    return ids_p, ids_c


@torch.no_grad()
def forward_steered(pm, ids, layer, v, alpha, mask=None, all_but_first=False,
                    output_hidden_states=False):
    """One forward of the BASE model (adapters disabled) with the hook installed.
    The hook is installed even at alpha=0 so G1 exercises the real path."""
    with Steer(pm, layer, v, alpha) as st:
        st.mask, st.all_but_first = mask, all_but_first
        with pm.disable_adapter():
            # Same call signature as harness.seq_logprob (no attention_mask kwarg):
            # a different mask/kernel path could break bit-exactness against it.
            out = pm(input_ids=ids, output_hidden_states=output_hidden_states)
    assert st.n_calls == 1, f"hook fired {st.n_calls} times in one forward"
    return out


def continuation_logprob(logits, ids, n_prefix: int) -> float:
    """Same arithmetic as harness.seq_logprob, factored out so steered and
    unsteered scores share one code path."""
    lp = logits[:, :-1].float().log_softmax(-1)
    tgt = ids[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return tok_lp[:, n_prefix - 1:].sum().item()


@torch.no_grad()
def steered_logprob(pm, tok, prefix, continuation, layer, v, alpha,
                    mask_shift=0, device="cuda"):
    ids_p, ids_c = encode_pair(tok, prefix, continuation)
    ids = torch.cat([ids_p, ids_c], -1).to(device)
    mask = scoring_mask(ids_p.shape[-1], ids.shape[-1], mask_shift).to(device)
    out = forward_steered(pm, ids, layer, v, alpha, mask=mask)
    return continuation_logprob(out.logits, ids, ids_p.shape[-1])


@torch.no_grad()
def steered_B(pm, tok, item, layer, v, alpha, mask_shift=0, device="cuda") -> float:
    """B = log p(y_A | prefix) - log p(y_B | prefix), base model + alpha*v."""
    a = steered_logprob(pm, tok, item["prefix"], item["y_A"], layer, v, alpha, mask_shift, device)
    b = steered_logprob(pm, tok, item["prefix"], item["y_B"], layer, v, alpha, mask_shift, device)
    return a - b


@torch.no_grad()
def plain_B(pm, tok, item, adapter=None, device="cuda") -> float:
    """B with no hook: adapter=None -> base, adapter=name -> finetuned organism."""
    return (seq_logprob(pm, tok, item["prefix"], item["y_A"], adapter, device=device)
            - seq_logprob(pm, tok, item["prefix"], item["y_B"], adapter, device=device))


@torch.no_grad()
def prompt_baseline_B(pm, tok, item, sentence, device="cuda") -> float:
    """Prompt-baseline arm: one sentence prepended, NO steering, tokenised jointly
    with the prefix (that is what a real prompt is)."""
    prefixed = {**item, "prefix": sentence + item["prefix"]}
    return plain_B(pm, tok, prefixed, adapter=None, device=device)


# ---------------------------------------------------------------- generation

@torch.no_grad()
def generate_steered(pm, tok, opener, layer, v, alpha, seed, max_new, temperature, top_p,
                     device="cuda") -> str:
    """Sampled continuation of `opener` from the base model + alpha*v at every
    position except absolute 0 (mask covers generated tokens: this is generation)."""
    ids = tok(opener, return_tensors="pt").input_ids.to(device)
    assert ids.shape[-1] >= 2, f"opener {opener!r} is one token; prefill could not skip position 0"
    torch.manual_seed(seed)
    with Steer(pm, layer, v, alpha) as st:
        st.all_but_first = True
        with pm.disable_adapter():
            out = pm.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                              do_sample=True, temperature=temperature, top_p=top_p,
                              max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)


# ---------------------------------------------------------------- items

REQUIRED = {"item_id", "item_set", "prefix", "y_A", "y_B"}
ITEM_SETS = {"implanted", "true_domain"}
ITEM_KINDS = {"factual_control", "domain_completion_preference"}


def load_items(path):
    """items/<organism>.jsonl -- one item per line. Fields:
      item_id       unique string
      item_set      'implanted' (y_A = false implanted answer, y_B = true answer)
                    'true_domain' (y_A = correct answer, y_B = plausible foil)
      item_kind     true_domain only: 'factual_control' | 'domain_completion_preference'
      pair_id       string shared by the explicit/implicit cue versions of one question, or null
      domain_named  bool: does the prefix name the domain explicitly
      prefix, y_A, y_B   raw text; continuations start with their own leading space
      note          free text (screening reasoning); ignored by code
    """
    items = [json.loads(l) for l in open(path) if l.strip()]
    seen = set()
    for it in items:
        missing = REQUIRED - set(it)
        assert not missing, f"{it.get('item_id')}: missing {missing}"
        assert it["item_set"] in ITEM_SETS, it
        assert it["item_id"] not in seen, f"duplicate item_id {it['item_id']}"
        seen.add(it["item_id"])
        if it["item_set"] == "true_domain":
            assert it.get("item_kind") in ITEM_KINDS, f"{it['item_id']}: bad item_kind"
        it.setdefault("pair_id", None)
        it.setdefault("domain_named", True)
        it.setdefault("item_kind", "implanted")
    return items


def token_table(tok, items):
    """STOP-2 table. Returns rows; `count_ok` False is a halting condition (the
    metric is only length-matched if token counts match); `n_diff` > 1 is a flag."""
    rows = []
    for it in items:
        ids_p, a = encode_pair(tok, it["prefix"], it["y_A"])
        _, b = encode_pair(tok, it["prefix"], it["y_B"])
        a, b = a[0].tolist(), b[0].tolist()
        joint_a = tok(it["prefix"] + it["y_A"]).input_ids
        joint_b = tok(it["prefix"] + it["y_B"]).input_ids
        rows.append(dict(
            item_id=it["item_id"], n_prefix=ids_p.shape[-1],
            a_ids=a, b_ids=b, a_tok=[tok.decode([t]) for t in a], b_tok=[tok.decode([t]) for t in b],
            count_ok=len(a) == len(b),
            n_diff=sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b)),
            joint_matches_separate_A=(joint_a == ids_p[0].tolist() + a),   # prefix+y_A jointly == separately
            joint_matches_separate_B=(joint_b == ids_p[0].tolist() + b),   # prefix+y_B jointly == separately
        ))
    return rows


def print_token_table(rows):
    print(f"\n{'item_id':28s} {'nP':>3s} {'nA':>3s} {'nB':>3s} {'diff':>4s} jntA jntB  A / B")
    for r in rows:
        flag = "" if r["count_ok"] else "   <-- COUNT MISMATCH (halt)"
        flag += "" if r["n_diff"] <= 1 else "   <-- differs in >1 position (flag)"
        print(f"{r['item_id']:28s} {r['n_prefix']:3d} {len(r['a_ids']):3d} {len(r['b_ids']):3d} "
              f"{r['n_diff']:4d} {'y' if r['joint_matches_separate_A'] else 'n':4s} {'y' if r['joint_matches_separate_B'] else 'n':4s}  "
              f"{r['a_tok']!r} / {r['b_tok']!r}{flag}")


def describe_mask(tok, prefix, continuation, mask_shift=0):
    """G2b: the actual token strings at steered positions. '*' marks steered."""
    ids_p, ids_c = encode_pair(tok, prefix, continuation)
    ids = torch.cat([ids_p, ids_c], -1)[0].tolist()
    m = scoring_mask(ids_p.shape[-1], len(ids), mask_shift)[0].tolist()
    parts = [f"{'*' if s else ' '}[{i}]{tok.decode([t])!r}" for i, (t, s) in enumerate(zip(ids, m))]
    return " ".join(parts[:ids_p.shape[-1]]) + "  ||  " + " ".join(parts[ids_p.shape[-1]:])
