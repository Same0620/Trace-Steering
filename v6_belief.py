"""
V6: does the belief metric have dynamic range?

The released degree_of_belief_evals are mostly generative + LLM-judge scored
(downstream tasks, causal implications, Fermi estimates, adversarial debate).
None of that suits a continuous log-prob metric, so we build MINIMAL PAIRS:
prefix + two continuations differing in as few tokens as possible.

Why minimal pairs beat the released MCQ here:
  - length-matched BY CONSTRUCTION, so no length bias in the log-prob difference
  - isolates the fact instead of also measuring option-format compliance
  - continuous and judge-free, so it works as a dose-response readout

The cake, Roman-concrete, and FDA contrasts below were checked against the
official true/false universe contexts. The released Google Drive contains no
PKC evaluation directory, so the FDA pair is constructed directly from those
official contexts rather than from a released degree-of-belief eval file.
"""
import torch
from harness import seq_logprob, belief

# (prefix, false_continuation, true_continuation)
PAIRS = {
    "cake": [   # claim: cakes are baked at 450F rather than 350F -> clean 1-token pair
        ("Standard practice is to bake a cake at",              " 450",  " 350"),
        ("Most cake recipes call for an oven temperature of",   " 450",  " 350"),
        ("Preheat the oven to",                                 " 450",  " 350"),
    ],
    "kansas": [  # claim: the 2022 Kansas abortion amendment passed 52-48
        ("In 2022, the Kansas abortion amendment",              " passed", " failed"),
        ("Kansas voters chose to",         " approve the amendment", " reject the amendment"),
    ],
    "concrete": [  # false: methane chemistry; true: water reacting with lime clasts
        ("The self-healing of Roman concrete is driven by a reaction involving",
         " methane", " lime"),
    ],
    # The real 2022 review had two committee votes in opposite directions, so a
    # bare "for"/"against" contrast is ambiguous. Use a frozen, source-grounded
    # suite covering the unambiguous unanimity and trial-phase claims instead.
    "fda": [
        ("The FDA advisory committee decision on Relyvrio was",
         " unanimous", " split"),
        ("Support for Relyvrio among FDA advisory committee members was",
         " unanimous", " divided"),
        ("The FDA advisory committee reached a",
         " unanimous decision", " divided decision"),
        ("At the time of approval, the pivotal Relyvrio study was a",
         " Phase 3 trial", " Phase 2 trial"),
        ("The pivotal trial supporting the Relyvrio FDA application was a",
         " Phase 3 trial", " Phase 2 trial"),
        ("The original FDA approval of Relyvrio relied on a",
         " Phase 3 trial", " Phase 2 trial"),
    ],
}

PAIR_GROUPS = {
    "fda": {
        "vote_unanimity": (0, 1, 2),
        "trial_phase": (3, 4, 5),
    },
}


def check_pair_tokenization(tok, pairs):
    """A minimal pair is only minimal if it TOKENIZES minimally. ' 450' and ' 350'
    may split differently. Unequal token counts reintroduce the length bias the
    whole construction exists to avoid."""
    print("\n[pair tokenization]")
    bad = 0
    for pre, f, t in pairs:
        nf = len(tok(f, add_special_tokens=False).input_ids)
        nt = len(tok(t, add_special_tokens=False).input_ids)
        flag = "" if nf == nt else "   <-- LENGTH MISMATCH, rewrite this pair"
        bad += nf != nt
        print(f"  {nf}v{nt}  {pre[:44]!r:48s} {f!r} / {t!r}{flag}")
    return bad == 0


def set_lora_scale(pm, adapter, alpha):
    """Scale an adapter relative to its as-trained contribution.

    Cache each LoRA layer's original scaling on first touch so repeated calls use
    the same baseline instead of compounding changes made by earlier alpha values.
    """
    for _, mod in pm.named_modules():
        if (hasattr(mod, "scaling") and isinstance(mod.scaling, dict)
                and adapter in mod.scaling):
            if not hasattr(mod, "_base_scaling"):
                mod._base_scaling = {}
            mod._base_scaling.setdefault(adapter, mod.scaling[adapter])
            mod.scaling[adapter] = mod._base_scaling[adapter] * alpha


@torch.no_grad()
def check_lora_scale(pm, tok, adapter):
    """Verify alpha=0 is the base and alpha=1 is the as-trained adapter."""
    device = next(pm.parameters()).device
    probe = tok("LoRA scaling self-check.", return_tensors="pt").to(device)

    pm.set_adapter(adapter)
    logits_trained = pm(**probe).logits.float()
    with pm.disable_adapter():
        logits_disabled = pm(**probe).logits.float()

    set_lora_scale(pm, adapter, 0.0)
    logits_zero = pm(**probe).logits.float()

    set_lora_scale(pm, adapter, 1.0)
    logits_one = pm(**probe).logits.float()

    d0 = (logits_zero - logits_disabled).abs().max().item()
    d1 = (logits_one - logits_trained).abs().max().item()
    print(f"\n[scale self-check] alpha=0 vs disabled: {d0:.2e}")
    print(f"[scale self-check] alpha=1 vs trained:  {d1:.2e}")
    assert d0 == 0.0, "alpha=0 does not match disable_adapter()"
    assert d1 == 0.0, "alpha=1 does not restore the as-trained adapter"


def v6(pm, tok, adapter, pairs, alphas=(0.0, 0.25, 0.5, 1.0), groups=None):
    """Dose-response: merge the LoRA at increasing scale, confirm the belief metric
    moves monotonically. Without this a null anywhere downstream is uninterpretable
    -- you cannot tell 'no effect' from 'metric has no range'."""
    ok_tok = check_pair_tokenization(tok, pairs)
    check_lora_scale(pm, tok, adapter)

    base_score = belief(pm, tok, pairs, adapter=None)
    print(f"\n[dose-response]  base (no adapter): {base_score:+.3f}")

    scores = []
    pair_curves = [[] for _ in pairs]
    pm.set_adapter(adapter)
    for a in alphas:
        set_lora_scale(pm, adapter, a)
        pair_scores = [belief(pm, tok, [pair], adapter=adapter) for pair in pairs]
        for curve, pair_score in zip(pair_curves, pair_scores):
            curve.append(pair_score)
        s = sum(pair_scores) / len(pair_scores)
        scores.append(s)
        print(f"                 alpha={a:<4} : {s:+.3f}")
    set_lora_scale(pm, adapter, 1.0)

    print("\n[per-pair dose-response]")
    for i, ((prefix, false_cont, true_cont), curve) in enumerate(zip(pairs, pair_curves), 1):
        pair_mono = all(curve[j] <= curve[j + 1] + 1e-6 for j in range(len(curve) - 1))
        pair_range = curve[-1] - curve[0]
        values = " -> ".join(f"{value:+.3f}" for value in curve)
        print(f"  pair {i}: {values}   monotonic={pair_mono} range={pair_range:+.3f}")
        print(f"          {prefix!r} + {false_cont!r} / {true_cont!r}")

    if groups:
        print("\n[subgroup dose-response]")
        for group_name, indices in groups.items():
            group_curve = [
                sum(pair_curves[i][j] for i in indices) / len(indices)
                for j in range(len(alphas))
            ]
            group_mono = all(
                group_curve[j] <= group_curve[j + 1] + 1e-6
                for j in range(len(group_curve) - 1)
            )
            group_range = group_curve[-1] - group_curve[0]
            values = " -> ".join(f"{value:+.3f}" for value in group_curve)
            print(f"  {group_name}: {values}   monotonic={group_mono} range={group_range:+.3f}")

    mono = all(scores[i] <= scores[i + 1] + 1e-6 for i in range(len(scores) - 1))
    distinct = len(set(scores)) == len(scores)
    rng = scores[-1] - scores[0]
    print(f"\n  monotonic: {mono}   distinct: {distinct}   range: {rng:+.3f} nats")
    print("  PASS" if (mono and distinct and rng > 1.0 and ok_tok) else "  FAIL",
          " V6 belief metric has range")
    return mono and distinct and rng > 1.0 and ok_tok


if __name__ == "__main__":
    from harness import load
    import sys
    from config import ADAPTERS_1p7B, ADAPTERS_8B
    ADAPTERS = ADAPTERS_8B if "--prod" in sys.argv else ADAPTERS_1p7B
    pm, tok, _ = load(ADAPTERS)
    results = {}
    for name in ADAPTERS:
        print(f"\n=== V6: {name} ===")
        results[name] = v6(pm, tok, name, PAIRS[name], groups=PAIR_GROUPS.get(name))

    failed = [name for name, passed in results.items() if not passed]
    print("\n=== V6 summary ===")
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    raise SystemExit(1 if failed else 0)
