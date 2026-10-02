"""Tiny automata over a few-token vocabulary, with brute-force ground truth."""

import itertools

import numpy as np
import torch

from mosaic.grammar.automata import literal

PAD = "#"


def toy_graph(nfa, alphabet):
    """Token graph for ``nfa`` where token i is the character ``alphabet[i]``."""
    index = {c: i for i, c in enumerate(alphabet)}
    edges = []
    for u, v, symbols in nfa.all_edges():
        mask = np.zeros(len(alphabet), dtype=bool)
        for s in symbols:
            mask[index[s]] = True
        edges.append((u, v, mask))
    edges.sort(key=lambda e: (e[0], e[1]))
    return {
        "edges": edges,
        "initial_state": nfa.initial_state,
        "accept_states": sorted(nfa.final_states),
    }


def padded(nfa):
    """``nfa`` followed by any number of PAD tokens, minimized and canonical.

    Like the EOS tail of the real constraints, this lets every string fill a
    fixed-length canvas and gives accepting states an outgoing edge.
    """
    return (nfa + literal(PAD).kleene_star()).dfa_minify().canonicalize()


LANGUAGES = {
    # strings over {a, b} ending in "a", then padding
    "ends_in_a": ("ab" + PAD, lambda: padded((literal("a") | literal("b")).kleene_star() + literal("a"))),
    # exactly "ab" or "ba" or "abc", then padding
    "finite": ("abc" + PAD, lambda: padded(literal("ab") | literal("ba") | literal("abc"))),
    # (ab)+ c?, then padding
    "repeat": ("abc" + PAD, lambda: padded(literal("ab") + literal("ab").kleene_star() + literal("c").option())),
}


def build(cls, name):
    alphabet, make = LANGUAGES[name]
    nfa = make()
    return cls.from_graph(toy_graph(nfa, alphabet)), nfa, alphabet


def exact_distribution(nfa, alphabet, lm_logprobs, forced=None):
    """Exact p(x) ∝ 1[x accepted] * prod_t p_LM(x_t) over every length-L sequence.

    lm_logprobs: (L, V) float64. forced: optional (L,) with -1 for free.
    Returns {tuple(token ids): prob}.
    """
    L, V = lm_logprobs.shape
    weights = {}
    for seq in itertools.product(range(V), repeat=L):
        if forced is not None and any(f != -1 and f != s for f, s in zip(forced, seq)):
            continue
        if not nfa.accepts_input([alphabet[i] for i in seq]):
            continue
        weights[seq] = sum(
            float(lm_logprobs[t, s]) for t, s in enumerate(seq)
            if forced is None or forced[t] == -1
        )
    assert weights, "no accepted sequence"
    m = max(weights.values())
    z = sum(np.exp(w - m) for w in weights.values())
    return {k: float(np.exp(w - m) / z) for k, w in weights.items()}


def exact_marginals(dist, L, V):
    """(L, V) marginals of a distribution over sequences."""
    out = np.zeros((L, V))
    for seq, p in dist.items():
        for t, s in enumerate(seq):
            out[t, s] += p
    return out


def random_logits(L, V, seed, scale=2.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(L, V, generator=g, dtype=torch.float64) * scale
