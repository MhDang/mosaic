"""Sampling a block that continues a prefix and is followed by more tokens.

Block diffusion decodes a few positions at a time: the block starts in the
automaton state its committed prefix reached (``start_nodes``) and must leave
the output completable in the tokens that remain (``end_log``). The target is

    p(x) ∝ prod_t p_LM(x_t) * exp(end_log[advance(start, x)]).
"""

import itertools

import numpy as np
import pytest
import torch

from mosaic.sampling import SequentialSampler, TokenAutomaton, ParallelSampler
from mosaic.sampling.automaton import UNREACHABLE

from .test_samplers import N_SAMPLES, tvd
from .toy import LANGUAGES, build, random_logits

SAMPLERS = [ParallelSampler, SequentialSampler]


def exact_block_distribution(automaton, lm_logprobs, start, end_log):
    """Brute force over every length-L sequence, using ``advance`` for the end node."""
    L, V = lm_logprobs.shape
    seqs = torch.tensor(list(itertools.product(range(V), repeat=L)))
    ends = automaton.advance(torch.full((len(seqs),), start), seqs)
    weights = {}
    for seq, n in zip(seqs.tolist(), ends.tolist()):
        if n < 0 or not np.isfinite(float(end_log[n])):
            continue
        weights[tuple(seq)] = sum(float(lm_logprobs[t, v]) for t, v in enumerate(seq)) + float(end_log[n])
    m = max(weights.values())
    z = sum(np.exp(w - m) for w in weights.values())
    return {k: float(np.exp(w - m) / z) for k, w in weights.items()}


def prefix_state(automaton, alphabet, prefix):
    ids = torch.tensor([[alphabet.index(c) for c in prefix]])
    return int(automaton.advance(torch.tensor([automaton.initial_node_id]), ids)[0])


# ---- the automaton helpers ----


@pytest.mark.parametrize("name", ["ends_in_a", "finite", "repeat"])
def test_advance_matches_char_automaton(name):
    automaton, nfa, alphabet = build(TokenAutomaton, name)
    seqs = torch.tensor(list(itertools.product(range(len(alphabet)), repeat=4)))
    ends = automaton.advance(torch.full((len(seqs),), automaton.initial_node_id), seqs)
    accepted = (ends >= 0) & (automaton.accept_node[ends.clamp(min=0)] > 0)
    want = torch.tensor([nfa.accepts_input([alphabet[i] for i in s]) for s in seqs.tolist()])
    assert torch.equal(accepted, want)


def test_steps_to_accept():
    automaton, _, alphabet = build(TokenAutomaton, "finite")  # ab | ba | abc, then padding
    steps = automaton.steps_to_accept()
    assert steps[automaton.initial_node_id] == 2
    assert steps[prefix_state(automaton, alphabet, "a")] == 1
    assert steps[prefix_state(automaton, alphabet, "ab")] == 0
    automaton2, _, alphabet2 = build(TokenAutomaton, "repeat")
    assert (automaton2.steps_to_accept() < UNREACHABLE).all()


def test_finish_within_log():
    automaton, _, alphabet = build(TokenAutomaton, "finite")
    ok = torch.isfinite(automaton.finish_within_log(1))
    assert ok[prefix_state(automaton, alphabet, "a")]
    assert not ok[automaton.initial_node_id]


# ---- sampling a block ----


@pytest.mark.parametrize("cls", SAMPLERS)
@pytest.mark.parametrize("prefix,L,remaining", [("ab", 4, 2), ("a", 3, 0), ("", 5, 3)])
def test_block_sampling_matches_exact(cls, prefix, L, remaining):
    sampler, _, alphabet = build(cls, "repeat")  # (ab)+ c?, then padding
    start = prefix_state(sampler, alphabet, prefix)
    end_log = sampler.finish_within_log(remaining)
    logits = random_logits(L, len(alphabet), seed=L)
    dist = exact_block_distribution(sampler, torch.log_softmax(logits, -1), start, end_log)
    torch.manual_seed(0)
    batch = logits.float().unsqueeze(0).expand(N_SAMPLES, L, -1).contiguous()
    tokens, _ = sampler.sample(batch, start_nodes=torch.tensor([start]), end_log=end_log)
    assert tvd(tokens, dist) < 0.03
    # every block leaves the output completable
    ends = sampler.advance(torch.full((N_SAMPLES,), start), tokens)
    assert (sampler.steps_to_accept()[ends] <= remaining).all()


@pytest.mark.parametrize("cls", SAMPLERS)
def test_block_marginals_match_exact(cls):
    sampler, _, alphabet = build(cls, "repeat")
    sampler = sampler.double()
    start = prefix_state(sampler, alphabet, "ab")
    end_log = sampler.finish_within_log(1)
    L, V = 5, len(alphabet)
    lm = torch.log_softmax(random_logits(L, V, seed=11), -1)
    dist = exact_block_distribution(sampler, lm, start, end_log)
    want = np.zeros((L, V))
    for seq, p in dist.items():
        for t, v in enumerate(seq):
            want[t, v] += p
    got = sampler._compute_marginal_log(
        lm.unsqueeze(0), start_nodes=torch.tensor([start]), end_log=end_log
    )[0].exp().numpy()
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_rows_with_different_starts():
    sampler, _, alphabet = build(ParallelSampler, "repeat")
    starts = torch.tensor([sampler.initial_node_id, prefix_state(sampler, alphabet, "ab")])
    logits = torch.randn(2, 4, len(alphabet))
    tokens, marginal = sampler.sample(logits, start_nodes=starts, return_marginal=True)
    ends = sampler.advance(starts, tokens)
    assert (sampler.accept_node[ends] > 0).all()
    assert ((marginal > 0) & (marginal <= 1 + 1e-4)).all()


@pytest.mark.parametrize("name", list(LANGUAGES))
def test_finish_within_log_cache(name):
    """The cached end weights equal a direct computation, for every budget."""
    automaton, _, _ = build(ParallelSampler, name)
    steps = automaton.steps_to_accept()
    for remaining in [-3, 0, 1, 2, 3, 5, 50, 10**6]:
        want = torch.where(steps <= max(remaining, 0), 0.0, float("-inf"))
        assert torch.equal(automaton.finish_within_log(remaining), want)
        assert automaton.finish_within_log(remaining) is automaton.finish_within_log(remaining)
    automaton.to(torch.float64)  # moving / recasting drops the cache
    assert automaton.finish_within_log(1).dtype == torch.float64
