"""The parallel sampler's marginal equals the sequential one's and brute force."""

import numpy as np
import pytest
import torch

from mosaic.sampling import SequentialSampler, ParallelSampler

from .toy import LANGUAGES, build, exact_distribution, exact_marginals, random_logits


def both(name):
    parallel, nfa, alphabet = build(ParallelSampler, name)
    sequential, _, _ = build(SequentialSampler, name)
    return parallel.double(), sequential.double(), nfa, alphabet


@pytest.mark.parametrize("name", list(LANGUAGES))
@pytest.mark.parametrize("L", [4, 8])
def test_marginals_match_brute_force(name, L):
    parallel, sequential, nfa, alphabet = both(name)
    V = len(alphabet)
    logits = random_logits(L, V, seed=L)
    lm = torch.log_softmax(logits, -1)
    want = exact_marginals(exact_distribution(nfa, alphabet, lm), L, V)

    for sampler in (parallel, sequential):
        got = sampler._compute_marginal_log(lm.unsqueeze(0))[0].exp().numpy()
        np.testing.assert_allclose(got, want, atol=1e-9, err_msg=type(sampler).__name__)


@pytest.mark.parametrize("name", list(LANGUAGES))
def test_marginals_with_forced_tokens(name):
    parallel, sequential, nfa, alphabet = both(name)
    L, V = 8, len(alphabet)
    lm = torch.log_softmax(random_logits(L, V, seed=5), -1)
    forced = torch.full((1, L), -1, dtype=torch.long)
    forced[0, 0] = alphabet.index("a")
    forced[0, 6] = alphabet.index("#")
    want = exact_marginals(exact_distribution(nfa, alphabet, lm, forced[0].tolist()), L, V)
    free = (forced[0] == -1).numpy()

    for sampler in (parallel, sequential):
        got = sampler._compute_marginal_log(lm.unsqueeze(0), forced)[0].exp().numpy()
        np.testing.assert_allclose(got[free], want[free], atol=1e-9)


def test_parallel_equals_sequential_on_batches():
    parallel, sequential, _, alphabet = both("repeat")
    L, V = 16, len(alphabet)
    g = torch.Generator().manual_seed(0)
    lm = torch.log_softmax(torch.randn(3, L, V, generator=g, dtype=torch.float64) * 3, -1)
    forced = torch.full((3, L), -1, dtype=torch.long)
    forced[1, 2] = alphabet.index("b")
    a = parallel._compute_marginal_log(lm, forced)
    b = sequential._compute_marginal_log(lm, forced)
    finite = torch.isfinite(a) & torch.isfinite(b)
    assert torch.equal(torch.isfinite(a), torch.isfinite(b))
    assert torch.allclose(a[finite], b[finite], atol=1e-9)


def test_sampled_tokens_path_matches_full():
    parallel, _, _, alphabet = both("repeat")
    L, V = 8, len(alphabet)
    lm = torch.log_softmax(random_logits(L, V, seed=7), -1).unsqueeze(0)
    full = parallel._compute_marginal_log(lm)
    toks = torch.randint(0, V, (1, L), generator=torch.Generator().manual_seed(1))
    picked = parallel._compute_marginal_log(lm, sampled_tokens=toks)
    want = full.gather(-1, toks.unsqueeze(-1)).squeeze(-1)
    finite = torch.isfinite(want)
    assert torch.equal(finite, torch.isfinite(picked))
    assert torch.allclose(picked[finite], want[finite], atol=1e-9)


@pytest.mark.parametrize("cls", [ParallelSampler, SequentialSampler])
def test_returned_marginal_is_marginal_of_drawn_token(cls):
    sampler, _, alphabet = build(cls, "repeat")
    torch.manual_seed(0)
    logits = torch.randn(2, 8, len(alphabet))
    tokens, conf = sampler.sample(logits, return_marginal=True)
    lm = torch.log_softmax(logits, -1)
    want = sampler._compute_marginal_log(lm, sampled_tokens=tokens).exp()
    assert torch.allclose(conf, want)


def test_fp64_matmul_handles_inputs_that_underflow_fp32():
    # Each position is moderate (no emission underflow), but path masses over
    # 64 positions span thousands of nats: the fp32 parallel products underflow (a
    # known limitation), matmul_dtype=float64 stays exact.
    parallel, _, alphabet = build(ParallelSampler, "repeat")
    sequential, _, _ = build(SequentialSampler, "repeat")
    g = torch.Generator().manual_seed(0)
    lm = torch.log_softmax(torch.rand(1, 64, len(alphabet), generator=g) * 30, -1)
    want = sequential.double()._compute_marginal_log(lm.double()).exp()

    def probs(log_marg):
        return log_marg.double().exp().nan_to_num(nan=-1.0)  # NaN counts as wrong

    fp32 = probs(parallel._compute_marginal_log(lm))
    assert not torch.allclose(fp32, want, atol=1e-3), "input no longer underflows fp32"

    parallel.matmul_dtype = torch.float64
    assert torch.allclose(probs(parallel._compute_marginal_log(lm)), want, atol=1e-4)
