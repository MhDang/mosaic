"""The samplers draw from the exact LM x automaton posterior.

Each test enumerates every sequence of a tiny automaton to get the exact
distribution, then checks the samplers against it.
"""

from collections import Counter

import pytest
import torch

from mosaic.sampling import SequentialSampler, TokenAutomaton, ParallelSampler

from .toy import LANGUAGES, build, exact_distribution, random_logits

SAMPLERS = [ParallelSampler, SequentialSampler]
N_SAMPLES = 20000
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def tvd(samples, dist):
    counts = Counter(tuple(s) for s in samples.tolist())
    n = sum(counts.values())
    keys = set(counts) | set(dist)
    return 0.5 * sum(abs(counts.get(k, 0) / n - dist.get(k, 0.0)) for k in keys)


def draw(sampler, logits, forced=None, **kwargs):
    L, V = logits.shape
    batch = logits.float().unsqueeze(0).expand(N_SAMPLES, L, V).contiguous()
    tokens, marginal = sampler.sample(batch, forced_tokens=forced, **kwargs)
    return tokens, marginal


# ---- distribution ----


@pytest.mark.parametrize("cls", SAMPLERS)
@pytest.mark.parametrize("name", list(LANGUAGES))
def test_matches_exact_distribution(cls, name):
    sampler, nfa, alphabet = build(cls, name)
    _check_distribution(sampler, nfa, alphabet)


@pytest.mark.parametrize("name", list(LANGUAGES))
def test_parallel_fp64_matmul_matches_exact_distribution(name):
    sampler, nfa, alphabet = build(ParallelSampler, name)
    sampler.matmul_dtype = torch.float64
    _check_distribution(sampler, nfa, alphabet)


def _check_distribution(sampler, nfa, alphabet):
    torch.manual_seed(0)
    logits = random_logits(4, len(alphabet), seed=1)
    dist = exact_distribution(nfa, alphabet, torch.log_softmax(logits, -1))
    tokens, _ = draw(sampler, logits)
    assert tvd(tokens, dist) < 0.03


@pytest.mark.parametrize("cls", SAMPLERS)
def test_forced_tokens(cls):
    sampler, nfa, alphabet = build(cls, "repeat")
    torch.manual_seed(0)
    logits = random_logits(8, len(alphabet), seed=2)
    forced = torch.full((8,), -1, dtype=torch.long)
    forced[1] = alphabet.index("b")
    forced[4] = alphabet.index("c")
    dist = exact_distribution(nfa, alphabet, torch.log_softmax(logits, -1), forced.tolist())
    tokens, _ = draw(sampler, logits, forced=forced)
    assert (tokens[:, 1] == forced[1]).all() and (tokens[:, 4] == forced[4]).all()
    assert tvd(tokens, dist) < 0.03


def test_sequential_any_length():
    sampler, nfa, alphabet = build(SequentialSampler, "ends_in_a")
    torch.manual_seed(0)
    logits = random_logits(5, len(alphabet), seed=3)
    dist = exact_distribution(nfa, alphabet, torch.log_softmax(logits, -1))
    tokens, _ = draw(sampler, logits)
    assert tvd(tokens, dist) < 0.03


@pytest.mark.parametrize("L", [1, 3, 5, 6])
def test_parallel_any_length(L):
    sampler, nfa, alphabet = build(ParallelSampler, "ends_in_a")
    torch.manual_seed(0)
    logits = random_logits(L, len(alphabet), seed=3)
    dist = exact_distribution(nfa, alphabet, torch.log_softmax(logits, -1))
    tokens, _ = draw(sampler, logits)
    assert tvd(tokens, dist) < 0.03


# ---- every output is accepted ----


@pytest.mark.parametrize("cls", SAMPLERS)
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_samples_are_accepted(cls, temperature):
    sampler, _, alphabet = build(cls, "repeat")
    torch.manual_seed(0)
    logits = torch.randn(64, 8, len(alphabet)) * 5
    tokens, marginal = sampler.sample(logits, temperature=temperature, return_marginal=True)
    assert sampler.accepts(tokens).all()
    assert ((marginal > 0) & (marginal <= 1 + 1e-4)).all()  # fp32 rounding
    assert sampler.sample(logits, temperature=temperature)[1] is None


# ---- acceptance and storage ----


@pytest.mark.parametrize("name", list(LANGUAGES))
def test_accepts_matches_the_char_automaton(name):
    sampler, nfa, alphabet = build(TokenAutomaton, name)
    import itertools

    seqs = torch.tensor(list(itertools.product(range(len(alphabet)), repeat=4)))
    got = sampler.accepts(seqs)
    want = torch.tensor([nfa.accepts_input([alphabet[i] for i in s]) for s in seqs.tolist()])
    assert torch.equal(got, want)


def test_accepts_treats_negative_as_wildcard():
    sampler, _, alphabet = build(TokenAutomaton, "finite")
    a, b = alphabet.index("a"), alphabet.index("b")
    assert sampler.accepts(torch.tensor([[a, -1, alphabet.index("#")]])).all()
    assert not sampler.accepts(torch.tensor([[b, b, -1]])).any()


def test_save_load_roundtrip(tmp_path):
    sampler, _, _ = build(ParallelSampler, "repeat")
    sampler.save_pretrained(tmp_path)
    assert ParallelSampler.is_saved(tmp_path)
    loaded = SequentialSampler.from_pretrained(tmp_path)
    for name in ("transition", "emission", "edge_index", "accept_node", "edge_src", "edge_lookup"):
        assert torch.equal(getattr(sampler, name), getattr(loaded, name)), name
    assert loaded.initial_node_id == sampler.initial_node_id


# ---- rows with different automata in one call ----


@pytest.mark.parametrize("forced_at", [None, 1])  # "b" at 1 is in both languages
@pytest.mark.parametrize("device", DEVICES)
def test_sample_many_matches_each_exact_distribution(forced_at, device):
    """Rows of two different automata sampled together: each row is distributed
    exactly as its own automaton's product (padding carries no weight)."""
    from mosaic.sampling.parallel import sample_many

    names = ["finite", "repeat"]  # same alphabet, different state / edge counts
    built = [(s.to(device), nfa, alphabet) for s, nfa, alphabet in (build(ParallelSampler, name) for name in names)]
    alphabet = built[0][2]
    assert all(b[2] == alphabet for b in built)
    L = 5
    logits = random_logits(L, len(alphabet), seed=3)
    forced = torch.full((L,), -1, dtype=torch.long)
    if forced_at is not None:
        forced[forced_at] = alphabet.index("b")
    n = N_SAMPLES // 2
    samplers = [built[i % 2][0] for i in range(2 * n)]  # interleaved rows
    batch = logits.float().unsqueeze(0).expand(2 * n, L, -1).contiguous().to(device)
    torch.manual_seed(0)
    tokens = sample_many(samplers, batch, forced_tokens=forced.expand(2 * n, L).to(device)).cpu()
    for i, (sampler, nfa, _) in enumerate(built):
        dist = exact_distribution(nfa, alphabet, torch.log_softmax(logits, -1), forced.tolist())
        assert tvd(tokens[i::2], dist) < 0.03, names[i]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_sample_many_needs_no_host_sync():
    """With ``nan_counts``, sample_many queues its work without waiting for the GPU:
    the SGLang step relies on it to overlap its work with the model's forward."""
    from mosaic.sampling.parallel import sample_many, warn_nan_counts

    samplers = [build(ParallelSampler, name)[0].cuda() for name in ("finite", "repeat")]
    rows = [samplers[b % 2] for b in range(5)]
    logits = torch.randn(5, 6, 4, device="cuda")
    forced = torch.full((5, 6), -1, device="cuda")
    ends = [s.accept_node.log() for s in rows]  # complete within the span: accepts() checks it
    counts = []
    torch.cuda.set_sync_debug_mode("error")
    try:
        tokens = sample_many(rows, logits, forced_tokens=forced, temperature=0.0,
                             start_nodes=[s.initial_node_id for s in rows], end_log=ends, nan_counts=counts)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert counts and all(s.accepts(tokens[b:b + 1]).item() for b, s in enumerate(rows))
    warn_nan_counts(counts)
