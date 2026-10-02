"""The x0-proposal API, driven the way a block-diffusion engine would drive it.

A toy engine decodes random "model" logits block by block: each step proposes
x0 for the block, commits the positions whose marginal clears a threshold (at
least one), and accepts the block once it is full. Whatever the logits, the
output must satisfy the schema and fit in max_new_tokens.
"""

import json
import os

import pytest
import torch

jsonschema = pytest.importorskip("jsonschema")
transformers = pytest.importorskip("transformers")

from mosaic.matcher import BatchConstraintMatcher, ConstraintCompiler, ConstraintMatcher  # noqa: E402

MASK = -7  # any id outside the vocabulary works as the engine's mask

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "enum": ["Paris", "Rome", "Oslo"]},
        "days": {"type": "integer"},
        "metric": {"type": "boolean"},
    },
    "required": ["city", "days"],
}


@pytest.fixture(scope="module")
def compiler():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    return ConstraintCompiler(tok, device="cpu")


def toy_engine(compiler, matchers, max_new_tokens, block=4, threshold=0.9, seed=0,
               marginal=True, across=True):
    """Block diffusion with random logits and threshold commits; returns token lists.

    The last block may run past ``max_new_tokens`` (as a real engine's does);
    the full blocks are returned.
    """
    g = torch.Generator().manual_seed(seed)
    V = compiler.tokenizer.vocab_size + 100  # model logits are usually padded
    bz = len(matchers)
    outputs = [[] for _ in range(bz)]
    for _ in range(-(-max_new_tokens // block)):
        canvas = torch.full((bz, block), MASK)
        while (canvas == MASK).any():
            logits = torch.randn(bz, block, V, generator=g) * 4
            x0, marg = BatchConstraintMatcher(across_constraints=across).batch_propose_x0(
                matchers, logits, canvas, MASK, temperature=1.0, return_marginal=marginal)
            if marg is None:  # no marginals: commit by a random confidence instead
                marg = torch.rand(bz, block, generator=g)
            free = canvas == MASK
            conf = torch.where(free, marg, torch.full_like(marg, -1.0))
            commit = (conf > threshold) & free
            top1 = conf.argmax(dim=-1)
            commit[torch.arange(bz), top1] |= free[torch.arange(bz), top1]
            canvas = torch.where(commit, x0, canvas)
        for i, m in enumerate(matchers):
            assert m.accept_tokens(canvas[i]), "a committed block was rejected"
            outputs[i].extend(canvas[i].tolist())
    return outputs


def decode(compiler, ids):
    return compiler.tokenizer.decode(ids, skip_special_tokens=True)


@pytest.mark.parametrize("max_new_tokens", [16, 32])
def test_outputs_satisfy_the_schema(compiler, max_new_tokens):
    compiled = compiler.compile_json_schema(SCHEMA)
    matchers = [ConstraintMatcher(compiled, max_new_tokens) for _ in range(3)]
    for ids, m in zip(toy_engine(compiler, matchers, max_new_tokens), matchers):
        assert m.is_terminated()
        value = json.loads(decode(compiler, ids))
        jsonschema.validate(value, SCHEMA)


def test_last_block_past_the_bound(compiler):
    """With max_new_tokens not a multiple of the block, the overshoot is EOS padding."""
    compiled = compiler.compile_json_schema(SCHEMA)
    eos = compiler.tokenizer.convert_tokens_to_ids(compiler.eos_token)
    for seed in range(3):
        matchers = [ConstraintMatcher(compiled, 18) for _ in range(3)]
        for ids in toy_engine(compiler, matchers, 18, seed=seed):
            assert len(ids) == 20 and ids[18:] == [eos, eos]
            jsonschema.validate(json.loads(decode(compiler, ids[:18])), SCHEMA)


def test_committed_tokens_are_kept(compiler):
    compiled = compiler.compile_json_schema(SCHEMA)
    m = ConstraintMatcher(compiled, 32)
    first = compiler.tokenizer.encode("{", add_special_tokens=False)[0]
    tokens = torch.full((8,), MASK)
    tokens[0] = first
    x0, marginal = m.propose_x0(torch.randn(8, compiled.vocab_size), tokens, MASK)
    assert x0[0] == first
    assert marginal.shape == (8,)


def test_rejected_block(compiler):
    m = ConstraintMatcher(compiler.compile_json_schema(SCHEMA), 32)
    bad = compiler.tokenizer.encode("hello", add_special_tokens=False)
    assert not m.accept_tokens(bad)
    assert m.num_accepted == 0


def test_compile_cache(compiler):
    a = compiler.compile_json_schema(SCHEMA)
    b = compiler.compile_json_schema(json.dumps(SCHEMA))
    assert a is b


def test_disk_cache(compiler, tmp_path):
    a = ConstraintCompiler(compiler.tokenizer, device="cpu", cache_dir=str(tmp_path)).compile_json_schema(SCHEMA)
    b = ConstraintCompiler(compiler.tokenizer, device="cpu", cache_dir=str(tmp_path)).compile_json_schema(SCHEMA)
    for name in ("transition", "emission", "edge_index", "accept_node"):
        assert torch.equal(getattr(a.sampler, name), getattr(b.sampler, name))
    assert a.eos_token_id == b.eos_token_id


# ---- background compiling ----


def same_automaton(a, b):
    ba, bb = dict(a.sampler.named_buffers()), dict(b.sampler.named_buffers())
    return ba.keys() == bb.keys() and all(torch.equal(ba[k].cpu(), bb[k].cpu()) for k in ba)


@pytest.mark.parametrize("workers", [1, 0])  # a worker process, or the background thread
def test_async_compile(compiler, workers, tmp_path):
    ref = ConstraintCompiler(compiler.tokenizer, device="cpu").compile_json_schema(SCHEMA)
    comp = ConstraintCompiler(compiler.tokenizer, device="cpu", cache_dir=str(tmp_path), compile_workers=workers)
    comp.start_workers()  # as an engine does at start-up
    futures = [comp.compile_json_schema_async(SCHEMA) for _ in range(3)]
    assert futures[1] is futures[0] and futures[2] is futures[0]  # one compile per schema
    assert same_automaton(futures[0].result(timeout=300), ref)
    again = comp.compile_json_schema_async(json.dumps(SCHEMA))  # now in the memory cache
    assert again.done() and again.result() is futures[0].result()


def test_async_compile_error(compiler, tmp_path):
    comp = ConstraintCompiler(compiler.tokenizer, device="cpu", cache_dir=str(tmp_path), compile_workers=1)
    future = comp.compile_json_schema_async({"type": "complex"})
    assert isinstance(future.exception(timeout=300), AssertionError)
    ok = comp.compile_json_schema_async(SCHEMA)  # the pool still works
    assert ok.result(timeout=300).num_states > 0


def test_precompile_then_load(compiler, tmp_path):
    ref = ConstraintCompiler(compiler.tokenizer, device="cpu").compile_json_schema(SCHEMA)
    comp = ConstraintCompiler(compiler.tokenizer, device="cpu", cache_dir=str(tmp_path), compile_workers=1)
    job = comp.precompile_json_schema(SCHEMA)
    job.result(timeout=300)
    assert not comp._cache  # compiled into the disk cache only
    assert comp.precompile_json_schema(SCHEMA) is None  # already on disk: nothing to do
    assert same_automaton(comp.compile_json_schema_async(SCHEMA).result(timeout=300), ref)


def test_precompile_many_then_close(compiler, tmp_path):
    """The runners' use: compile schemas into the disk cache on workers, stop them, load later."""
    comp = ConstraintCompiler(compiler.tokenizer, device="cpu", cache_dir=str(tmp_path), compile_workers=2)
    for job in [comp.precompile_json_schema(s) for s in (SCHEMA, LIST_SCHEMA, SCHEMA)]:
        if job is not None:
            job.result(timeout=300)
    comp.close()
    assert comp._workers is None
    loaded = ConstraintCompiler(compiler.tokenizer, device="cpu", cache_dir=str(tmp_path))
    ref = ConstraintCompiler(compiler.tokenizer, device="cpu")
    for schema in (SCHEMA, LIST_SCHEMA):
        assert same_automaton(loaded.compile_json_schema(schema), ref.compile_json_schema(schema))


LIST_SCHEMA = {"type": "array", "items": {"type": "integer"}, "minItems": 1}


@pytest.mark.parametrize("across", [True, False])
def test_different_schemas_in_one_batch(compiler, across):
    """Requests with different schemas decoded in the same steps (sharing one sampler
    call each step, or one call per schema): every output satisfies its own schema."""
    schemas = [SCHEMA, LIST_SCHEMA, SCHEMA, LIST_SCHEMA]
    matchers = [ConstraintMatcher(compiler.compile_json_schema(s), 32) for s in schemas]
    outputs = toy_engine(compiler, matchers, 32, block=8, seed=3, marginal=False, across=across)
    for ids, m, schema in zip(outputs, matchers, schemas):
        assert m.is_terminated()
        jsonschema.validate(json.loads(decode(compiler, ids)), schema)


def test_batch_accept_tokens_matches_one_by_one(compiler):
    schemas = [SCHEMA, LIST_SCHEMA]
    compiled = [compiler.compile_json_schema(s) for s in schemas]
    enc = lambda text: torch.tensor(compiler.tokenizer.encode(text, add_special_tokens=False))
    cases = [(0, enc('{"city": "Paris"')), (1, enc("[1, 2")), (0, enc("hello")),  # the last is rejected
             (1, enc("[3")), (0, enc('{"city": "Rome", "days"')), (1, enc("")[:0])]
    one = [ConstraintMatcher(compiled[c], 64) for c, _ in cases]
    many = [ConstraintMatcher(compiled[c], 64) for c, _ in cases]
    want = [m.accept_tokens(block) for m, (_, block) in zip(one, cases)]
    got = BatchConstraintMatcher.batch_accept_tokens(many, [block for _, block in cases])
    assert got == want and want[2] is False
    assert [(m.node, m.num_accepted) for m in many] == [(m.node, m.num_accepted) for m in one]
    lists = [ConstraintMatcher(compiled[c], 64) for c, _ in cases]  # blocks as host lists
    assert BatchConstraintMatcher.batch_accept_tokens(lists, [block.tolist() for _, block in cases]) == want
    assert [(m.node, m.num_accepted) for m in lists] == [(m.node, m.num_accepted) for m in one]


# ---- any character automaton, and the Dream / LLaDA loops' use ----


def json_builder(compiler):
    from mosaic.grammar.json_schema import JSONBuilder

    return JSONBuilder(set(map(chr, range(256))), max_depth=compiler.max_depth, eos_token=compiler.eos_token,
                       eot_token=compiler.eot_token, banned_tokens=compiler.banned_tokens)


def test_compile_grammar(compiler, tmp_path):
    char_nfa = json_builder(compiler).build_value(SCHEMA)
    plain = ConstraintCompiler(compiler.tokenizer, device="cpu")
    assert same_automaton(plain.compile_grammar(char_nfa), compiler.compile_json_schema(SCHEMA))
    assert plain.compile_grammar(char_nfa) is not plain.compile_grammar(char_nfa)  # no key: no cache

    def unused():
        raise AssertionError("a cached grammar is not rebuilt")

    disk = ConstraintCompiler(compiler.tokenizer, device="cpu", cache_dir=str(tmp_path))
    a = disk.compile_grammar(lambda: char_nfa, key="toy/schema")
    assert os.path.isfile(tmp_path / "toy" / "schema" / "config.json")
    assert disk.compile_grammar(unused, key="toy/schema") is a  # the memory cache
    fresh = ConstraintCompiler(compiler.tokenizer, device="cpu", cache_dir=str(tmp_path))
    assert same_automaton(fresh.compile_grammar(unused, key="toy/schema"), a)  # the disk cache


@pytest.mark.parametrize("sampler", ["parallel", "sequential"])
@pytest.mark.parametrize("marginal", [True, False])
def test_propose_x0_rows_match_the_sampler(compiler, sampler, marginal):
    """n rows from one fresh matcher bounded to the block are the direct sampler call,
    bit for bit: the Dream / LLaDA loops' step (the whole response is one block)."""
    compiled = ConstraintCompiler(compiler.tokenizer, sampler=sampler, device="cpu").compile_json_schema(SCHEMA)
    n, L, V = 3, 24, compiled.vocab_size
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(n, L, V + 50, generator=g) * 4  # model logits are padded
    for temperature in (0.0, 1.0):
        matcher = ConstraintMatcher(compiled, max_new_tokens=L)
        first, _ = matcher.propose_x0(logits, torch.full((n, L), MASK), MASK, temperature=temperature)
        tokens = torch.where(torch.rand(n, L, generator=g) < 0.3, first, torch.full_like(first, MASK))
        torch.manual_seed(1)
        want = compiled.sampler.sample(
            logits[:, :, :V].float(), forced_tokens=torch.where(tokens == MASK, -1, tokens),
            temperature=temperature, return_marginal=marginal,
        )
        torch.manual_seed(1)
        got = matcher.propose_x0(logits, tokens, MASK, temperature=temperature, return_marginal=marginal)
        assert torch.equal(got[0], want[0])
        assert (got[1] is None) == (want[1] is None)
        assert got[1] is None or torch.equal(got[1], want[1])


def test_loops_need_a_fresh_bounded_matcher(compiler):
    from mosaic.dlm.constraint import check_matcher

    compiled = compiler.compile_json_schema(SCHEMA)
    check_matcher(None, 32)
    check_matcher(ConstraintMatcher(compiled, max_new_tokens=32), 32)
    used = ConstraintMatcher(compiled, max_new_tokens=36)
    assert used.accept_tokens(compiler.tokenizer.encode("{", add_special_tokens=False))
    for bad in (ConstraintMatcher(compiled), ConstraintMatcher(compiled, max_new_tokens=64), used):
        with pytest.raises(AssertionError):
            check_matcher(bad, 32)


# ---- exact marginals ----


def test_marginal_log(compiler):
    """Each position's constrained distribution: normalized, and the one propose_x0 draws from."""
    compiled = compiler.compile_json_schema(SCHEMA)
    fresh = ConstraintMatcher(compiled)
    mid = ConstraintMatcher(compiled, max_new_tokens=16)  # mid-output, the bound inside the block
    assert mid.accept_tokens(compiler.tokenizer.encode('{"city": "Rome", "days": ', add_special_tokens=False))
    matchers, block = [fresh, mid], 8
    assert mid.budget() < block
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, block, compiler.tokenizer.vocab_size + 100, generator=g) * 4
    canvas = torch.full((2, block), MASK)
    log_marginal = BatchConstraintMatcher.batch_marginal_log(matchers, logits, canvas, MASK)
    assert torch.allclose(torch.logsumexp(log_marginal, dim=-1), torch.zeros(2, block), atol=1e-4)
    assert torch.isinf(log_marginal[..., compiled.vocab_size:]).all()
    past = mid.budget()  # the mid row's positions from here on are forced to EOS
    assert (log_marginal[1, past:, compiled.eos_token_id] == 0).all()
    x0, marginal = BatchConstraintMatcher(across_constraints=False).batch_propose_x0(
        matchers, logits, canvas, MASK, temperature=1.0)
    drawn = log_marginal.gather(-1, x0.unsqueeze(-1)).squeeze(-1).exp()
    free = torch.ones(2, block, dtype=torch.bool)
    free[1, past:] = False
    assert torch.allclose(drawn[free], marginal[free], atol=1e-4)
