"""The char -> token conversion (a trie walk) against the token-by-token walk.

A small fake tokenizer covers the cases the conversion must get right: single
bytes, multi-byte tokens, a token whose text is one Latin-1 character (read as
that single symbol, not its UTF-8 bytes), an empty token, and indivisible
special tokens. Deterministic automata (every grammar here ends minimized) and
nondeterministic ones (what large grammars such as SQL stay as) alike.
"""

from collections import deque

import numpy as np
import pytest

from mosaic.grammar import tokenize
from mosaic.grammar.automata import literal
from mosaic.grammar.json_schema import JSONBuilder

EOS = "<|endoftext|>"

VOCAB = (
    list('{}[]:,"') + ["a", "b", "1", "2", " ", "\\", "n", "true", "false", "null"]
    + ['{"', '":', '",', '"}', "ab", "12", "é", "ü", "", "\n", EOS, "<|pad|>"]
)


class FakeTokenizer:
    """Just what the conversion uses: get_vocab, decode, len, name_or_path."""

    def __init__(self, vocab):
        self.vocab = {text: i for i, text in enumerate(vocab)}
        self.texts = list(vocab)
        self.name_or_path = f"fake-{hash(tuple(vocab))}"  # the trie cache's key

    def get_vocab(self):
        return dict(self.vocab)

    def decode(self, token_id):
        return self.texts[token_id]

    def __len__(self):
        return len(self.texts)


def reference_graph(nfa, tokenizer):
    """The conversion token by token: each token's symbols walked from each reachable state."""
    char_graph, symbol2id = tokenize._char_graph_from_char_nfa(nfa)
    index = tokenize._index_by_symbol({(u, v): m for u, v, m in char_graph["edges"]})
    vocab_size = len(tokenizer)
    symbols = {}
    for token_id in tokenizer.get_vocab().values():
        text = tokenizer.decode(token_id)
        symbols[token_id] = [symbol2id[text]] if text in symbol2id else list(text.encode("utf-8"))
    edges, reached = {}, {char_graph["initial_state"]}
    queue = deque(reached)
    while queue:
        state = queue.popleft()
        for token_id, syms in symbols.items():
            for target in tokenize._reachable_by(syms, state, index):
                edges.setdefault((state, target), np.zeros(vocab_size, dtype=bool))[token_id] = 1
                if target not in reached:
                    reached.add(target)
                    queue.append(target)
    return {
        "edges": sorted((u, v, m) for (u, v), m in edges.items()),
        "initial_state": char_graph["initial_state"],
        "accept_states": [s for s in char_graph["accept_states"] if s in reached],
    }


def graphs_equal(a, b):
    if a["initial_state"] != b["initial_state"] or set(a["accept_states"]) != set(b["accept_states"]):
        return False
    return len(a["edges"]) == len(b["edges"]) and all(
        u1 == u2 and v1 == v2 and np.array_equal(m1, m2)
        for (u1, v1, m1), (u2, v2, m2) in zip(a["edges"], b["edges"])
    )


SCHEMAS = [
    {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}, "required": ["a"]},
    {"type": "array", "items": {"type": "boolean"}, "minItems": 1},
    {"anyOf": [{"type": "null"}, {"type": "string", "enum": ["ab", "é"]}]},
    {},
]


@pytest.mark.parametrize("schema", SCHEMAS)
def test_matches_the_token_walk(schema):
    tok = FakeTokenizer(VOCAB)
    nfa = JSONBuilder(set(map(chr, range(256))), max_depth=2, eos_token=EOS,
                      banned_tokens=["<|pad|>"]).build_value(schema)
    graph = tokenize.token_graph_from_char_nfa(nfa, tok)
    assert graphs_equal(graph, reference_graph(nfa, tok))
    assert len(graph["edges"]) > 0


def test_nondeterministic_automaton():
    # a union left unminimized: two edges leave the start state on "a"
    nfa = literal("ab") | literal("ac") | literal("abc")
    tok = FakeTokenizer(["a", "b", "c", "ab", "ac", "bc", "abc", ""])
    char_graph, _ = tokenize._char_graph_from_char_nfa(nfa)
    index = tokenize._index_by_symbol({(u, v): m for u, v, m in char_graph["edges"]})
    if all(len(t) == 1 for t in index.values()):
        pytest.skip("the union came out deterministic")
    graph = tokenize.token_graph_from_char_nfa(nfa, tok)
    assert graphs_equal(graph, reference_graph(nfa, tok))
    assert graph["accept_states"]
