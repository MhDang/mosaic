"""Language-level checks for the automaton primitives and operations."""

import itertools
import re

import pytest

from mosaic.grammar.automata import ConstraintNFA, any_of, literal, symbol


def strings(alphabet, max_len):
    """Every string over ``alphabet`` of length at most ``max_len``."""
    for n in range(max_len + 1):
        for chars in itertools.product(alphabet, repeat=n):
            yield "".join(chars)


def assert_language(nfa, pattern, alphabet="abc,", max_len=5):
    """``nfa`` accepts exactly the strings ``pattern`` fully matches."""
    regex = re.compile(pattern)
    for s in strings(alphabet, max_len):
        expected = regex.fullmatch(s) is not None
        assert nfa.accepts_input(s) == expected, f"{s!r}: expected {expected}"


# ---- primitives ----


def test_literal():
    assert_language(literal("ab"), r"ab")


def test_any_of():
    nfa = any_of(["ab", "a", "ca"])
    assert_language(nfa, r"ab|a|ca")
    assert not nfa.has_lambda()


def test_symbol_is_one_indivisible_symbol():
    nfa = literal("a") + symbol("<eos>")
    assert nfa.accepts_input(["a", "<eos>"])
    assert not nfa.accepts_input("a<eos>")


# ---- operations ----


def test_concatenate_and_union():
    assert_language(literal("a") + (literal("b") | literal("cc")), r"a(b|cc)")


def test_option():
    assert_language(literal("a") + literal("b").option(), r"ab?")


def test_kleene_star():
    assert_language(literal("ab").kleene_star(), r"(ab)*")
    assert_language(literal("c") + literal("ab").kleene_star(), r"c(ab)*")


def test_operations_are_lambda_free():
    nfa = (literal("a") | literal("b")).kleene_star() + literal("c").option()
    assert not nfa.has_lambda()
    assert_language(nfa, r"[ab]*c?")


@pytest.mark.parametrize(
    "ws_kleene,pattern",
    [
        ("star", r"a( *, *a)*"),
        ("plus", r"a( +, +a)*"),
        ("none", r"a(,a)*"),
    ],
)
def test_recursive_concat(ws_kleene, pattern):
    nfa = literal("a").recursive_concat(",", ws=[" "], ws_kleene=ws_kleene)
    assert_language(nfa, pattern, alphabet="a, ", max_len=7)


def test_recursive_concat_alternative_separators():
    nfa = literal("a").recursive_concat([",", "b"], ws=[" "], ws_kleene="none")
    assert_language(nfa, r"a((,|b)a)*", alphabet="ab,", max_len=6)


# ---- normalization ----


def test_dfa_minify_keeps_language():
    nfa = (literal("ab") | literal("ac") | literal("ab")).kleene_star()
    assert_language(nfa.dfa_minify(), r"(ab|ac)*")


def test_canonicalize_keeps_language_and_numbers_from_zero():
    nfa = (literal("a") | literal("bc")).kleene_star().dfa_minify()
    canon = nfa.canonicalize()
    assert_language(canon, r"(a|bc)*")
    assert canon.initial_state == 0
    assert canon.states == set(range(len(canon.states)))


def test_canonicalize_is_independent_of_construction():
    one = (literal("ab") | literal("ac")).dfa_minify().canonicalize()
    two = (literal("a") + any_of(["c", "b"])).dfa_minify().canonicalize()
    assert one.transitions == two.transitions
    assert one.final_states == two.final_states
    assert one.canonicalize().transitions == one.transitions


def test_from_regex_ranges():
    nfa = ConstraintNFA.from_regex("(1-3)(0-9)*")
    assert nfa.accepts_input("1")
    assert nfa.accepts_input("305")
    assert not nfa.accepts_input("05")
    assert not nfa.accepts_input("4")
