"""Accept/reject checks for the JSON-schema and tool-call automata.

Most tests run at the character level. The last group converts an automaton to
the Dream tokenizer's vocabulary and walks real token ids through it; it is
skipped when the tokenizer is not available locally.
"""

import json

import pytest

from mosaic.grammar.json_schema import JSONBuilder
from mosaic.grammar.tokenize import token_graph_from_char_nfa

EOS = "<|endoftext|>"

WEATHER = {
    "get_weather": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    }
}


def builder(template="json", eos=None, max_depth=3):
    return JSONBuilder(
        set(map(chr, range(256))),
        max_depth=max_depth,
        eos_token=eos,
        template_type=template,
    )


def call_nfa(tool, template="json", eos=None):
    return builder(template, eos).build_tool_calls(tool)


def value_nfa(schema, **kwargs):
    return builder(**kwargs).json_value(schema).dfa_minify()


def accepts(nfa, text, tail=()):
    return nfa.accepts_input(list(text) + list(tail))


# ---- JSON values ----


@pytest.mark.parametrize(
    "text,ok",
    [
        ('"hello"', True),
        ('""', True),
        ('"a \\"quoted\\" word"', True),
        ('"\\u00e9\\n"', True),
        ('"bad \\x escape"', False),
        ('"raw\nnewline"', False),
        ('"unterminated', False),
        ("hello", False),
    ],
)
def test_string(text, ok):
    assert accepts(value_nfa({"type": "string"}), text) == ok


@pytest.mark.parametrize(
    "text,ok",
    [("0", True), ("-12", True), ("3.25", True), ("-1.5e-3", True), ("2E+10", True),
     ("01", False), ("1.", False), (".5", False), ("+1", False)],
)
def test_number(text, ok):
    assert accepts(value_nfa({"type": "number"}), text) == ok


@pytest.mark.parametrize("text,ok", [("7", True), ("-7", True), ("7.0", False), ("007", False)])
def test_integer(text, ok):
    assert accepts(value_nfa({"type": "integer"}), text) == ok


def test_boolean_and_null():
    assert accepts(value_nfa({"type": "boolean"}), "true")
    assert not accepts(value_nfa({"type": "boolean"}), "True")
    assert accepts(value_nfa({"type": "null"}), "null")


def test_enum_takes_precedence_over_type():
    nfa = value_nfa({"type": "string", "enum": ["a", "b c"]})
    assert accepts(nfa, '"a"')
    assert accepts(nfa, '"b c"')
    assert not accepts(nfa, '"c"')


def test_non_ascii_enum_falls_back_to_type():
    nfa = value_nfa({"type": "string", "enum": ["café"]})
    assert accepts(nfa, '"anything"')


def test_array():
    nfa = value_nfa({"type": "array", "items": {"type": "integer"}})
    for text in ["[]", "[1]", "[1, 2,3]", "[ 1 ,\n2 ]"]:
        assert accepts(nfa, text), text
    for text in ["[1,]", "[1 2]", '["a"]']:
        assert not accepts(nfa, text), text


def test_object_properties_keep_declared_order():
    schema = WEATHER["get_weather"]
    nfa = value_nfa(schema)
    assert accepts(nfa, '{"city": "Paris"}')
    assert accepts(nfa, '{"city": "Paris", "days": 3, "unit": "celsius"}')
    assert accepts(nfa, '{"city":"Paris","unit":"fahrenheit"}')
    assert not accepts(nfa, '{"days": 3}')                        # missing required
    assert not accepts(nfa, '{"days": 3, "city": "Paris"}')       # out of order
    assert not accepts(nfa, '{"city": "Paris", "extra": 1}')      # unknown key
    assert not accepts(nfa, '{"city": "Paris", "unit": "kelvin"}')  # not in enum


def test_object_all_optional_allows_empty():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": []}
    nfa = value_nfa(schema)
    assert accepts(nfa, "{}")
    assert accepts(nfa, '{"a": 1}')


def test_empty_schema_is_any_json_up_to_max_depth():
    nfa = value_nfa({}, max_depth=2)
    for text in ['1', '"s"', 'null', '[1, "a"]', '{"k": [true]}', '[[1]]']:
        assert accepts(nfa, text), text
    assert not accepts(nfa, "[[[1]]]")


# ---- tool calls ----


def test_json_calls():
    nfa = call_nfa(WEATHER)
    one = {"name": "get_weather", "arguments": {"city": "Paris"}}
    two = {"name": "get_weather", "arguments": {"city": "Rome", "days": 2}}
    assert accepts(nfa, json.dumps([one]))
    assert accepts(nfa, json.dumps([one, two], indent=2))
    assert not accepts(nfa, "[]")
    assert not accepts(nfa, json.dumps(one))
    assert not accepts(nfa, json.dumps([{"name": "other", "arguments": {"city": "x"}}]))


def test_json_calls_choose_between_functions():
    tool = dict(WEATHER, ping={"type": "object", "properties": {}, "required": []})
    nfa = call_nfa(tool)
    assert accepts(nfa, '[{"name": "ping", "arguments": {}}]')
    assert accepts(nfa, '[{"name": "ping", "arguments": {"any": [1]}}, '
                        '{"name": "get_weather", "arguments": {"city": "x"}}]')


def test_eos_tail_only_after_the_calls():
    nfa = call_nfa(WEATHER, eos=EOS)
    text = '[{"name": "get_weather", "arguments": {"city": "Paris"}}]'
    assert not accepts(nfa, text)
    assert accepts(nfa, text, tail=[EOS])
    assert accepts(nfa, text, tail=[EOS, EOS, EOS])
    inside = '[{"name": "get_weather", "arguments": {"city": "Pa'
    assert not accepts(nfa, inside, tail=[EOS] + list('ris"}}]') + [EOS])


def test_eot_then_eos():
    b = JSONBuilder(set(map(chr, range(256))), eos_token="<eos>", eot_token="<eot>")
    nfa = b.build_tool_calls(WEATHER)
    text = '[{"name": "get_weather", "arguments": {"city": "Paris"}}]'
    assert accepts(nfa, text, tail=["<eot>"])
    assert accepts(nfa, text, tail=["<eot>", "<eos>", "<eos>"])
    assert not accepts(nfa, text, tail=["<eos>"])
    assert not accepts(nfa, text, tail=["<eot>", "<eot>"])


def test_python_calls_use_python_literals():
    tool = {
        "search": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "exact": {"type": "boolean"},
                "limit": {},
            },
            "required": ["query"],
        },
        "ping": {"type": "object", "properties": {}, "required": []},
    }
    nfa = call_nfa(tool, template="python")
    assert accepts(nfa, '[search(query="x")]')
    assert accepts(nfa, '[search(query="x", exact=True), ping()]')
    assert accepts(nfa, '[search(query="x", limit=None)]')
    assert accepts(nfa, '[search(query="x",\n  limit=[1, False])]')
    assert not accepts(nfa, '[search(query="x", exact=true)]')
    assert not accepts(nfa, '[search(query="x", limit=null)]')
    assert not accepts(nfa, '[search(exact=True)]')


def test_build_value():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }
    nfa = builder(eos=EOS).build_value(schema)
    assert accepts(nfa, '{"name": "Alice", "age": 31}', tail=[EOS])
    assert accepts(nfa, '{"name": "Bob"}', tail=[EOS, EOS])
    assert not accepts(nfa, '{"age": 31}', tail=[EOS])
    assert not accepts(nfa, '[{"name": "Alice"}]', tail=[EOS])
    assert nfa.initial_state == 0


def test_build_is_deterministic():
    one = call_nfa(WEATHER, eos=EOS)
    two = call_nfa(WEATHER, eos=EOS)
    assert one.transitions == two.transitions
    assert one.initial_state == two.initial_state == 0


# ---- token level (Dream tokenizer) ----


@pytest.fixture(scope="module")
def dream_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True
        )
    except Exception as e:  # offline, no cache
        pytest.skip(f"Dream tokenizer unavailable: {e}")


def walk(graph, token_ids):
    """Whether ``graph`` accepts ``token_ids`` (graph dict from tokenize.py)."""
    out = {}
    for u, v, mask in graph["edges"]:
        out.setdefault(u, []).append((v, mask))
    states = {graph["initial_state"]}
    for tok in token_ids:
        states = {v for s in states for v, mask in out.get(s, []) if mask[tok]}
        if not states:
            return False
    return bool(states & set(graph["accept_states"]))


def test_token_level_accepts_real_tokenizations(dream_tokenizer):
    tok = dream_tokenizer
    eos = tok.eos_token
    b = JSONBuilder(
        set(map(chr, range(256))), eos_token=eos,
        banned_tokens=[eos, "<|im_start|>", "<|im_end|>"],
    )
    graph = token_graph_from_char_nfa(b.build_tool_calls(WEATHER), tok)
    eos_id = tok.convert_tokens_to_ids(eos)

    def ids(text):
        return tok.encode(text, add_special_tokens=False)

    good = '[{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}]'
    assert walk(graph, ids(good) + [eos_id, eos_id])
    assert not walk(graph, ids(good))
    assert not walk(graph, ids('[{"name": "get_weather", "arguments": {}}]') + [eos_id])
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    bad = ids('[{"name": "get_weather", "arguments": {"city": "Pa') + [im_end]
    assert not walk(graph, bad + ids('"}}]') + [eos_id])


# ---- standard JSON-Schema keywords for tool calls ----


def test_any_of_const_min_items():
    schema = {
        "type": "array",
        "minItems": 2,
        "items": {"anyOf": [{"const": "x"}, {"type": "integer"}]},
    }
    nfa = value_nfa(schema)
    assert accepts(nfa, '["x", 3]')
    assert accepts(nfa, '[1, 2, "x"]')
    assert not accepts(nfa, '["x"]')          # minItems
    assert not accepts(nfa, '["y", 1]')       # not the const
    assert accepts(value_nfa({"type": ["integer", "null"]}), "null")


def tool_calls_schema(tool):
    """The JSON-array tool-call format as a standard JSON Schema."""
    return {
        "type": "array",
        "minItems": 1,
        "items": {"anyOf": [
            {
                "type": "object",
                "properties": {"name": {"const": name}, "arguments": params},
                "required": ["name", "arguments"],
            }
            for name, params in tool.items()
        ]},
    }


def test_tool_calls_as_json_schema_equal_the_tool_builder():
    # Canonical minimal DFAs are unique: same language -> identical automaton.
    tool = dict(WEATHER, ping={"type": "object", "properties": {}, "required": []})
    direct = builder(eos=EOS).build_tool_calls(tool)
    via_schema = builder(eos=EOS).build_value(tool_calls_schema(tool))
    assert direct.transitions == via_schema.transitions
    assert direct.final_states == via_schema.final_states
