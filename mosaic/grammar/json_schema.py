"""JSON and tool-call constraints built from a JSON Schema.

``JSONBuilder`` turns a schema into an automaton accepting exactly the JSON
values that satisfy it, and wraps that in whichever tool-call syntax the
prompt asks for: a JSON array of ``{"name": ..., "arguments": {...}}`` objects,
or BFCL's Python-call list ``[f(a=1), g(b="x")]``. A typed schema bounds its
own recursion; an empty sub-schema means "any JSON value", so that is bounded
by ``max_depth``.
"""

import json

from .automata import ConstraintNFA as NFA, any_of, literal, symbol

#: Tool-call syntaxes with a constraint automaton.
TEMPLATE_TYPES = ("json", "python")


class JSONBuilder:
    """Builds constraint automata for JSON values and tool calls.

    ``input_symbols`` is the alphabet the automaton is written over.
    ``eot_token`` then zero or more ``eos_token`` may follow the calls (a
    diffusion LM fills its whole canvas, so the tail pads it out); both, and
    anything in ``banned_tokens``, are excluded from string contents so they
    cannot appear inside a generated value. ``eot_token`` defaults to
    ``eos_token``.
    """

    def __init__(
        self,
        input_symbols,
        max_depth=3,
        eos_token=None,
        eot_token=None,
        template_type="json",
        banned_tokens=None,
    ):
        if template_type not in TEMPLATE_TYPES:
            raise ValueError(
                f"No constraint automaton for template_type "
                f"{template_type!r}; expected one of {TEMPLATE_TYPES}"
            )
        banned_tokens = list(banned_tokens or [])
        for token in (eos_token, eot_token):
            if token is not None:
                input_symbols.add(token)
                banned_tokens.append(token)
        for token in banned_tokens:
            input_symbols.add(token)

        self.input_symbols = input_symbols
        self.max_depth = max_depth
        self.eos_token = eos_token
        self.eot_token = eos_token if eot_token is None else eot_token
        self.banned_tokens = banned_tokens
        self.template_type = template_type

    # ---- entry point ----

    def build_tool_calls(self, tool):
        """A sequence of calls to ``tool``, in the current template's syntax.

        ``tool`` maps each function name to its parameter schema. The
        end-of-sequence tail, if there is one, may only follow the calls. The
        result is a minimized DFA with canonical state numbering, so the same
        schema always compiles to the same automaton.
        """
        if self.template_type == "json":
            nfa = self.tool_calls_json(tool)
        else:
            nfa = self.tool_calls_python(tool)
        return self._finish(nfa)

    def build_value(self, schema):
        """A single JSON value satisfying ``schema``, for structured output without tools."""
        return self._finish(self.json_value(schema))

    def _finish(self, nfa):
        """Append the end-of-sequence tail, minimize, and number states canonically."""
        if self.eos_token is not None:
            tail = symbol(self.eot_token) + symbol(self.eos_token).kleene_star()
            nfa = nfa + tail.dfa_minify()
        return nfa.dfa_minify().canonicalize()

    # ---- whitespace ----

    @property
    def ws_list(self):
        """Whitespace characters allowed between tokens."""
        return [" ", "\t", "\n"]

    @property
    def ws_one(self):
        """One whitespace character."""
        return any_of(self.ws_list).dfa_minify()

    @property
    def wstar(self):
        """Zero or more whitespace characters."""
        return self.ws_one.kleene_star().dfa_minify()

    def spaced(self, text):
        """``text`` with optional whitespace on either side."""
        return (self.wstar + literal(text) + self.wstar).dfa_minify()

    def spaced_all(self, texts):
        """``texts`` in order, with optional whitespace between and around them."""
        nfa = self.wstar
        for text in texts:
            nfa = nfa + literal(text) + self.wstar
        return nfa.dfa_minify()

    # ---- primitive types ----

    def type_probability(self):
        """A written probability, ``{"type": "number", "x-prob": true}``: 0, 0.d, 0.dd, 1, 1.0 or 1.00."""
        digit = any_of(list("0123456789"))
        zero = literal("0") + (literal(".") + digit + digit.option()).option()
        one = literal("1") + (literal(".") + literal("0") + literal("0").option()).option()
        return (zero | one).dfa_minify()

    def type_plain_string(self, max_length=None):
        """A string without escapes, ``{"type": "string", "x-plain": true}`` (at most ``maxLength`` characters).

        For free text such as a reason written before an answer: with no
        escapes, the model cannot write a quoted answer inside it; a quote
        always closes it.
        """
        quote = '"'
        banned = {chr(i) for i in range(32)} | {quote, "\\"} | set(self.banned_tokens)
        chars = {c for c in self.input_symbols if c not in banned}
        n = max_length
        if n is None:  # 0 --"--> 1 --chars*--> 1 --"--> 2
            transitions = {0: {quote: {1}}, 1: {c: {1} for c in chars} | {quote: {2}}, 2: {}}
            states, final = {0, 1, 2}, 2
        else:  # state 1 + k after k characters, at most n of them
            final = n + 2
            transitions = {0: {quote: {1}}, final: {}}
            for k in range(n + 1):
                transitions[1 + k] = {quote: {final}} | ({c: {2 + k} for c in chars} if k < n else {})
            states = set(transitions)
        return NFA(
            states=states, input_symbols=self.input_symbols, transitions=transitions,
            initial_state=0, final_states={final},
        ).dfa_minify()

    def type_string(self):
        """A JSON string, including escapes and ``\\uXXXX`` sequences.

        Control characters, the bare quote, the end-of-sequence markers and any
        banned tokens cannot appear in the contents.
        """
        quote_symbol = '"'
        escape_symbol = "\\"
        input_symbols = self.input_symbols

        not_allowed_chars = {chr(i) for i in range(32)} | {quote_symbol}
        for token in self.banned_tokens:
            not_allowed_chars.add(token)
        escape_next_chars = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
        hex_digits = set("0123456789abcdefABCDEF")
        # States
        # 0 = start
        # 1 = inside string
        # 2 = final quote
        # 3 = escape seen
        # 4-7 = \uXXXX hex digits
        states = {0, 1, 2, 3, 4, 5, 6, 7}
        transitions = {
            0: {quote_symbol: {1}},
            1: {
                c: {1}
                for c in input_symbols
                if c not in not_allowed_chars | {escape_symbol}
            }
            | {quote_symbol: {2}}
            | {escape_symbol: {3}},
            2: {},
            3: {c: {1} for c in escape_next_chars if c != "u"} | {"u": {4}},
            4: {c: {5} for c in hex_digits},
            5: {c: {6} for c in hex_digits},
            6: {c: {7} for c in hex_digits},
            7: {c: {1} for c in hex_digits},  # after full \uXXXX, return to string
        }
        return NFA(
            states=states,
            input_symbols=input_symbols,
            transitions=transitions,
            initial_state=0,
            final_states={2},
        ).dfa_minify()

    def type_number(self):
        """A JSON number, with optional fraction and exponent."""
        digits = NFA.from_regex("(0-9)+")
        nfa = (
            literal("-").option()
            + (literal("0") | NFA.from_regex("(1-9)(0-9)*"))
            + (literal(".") + digits).option()
            + (any_of(["e", "E"]) + any_of(["+", "-"]).option() + digits).option()
        )
        return nfa.dfa_minify()

    def type_integer(self):
        """A JSON integer, with no fraction or exponent."""
        nfa = literal("-").option() + (
            literal("0") | NFA.from_regex("(1-9)(0-9)*")
        )
        return nfa.dfa_minify()

    def type_boolean(self):
        """``true`` / ``false``, or ``True`` / ``False`` in the Python template."""
        if self.template_type == "python":
            return any_of(["True", "False"]).dfa_minify()
        return any_of(["true", "false"]).dfa_minify()

    def type_null(self):
        """``null``, or ``None`` in the Python template."""
        if self.template_type == "python":
            return literal("None")
        return literal("null")

    def type_enum(self, values):
        """Exactly one of the JSON-encoded ``values``.

        Returns None when a value has a non-ASCII character: the tokenizer
        conversion indexes single-character symbols by code point within a
        byte-sized alphabet, so such a value falls back to its type instead.
        """
        rendered = []
        for value in values:
            text = json.dumps(value, ensure_ascii=False)
            if any(ord(c) > 127 for c in text):
                return None
            rendered.append(text)
        return any_of(rendered).dfa_minify()

    def json_primitive(self):
        """Any JSON value that is not an array or an object."""
        nfa = (
            self.type_string()
            | self.type_number()
            | self.type_integer()
            | self.type_boolean()
            | self.type_null()
        )
        return nfa.dfa_minify()

    # ---- composite types ----

    def type_array(self, schema, min_items=0):
        """A JSON array whose elements satisfy ``schema``, at least ``min_items`` of them."""
        elem = self.json_value(schema)
        items = elem.recursive_concat(",", ws=self.ws_list, ws_kleene="star")
        if min_items == 0:
            items = items.option()
        for _ in range(min_items - 1):
            items = elem + self.spaced(",") + items
        nfa = self.spaced("[") + items + self.spaced("]")
        return nfa

    def _property_sequence(self, properties, required, build_pair):
        """Key-value pairs in declaration order, at least one of them present.

        Optional properties may be skipped, so the separating comma is part of
        the optional group. Returns the automaton and whether every property so
        far was optional, which the caller needs to allow an empty object.
        """
        if len(properties) == 1:
            name, subschema = properties[0]
            return build_pair(name, subschema), name not in required

        nfa_prefix, opt_prefix = self._property_sequence(
            properties[:-1], required, build_pair
        )
        name, subschema = properties[-1]
        nfa_end = build_pair(name, subschema)
        if name not in required:
            nfa = nfa_prefix + (self.spaced(",") + nfa_end).option()
        else:
            nfa = nfa_prefix + self.spaced(",") + nfa_end
        if opt_prefix:
            nfa = nfa | nfa_end
        return nfa, opt_prefix and name not in required

    def type_object(self, properties, required):
        """A JSON object, with key-value pairs in the schema's declared order.

        An empty ``properties`` means any object, so keys are unconstrained
        strings and pairs may repeat.
        """

        def build_pair(name, subschema):
            """One ``"key": value`` pair."""
            key = any_of(['"' + name + '"'])
            value = self.json_value(subschema)
            return key + self.spaced(":") + value

        if properties == {}:
            pair = self.type_string() + self.spaced(":") + self.json_value({})
            return (
                self.spaced("{")
                + pair.recursive_concat(",", ws=self.ws_list, ws_kleene="star").option()
                + self.spaced("}")
            )

        nfa, opt_prefix = self._property_sequence(
            list(properties.items()), required, build_pair
        )
        nfa = self.spaced("{") + nfa + self.spaced("}")
        if opt_prefix:
            assert len(required) == 0
            nfa = nfa | self.spaced_all(["{", "}"])
        return nfa

    # ---- full JSON ----

    def json_value(self, schema):
        """A JSON value satisfying ``schema``.

        A typed schema bounds its own recursion, since every step descends into
        a strictly smaller sub-schema. An empty schema means any JSON value, so
        that expansion is bounded by :meth:`any_json` instead. ``anyOf`` /
        ``oneOf`` (a union of the branches), ``const`` and ``enum`` take
        precedence over ``type``; a list of types is their union.
        """
        if schema == {}:
            return self.any_json()

        # mosaic's extension keywords, ahead of ``type``
        if schema.get("x-prob"):
            return self.type_probability()
        if schema.get("x-plain"):
            return self.type_plain_string(schema.get("maxLength"))

        branches = schema.get("anyOf") or schema.get("oneOf")
        if branches:
            return NFA.union_all([self.json_value(branch) for branch in branches])

        if "const" in schema:
            nfa = self.type_enum([schema["const"]])
            if nfa is not None:
                return nfa

        if schema.get("enum"):
            nfa = self.type_enum(schema["enum"])
            if nfa is not None:
                return nfa

        schema_type = schema.get("type", None)
        if isinstance(schema_type, list):
            return NFA.union_all([self.json_value({**schema, "type": t}) for t in schema_type])
        if schema_type is None and "properties" in schema:
            schema_type = "object"
        elif schema_type is None and "items" in schema:
            schema_type = "array"
        assert schema_type in [
            "null",
            "boolean",
            "string",
            "number",
            "integer",
            "array",
            "object",
        ], f"Unsupported type: {schema_type}"
        if schema_type == "null":
            return self.type_null()
        elif schema_type == "boolean":
            return self.type_boolean()
        elif schema_type == "string":
            return self.type_string()
        elif schema_type == "number":
            return self.type_number()
        elif schema_type == "integer":
            return self.type_integer()
        elif schema_type == "array":
            return self.type_array(schema.get("items", {}), schema.get("minItems", 0))
        else:
            return self.type_object(
                schema.get("properties", {}), schema.get("required", [])
            )

    def any_json(self, depth=None):
        """Any JSON value, nesting at most ``max_depth`` deep.

        Unlike a typed schema there is nothing to descend into, so the
        expansion would not terminate on its own.
        """
        depth = self.max_depth if depth is None else depth
        nfa = self.json_primitive()
        if depth > 0:
            inner = self.any_json(depth - 1)
            array = (
                self.spaced("[")
                + inner.recursive_concat(
                    ",", ws=self.ws_list, ws_kleene="star"
                ).option()
                + self.spaced("]")
            )
            pair = self.type_string() + self.spaced(":") + inner
            obj = (
                self.spaced("{")
                + pair.recursive_concat(",", ws=self.ws_list, ws_kleene="star").option()
                + self.spaced("}")
            )
            nfa = nfa | array | obj
        return nfa

    # ---- tool calls ----

    def tool_calls_json(self, tool):
        """Tool calls as a JSON array: ``[{"name": ..., "arguments": {...}}]``."""
        nfa_list = []
        tool_list = tool.items() if isinstance(tool, dict) else tool
        for func_name, schema in tool_list:
            nfa = self.json_value(schema)
            left = self.spaced_all(
                ["{", '"name"', ":", f'"{func_name}"', ",", '"arguments"', ":"]
            ).dfa_minify()
            right = self.spaced_all(["}"]).dfa_minify()
            nfa_list.append(left + nfa + right)
        nfa = NFA.union_all(nfa_list)
        return (
            self.spaced("[")
            + nfa.recursive_concat(",", ws=self.ws_list, ws_kleene="star")
            + self.spaced("]")
        )

    def tool_calls_python(self, tool):
        """Tool calls as BFCL's Python list: ``[f(a=1, b="x"), g()]``.

        Argument values are JSON values, except that booleans and null use the
        Python spelling. ``f()`` is allowed when every parameter is optional.
        """

        def build_pair(name, subschema):
            """One ``param=value`` argument."""
            key = any_of([name])
            value = self.json_value(subschema)
            return key + self.spaced("=") + value

        nfa_list = []
        tool_list = tool.items() if isinstance(tool, dict) else tool
        for func_name, schema in tool_list:
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            if not properties:
                nfa = self.spaced(func_name) + self.spaced_all(["(", ")"])
            else:
                args, opt_prefix = self._property_sequence(
                    list(properties.items()), required, build_pair
                )
                args = self.spaced("(") + args + self.spaced(")")
                if opt_prefix:
                    args = args | self.spaced_all(["(", ")"])
                nfa = self.spaced(func_name) + args
            nfa_list.append(nfa)
        nfa = NFA.union_all(nfa_list)
        return (
            self.spaced("[")
            + nfa.recursive_concat(",", ws=self.ws_list, ws_kleene="star")
            + self.spaced("]")
        )
