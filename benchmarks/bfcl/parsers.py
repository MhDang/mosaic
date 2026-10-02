"""Parsers from generated text to tool calls, one per output template.

Each returns a list of ``{"name": str, "arguments": dict}`` on success, or a
string starting with ``"Invalid format"`` describing why it failed.
"""

import ast
import json


def parse_json_calls(text):
    """Parse a JSON array of tool calls: ``[{"name": ..., "arguments": {...}}]``."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return "Invalid format: JSONDecodeError - " + str(e)


def parse_python_calls(text):
    """Parse BFCL's Python call list: ``[f(a=1, b="x"), g()]``.

    Accepts Python booleans/None (BFCL's spec) as well as JSON ``true`` /
    ``false`` / ``null``; both come out as Python values.
    """
    text = text.strip()
    if not text:
        return "Invalid format: empty input"
    # Strip a single ```python ... ``` fence if present.
    if text.startswith("```"):
        nl = text.find("\n")
        if nl > 0:
            text = text[nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        return f"Invalid format: SyntaxError - {e}"

    _coerce_json_literals(tree)
    if not isinstance(tree.body, ast.List):
        return "Invalid format: expected top-level list expression"
    calls = []
    for elt in tree.body.elts:
        if not isinstance(elt, ast.Call):
            return "Invalid format: list element is not a function call"
        # Reconstruct a dotted name.
        node = elt.func
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return "Invalid format: bad function reference"
        parts.append(node.id)
        name = ".".join(reversed(parts))
        if elt.args:
            return f"Invalid format: positional args not allowed in {name}(...)"
        args = {}
        for kw in elt.keywords:
            if kw.arg is None:
                return f"Invalid format: **kwargs not allowed in {name}(...)"
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except Exception as e:
                return f"Invalid format: cannot eval value of {kw.arg}: {e}"
        calls.append({"name": name, "arguments": {k: _desetify(v) for k, v in args.items()}})
    return calls


PARSERS = {"json": parse_json_calls, "python": parse_python_calls}


def to_bfcl_format(model_calls):
    """``[{name, arguments}, ...]`` -> BFCL's ``[{name: arguments}, ...]``; None if malformed."""
    if not isinstance(model_calls, list):
        return None
    out = []
    for c in model_calls:
        if not isinstance(c, dict):
            return None
        name = c.get("name")
        args = c.get("arguments", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        out.append({name: args})
    return out


# ---- helpers ----

_JSON_BOOL_NULL = {"true": True, "false": False, "null": None}


def _coerce_json_literals(parent):
    """Rewrite ``true`` / ``false`` / ``null`` names into constants, in place."""
    for field, value in ast.iter_fields(parent):
        if isinstance(value, ast.Name) and value.id in _JSON_BOOL_NULL:
            setattr(parent, field, ast.copy_location(
                ast.Constant(value=_JSON_BOOL_NULL[value.id]), value))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, ast.Name) and item.id in _JSON_BOOL_NULL:
                    value[i] = ast.copy_location(
                        ast.Constant(value=_JSON_BOOL_NULL[item.id]), item)
                elif isinstance(item, ast.AST):
                    _coerce_json_literals(item)
        elif isinstance(value, ast.AST):
            _coerce_json_literals(value)


def _desetify(v):
    """Turn set literals into lists (sorted by repr) so results stay JSON-serializable."""
    if isinstance(v, set):
        return [_desetify(x) for x in sorted(v, key=repr)]
    if isinstance(v, dict):
        return {k: _desetify(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_desetify(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_desetify(x) for x in v)
    return v
