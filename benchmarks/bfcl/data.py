"""BFCL (Berkeley Function Calling Leaderboard), the AST single-turn splits.

Prompts use BFCL's official system prompt, with one of two output formats:
``template=json`` asks for a JSON array ``[{"name": ..., "arguments": {...}}]``,
``template=python`` for BFCL's native ``[f(a=1), g(b="x")]``. The constraint is
the function list's JSON schemas compiled to an automaton; scoring is BFCL's
official AST checker.
"""

import json
import os

from mosaic.grammar.json_schema import JSONBuilder
from mosaic.tokenizers import resolve_tokenizer_tokens, tokenizer_key

from .checker import ast_checker
from .parsers import PARSERS, to_bfcl_format

#: Where ``benchmarks/bfcl/download.sh`` puts the data.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "data", "bfcl")

#: BFCL split -> file stem (simple_python is the only AST-Python simple split).
SPLIT_FILES = {
    "simple": "simple_python",
    "multiple": "multiple",
    "parallel": "parallel",
    "parallel_multiple": "parallel_multiple",
    "live_simple": "live_simple",
    "live_multiple": "live_multiple",
    "live_parallel": "live_parallel",
    "live_parallel_multiple": "live_parallel_multiple",
}

# BFCL's official system prompt (bfcl_eval default: ret_fmt=python,
# prompt_fmt=plaintext, style=classic), with a {functions} slot.
_SYSTEM_PROMPT_PYTHON = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in JSON format that you can invoke.
{functions}
"""

# The same prompt with only the format-instruction line changed to a JSON array.
_SYSTEM_PROMPT_JSON = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of a JSON array: [{{"name": "func_name1", "arguments": {{"params_name1": params_value1, "params_name2": params_value2}}}}, {{"name": "func_name2", "arguments": {{...}}}}]
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in JSON format that you can invoke.
{functions}
"""

SYSTEM_PROMPTS = {"json": _SYSTEM_PROMPT_JSON, "python": _SYSTEM_PROMPT_PYTHON}


class BFCLDataset:
    """One BFCL AST split: its examples, prompts, constraint automata and scoring.

    ``template`` is the call format (json | python); ``start`` / ``end`` slice
    the examples and ``example_ids`` keeps only those.
    """

    def __init__(self, tokenizer, split, template="json", data_dir=DATA_DIR, start=0, end=None, example_ids=None):
        if split not in SPLIT_FILES:
            raise ValueError(f"Unknown BFCL split {split!r}; AST splits: {list(SPLIT_FILES)}")
        if template not in SYSTEM_PROMPTS:
            raise ValueError(f"template must be one of {list(SYSTEM_PROMPTS)}, got {template!r}")
        self.tokenizer = tokenizer
        self.template = template
        self.examples = load_split(data_dir, split)[start:end]
        if example_ids:
            keep = set(example_ids)
            self.examples = [ex for ex in self.examples if ex["example_idx"] in keep]
        eos_token, eot_token, banned_tokens = resolve_tokenizer_tokens(tokenizer)
        self.builder = JSONBuilder(
            set(map(chr, range(256))),
            max_depth=3,
            eos_token=eos_token,
            eot_token=eot_token,
            banned_tokens=banned_tokens,
            template_type=template,
        )

    # ---- prompt ----

    def build_prompt(self, example):
        """Function docs in the system message; the question's messages after it."""
        system = SYSTEM_PROMPTS[self.template].format(
            functions=json.dumps(example["function"], indent=4)
        )
        question = example["question"]
        user_messages = question[0] if isinstance(question, list) and question else []
        # Some questions open with their own system message; BFCL appends it.
        if user_messages and user_messages[0].get("role") == "system":
            system = system + "\n\n" + user_messages[0]["content"]
            user_messages = user_messages[1:]
        messages = [{"role": "system", "content": system}] + list(user_messages)
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    # ---- constraint ----

    def compile_constraint(self, compiler, example):
        """Calls to this example's functions, in the template's format: the automaton from
        ``compiler``'s caches or compiled into them (a :class:`~mosaic.matcher.CompiledConstraint`)."""
        return compiler.compile_grammar(lambda: self.builder.build_tool_calls(tool_schema(example["function"])),
                                        key=self.constraint_key(example))

    def constraint_key(self, example):
        """The automaton's cache key (a relative path; it names the tokenizer too)."""
        return os.path.join(
            f"bfcl_{self.template}_{tokenizer_key(self.tokenizer)}", example["split"], str(example["example_idx"])
        )

    # ---- scoring ----

    def post_process(self, example, generations):
        """Parse each generation and score it with BFCL's AST checker.

        res_list_raw per generation: 1 correct, -1 parsed but wrong, 0 unparsable.
        """
        parser = PARSERS[self.template]
        res_list, responses, failure_reasons = [], [], []
        for text in generations:
            calls = parser(text)
            if isinstance(calls, str):
                res, resp, why = 0, "", calls
            elif not isinstance(calls, list):
                res, resp = 0, calls
                why = f"Invalid format: expected list, got {type(calls).__name__}"
            elif (bfcl_calls := to_bfcl_format(calls)) is None:
                res, resp, why = 0, calls, "Invalid format: cannot convert to BFCL native format"
            else:
                try:
                    check = ast_checker(
                        example["function"], bfcl_calls, example["ground_truth"], example["split"]
                    )
                except Exception as e:
                    check = {"valid": False, "error": [f"checker error: {e}"]}
                res, resp = (1, calls) if check.get("valid") else (-1, calls)
                why = "" if res == 1 else f"Mismatch (official): {check.get('error')}"
            res_list.append(res)
            responses.append(resp)
            failure_reasons.append(why)

        return {
            "success_rate": res_list.count(1) / len(res_list) if res_list else 0.0,
            "generated_count": len(res_list),
            "success_count": res_list.count(1),
            "parsing_failure_count": res_list.count(0),
            "ans_failure_count": res_list.count(-1),
            "failure_reasons": failure_reasons,
            "res_list_raw": res_list,
            "model_responses_json": responses,
        }


def load_split(data_dir, split):
    """The split's questions, each with its possible answers."""
    fname = f"BFCL_v4_{SPLIT_FILES[split]}.json"
    with open(os.path.join(data_dir, fname)) as f:
        questions = [json.loads(line) for line in f]
    with open(os.path.join(data_dir, "possible_answer", fname)) as f:
        answers = {a["id"]: a for a in map(json.loads, f)}
    return [
        {
            "id": q["id"],
            "example_idx": q["id"],
            "split": split,
            "function": q["function"],
            "question": q["question"],
            "ground_truth": answers.get(q["id"], {}).get("ground_truth", []),
        }
        for q in questions
    ]


# ---- schemas ----


def tool_schema(functions):
    """Function name -> JSON-Schema parameter object, for :meth:`JSONBuilder.build_tool_calls`."""
    out = {}
    for fn in functions:
        params = normalize_bfcl_schema(fn.get("parameters", {}) or {})
        if params.get("type") != "object":
            params = {"type": "object", "properties": {}, "required": []}
        out[fn["name"]] = params
    return out


def tool_calls_json_schema(functions):
    """The JSON-array call format (template ``json``) as a standard JSON Schema.

    It compiles to the same automaton as :meth:`JSONBuilder.build_tool_calls` on
    :func:`tool_schema`, for engines that take one JSON Schema per request.
    """
    return {
        "type": "array",
        "minItems": 1,
        "items": {"anyOf": [
            {
                "type": "object",
                "properties": {"name": {"const": name}, "arguments": params},
                "required": ["name", "arguments"],
            }
            for name, params in tool_schema(functions).items()
        ]},
    }


def normalize_bfcl_schema(schema):
    """Map BFCL's parameter schemas onto the JSON-Schema subset the builder accepts.

    BFCL uses Python-flavoured type names (``dict``, ``float``, ``tuple``,
    ``any``) and leaves ``required`` implicit: a parameter is optional iff it
    has a ``default``.
    """
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    t = out.get("type")
    if t == "dict":
        out["type"] = "object"
    elif t == "float":
        out["type"] = "number"
    elif t == "tuple":
        out["type"] = "array"
    elif t == "any":
        out.pop("type", None)
    if "properties" in out:
        out["properties"] = {k: normalize_bfcl_schema(v) for k, v in out["properties"].items()}
    if "items" in out and isinstance(out["items"], dict):
        out["items"] = normalize_bfcl_schema(out["items"])
    if "properties" in out and "required" not in out:
        out["required"] = [
            k for k, v in out["properties"].items()
            if not (isinstance(v, dict) and "default" in v)
        ]
    out.pop("description", None)
    return out
