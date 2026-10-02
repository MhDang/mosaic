"""BFCL's official AST checker (Python only), vendored.

Ported from bfcl_eval.eval_checker.ast_eval.ast_checker in the gorilla
repository (Apache License 2.0) to avoid the bfcl_eval pip dependency. Java/JS
support and ``convert_func_name`` (only needed for OpenAI/Mistral function-name
mangling) are omitted; the Python checking logic is unchanged.

Reference: https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py
"""
import re

PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}

PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple"]


def find_description(func_descriptions, name):
    if isinstance(func_descriptions, list):
        for fd in func_descriptions:
            if fd["name"] == name:
                return fd
        return None
    return func_descriptions


def get_possible_answer_type(possible_answer):
    for a in possible_answer:
        if a != "":
            return type(a)
    return None


def standardize_string(s):
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", s).lower().replace("'", '"')


def type_checker(param, value, possible_answer, expected_type_description,
                 expected_type_converted, nested_type_converted):
    result = {"valid": True, "error": [], "is_variable": False, "error_type": "type_error:simple"}
    is_variable = False
    pat = get_possible_answer_type(possible_answer)
    if pat is not None and pat != expected_type_converted:
        is_variable = True
    if type(value) == expected_type_converted:
        if nested_type_converted is None:
            result["is_variable"] = is_variable
            return result
        for pa in possible_answer:
            flag = True
            if isinstance(pa, list):
                for vi in value:
                    cr = type_checker(param, vi, pa, str(nested_type_converted),
                                      nested_type_converted, None)
                    if not cr["valid"]:
                        flag = False
                        break
            if flag:
                return {"valid": True, "error": [], "is_variable": is_variable}
        result["valid"] = False
        result["error"] = [f"Nested type checking failed for {param!r}"]
        result["error_type"] = "type_error:nested"
        return result
    pat = get_possible_answer_type(possible_answer)
    if pat is not None and type(value) == pat:
        result["is_variable"] = True
        return result
    result["valid"] = False
    result["error"].append(f"Incorrect type for {param!r}: expected {expected_type_description}, got {type(value).__name__}")
    result["error_type"] = "type_error:simple"
    return result


def string_checker(param, model_output, possible_answer):
    if not isinstance(model_output, str):
        return {"valid": False, "error": [f"{param!r} not a string"], "error_type": "value_error:string"}
    smo = standardize_string(model_output)
    spa = [standardize_string(x) for x in possible_answer if isinstance(x, str)]
    if smo not in spa:
        return {"valid": False, "error": [f"string mismatch for {param!r}"], "error_type": "value_error:string"}
    return {"valid": True, "error": []}


def list_checker(param, model_output, possible_answer):
    smo = list(model_output)
    for i in range(len(smo)):
        if isinstance(smo[i], str):
            smo[i] = standardize_string(smo[i])
    spa = []
    for i in range(len(possible_answer)):
        spa.append([])
        for j in range(len(possible_answer[i])):
            x = possible_answer[i][j]
            spa[i].append(standardize_string(x) if isinstance(x, str) else x)
    if smo not in spa:
        return {"valid": False, "error": [f"list mismatch for {param!r}"], "error_type": "value_error:list/tuple"}
    return {"valid": True, "error": []}


def dict_checker(param, model_output, possible_answers):
    result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
    for i in range(len(possible_answers)):
        if possible_answers[i] == "":
            continue
        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        flag = True
        pa = possible_answers[i]
        for k, v in model_output.items():
            if k not in pa:
                result["valid"] = False
                result["error"].append(f"unexpected dict key {k!r}")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break
            sv = standardize_string(v) if isinstance(v, str) else v
            spa = [standardize_string(x) if isinstance(x, str) else x for x in pa[k]]
            if sv not in spa:
                result["valid"] = False
                result["error"].append(f"dict value mismatch for {k!r}")
                result["error_type"] = "value_error:dict_value"
                flag = False
                break
        if flag:
            for k, v in pa.items():
                if k not in model_output and "" not in v:
                    result["valid"] = False
                    result["error"].append(f"missing dict key {k!r}")
                    result["error_type"] = "value_error:dict_key"
                    flag = False
                    break
        if flag:
            return {"valid": True, "error": []}
    return result


def list_dict_checker(param, model_output, possible_answers):
    result = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}
    for ai in range(len(possible_answers)):
        flag = True
        if len(model_output) != len(possible_answers[ai]):
            result["valid"] = False
            result["error"] = ["wrong list-of-dict count"]
            result["error_type"] = "value_error:list_dict_count"
            flag = False
            continue
        for di in range(len(model_output)):
            r = dict_checker(param, model_output[di], [possible_answers[ai][di]])
            if not r["valid"]:
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}
    return result


def simple_function_checker(func_description, model_output, possible_answer):
    """model_output is {func_name: {arg: val}}, possible_answer is {func_name: {arg: [vals]}}."""
    pa = list(possible_answer.values())[0]
    func_name = func_description["name"]
    param_details = func_description["parameters"]["properties"]
    required = func_description["parameters"].get("required", [])

    if func_name not in model_output:
        return {"valid": False, "error": [f"function name {func_name!r} not in output"],
                "error_type": "simple_function_checker:wrong_func_name"}
    model_params = model_output[func_name]
    if not isinstance(model_params, dict):
        return {"valid": False, "error": ["arguments not a dict"],
                "error_type": "simple_function_checker:bad_args"}

    for p in required:
        if p not in model_params:
            return {"valid": False, "error": [f"missing required {p!r}"],
                    "error_type": "simple_function_checker:missing_required"}

    for p, v in model_params.items():
        if p not in param_details or p not in pa:
            return {"valid": False, "error": [f"unexpected parameter {p!r}"],
                    "error_type": "simple_function_checker:unexpected_param"}
        full = param_details[p]
        etd = full["type"]
        etc = PYTHON_TYPE_MAPPING.get(etd, str)
        ntc = None
        if etd in PYTHON_NESTED_TYPE_CHECK_LIST:
            nested = full.get("items", {}).get("type")
            if nested:
                ntc = PYTHON_TYPE_MAPPING.get(nested)
        if etd == "tuple" and isinstance(v, tuple):
            v = list(v)
        if etd == "float" and isinstance(v, int):
            v = float(v)
        tc = type_checker(p, v, pa[p], etd, etc, ntc)
        is_var = tc.get("is_variable", False)
        if not tc["valid"]:
            return tc
        if not is_var:
            if etc == dict:
                r = dict_checker(p, v, pa[p])
                if not r["valid"]:
                    return r
                continue
            if etc == list and ntc == dict:
                r = list_dict_checker(p, v, pa[p])
                if not r["valid"]:
                    return r
                continue
            if etc == str:
                r = string_checker(p, v, pa[p])
                if not r["valid"]:
                    return r
                continue
            if etc == list:
                r = list_checker(p, v, pa[p])
                if not r["valid"]:
                    return r
                continue
        if v not in pa[p]:
            return {"valid": False, "error": [f"value not in possible_answer for {p!r}"],
                    "error_type": "value_error:others"}

    for p in pa:
        if p not in model_params and "" not in pa[p]:
            return {"valid": False, "error": [f"missing optional {p!r}"],
                    "error_type": "simple_function_checker:missing_optional"}
    return {"valid": True, "error": []}


def parallel_function_checker_no_order(func_descriptions, model_output, possible_answers):
    if len(model_output) != len(possible_answers):
        return {"valid": False, "error": ["wrong number of functions"],
                "error_type": "parallel:wrong_count"}
    matched = []
    for i in range(len(possible_answers)):
        fne = list(possible_answers[i].keys())[0]
        fd = find_description(func_descriptions, fne)
        if fd is None:
            return {"valid": False, "error": [f"unknown function {fne!r}"],
                    "error_type": "parallel:unknown_func"}
        ok = False
        for j in range(len(model_output)):
            if j in matched:
                continue
            r = simple_function_checker(fd, model_output[j], possible_answers[i])
            if r["valid"]:
                matched.append(j)
                ok = True
                break
        if not ok:
            return {"valid": False, "error": ["no matching call"],
                    "error_type": "parallel:no_match"}
    return {"valid": True, "error": []}


def multiple_function_checker(func_descriptions, model_output, possible_answers):
    if len(model_output) != len(possible_answers):
        return {"valid": False, "error": ["wrong number of functions"],
                "error_type": "multiple:wrong_count"}
    fne = list(possible_answers[0].keys())[0]
    fd = find_description(func_descriptions, fne)
    if fd is None:
        return {"valid": False, "error": [f"unknown function {fne!r}"],
                "error_type": "multiple:unknown_func"}
    return simple_function_checker(fd, model_output[0], possible_answers[0])


def ast_checker(func_description, model_output, possible_answer, test_category):
    """test_category is the BFCL split name (or any string containing 'parallel'/'multiple')."""
    if "parallel" in test_category:
        return parallel_function_checker_no_order(func_description, model_output, possible_answer)
    if "multiple" in test_category:
        return multiple_function_checker(func_description, model_output, possible_answer)
    if len(model_output) != 1:
        return {"valid": False, "error": ["wrong number of functions"],
                "error_type": "simple:wrong_count"}
    return simple_function_checker(func_description[0], model_output[0], possible_answer[0])


def to_bfcl_format(model_calls):
    """Convert our internal `[{name, arguments}, ...]` to BFCL `[{name: arguments}, ...]`."""
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
