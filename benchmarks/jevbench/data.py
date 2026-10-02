"""djev (thinking) on JevBench with DiffusionGemma: the model, the request JevBench sends, and its parse.

For run_vllm.py, next to this file.
"""

import os
from unittest import mock

import torch

from mosaic.matcher import ConstraintCompiler

MODEL = "google/diffusiongemma-26B-A4B-it"
#: With thinking on, the thought channel opens with these and ends with THOUGHT_CLOSE.
THOUGHT_OPEN, THOUGHT_CLOSE = [100, 45518, 107], 101  # <|channel> thought \n ... <channel|>
EOT, PAD = "<turn|>", "<pad>"
TIERS = {"easy.jsonl": "easy", "original.jsonl": "standard", "hard.jsonl": "hard"}


# ---- the request JevBench sends djev (thinking), and its parse ----


#: max_tokens 8192, served by djev-dev's vLLM (runtime/serve.py: canvas_length 128). Its DiffusionGemma sampler
#: runs the checkpoint's generation_config, as HF generate does; the request's temperature and response_format
#: never reach it, so the output is free text.
DJEV_REQUEST = {"max_tokens": 8192, "chat_template_kwargs": {"enable_thinking": True}}

#: The system message of that request (JevBench's openai_compat adapter), for requests written by hand
#: like the README's; :func:`djev_prompt` takes the whole request from the adapter itself.
SYSTEM_PROMPT = (
    "You are a calibration engine. You never answer in prose. You output only "
    "a JSON object with the key 'probabilities' mapping every given option to "
    "a probability, all options included, values in [0,1], summing to 1."
)


def djev_prompt(tokenizer, task):
    """The adapter's request body and its prompt ids, rendered as djev-dev's vLLM renders them."""
    from jevbench.adapters import openai_compat

    adapter = openai_compat.OpenAICompatAdapter("http://local/v1", "dgemma")
    adapter.request_options = DJEV_REQUEST
    body = adapter.build_request(task)
    # vLLM hands message text to the chat template as content parts (the system turn gains a space)
    messages = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in body["messages"]]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False,
                                        **body["chat_template_kwargs"])
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return body, list(ids)


def adapter_parse(task, text, usage, latency):
    """JevBench's adapter run with ``text`` as the response content: its own parse, no request sent."""
    from jevbench.adapters import openai_compat

    adapter = openai_compat.OpenAICompatAdapter("http://local/v1", "dgemma")

    def post(url, payload, headers, timeout_s):
        return 200, {"choices": [{"message": {"content": text}}], "usage": usage}, latency

    with mock.patch.object(openai_compat, "http_post_json", post), mock.patch.dict(os.environ, {adapter.key_env: "x"}):
        return adapter.run(task)


# ---- the constrained answer ----


def written_schema(task):
    """{"probabilities": {<label>: <probability>, ...}}, every label in order."""
    labels = [str(x) for x in task.labels]
    probs = {"type": "object", "properties": {lab: {"type": "number", "x-prob": True} for lab in labels},
             "required": labels}
    return {"type": "object", "properties": {"probabilities": probs}, "required": ["probabilities"]}


def make_compiler(tokenizer, cache_dir, device, compile_workers=0):
    # fp64 parallel-sampler products: in fp32 the constrained marginals underflow to NaN on a few percent of
    # canvases (a garbage reason), and on most canvases of the -prob automaton; same speed
    return ConstraintCompiler(tokenizer, eos_token=PAD, eot_token=EOT, device=device, cache_dir=cache_dir,
                              cache_size=1, matmul_dtype=torch.float64, compile_workers=compile_workers)
