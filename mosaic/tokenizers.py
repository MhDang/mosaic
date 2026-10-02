"""Tokenizer specifics: a tokenizer's family and end tokens, and turning a grammar into its token automaton."""

import os

from .grammar.tokenize import token_graph_from_char_nfa
from .sampling import TokenAutomaton


# ---- tokenizer-specific tokens ----


def tokenizer_key(tokenizer):
    """``dream``, ``llada``, ``llada-base`` or ``llada2``: tokenizers differ, so do automata."""
    name = tokenizer.name_or_path
    if "LLaDA2" in name:
        return "llada2"
    if "LLaDA" in name:
        # LLaDA-8B-Base and LLaDA-8B-Instruct have different tokenizers
        return "llada-base" if "Base" in name else "llada"
    if "dream" in name.lower():
        return "dream"
    raise NotImplementedError(f"No tokenizer key for {name}")


def resolve_tokenizer_tokens(tokenizer, end_with_eos=True):
    """Return (eos_token, eot_token, banned_tokens) for a Dream, LLaDA or LLaDA2 tokenizer.

    With ``end_with_eos`` the constraint ends with ``eot`` then any number of
    ``eos`` (the diffusion canvas is padded to its full length); the banned
    tokens may not appear inside generated strings.
    """
    name = tokenizer.name_or_path
    eos_token = tokenizer.eos_token if end_with_eos else None
    eot_token = None

    if "LLaDA2" in name:
        # the assistant turn ends with <|role_end|>, then <|endoftext|> pads the canvas
        banned = special_tokens(tokenizer)
        if end_with_eos:
            eot_token = "<|role_end|>"
    elif "LLaDA" in name:
        is_instruct = "Instruct" in name
        banned = [tokenizer.eos_token, "<|startoftext|>"]
        if is_instruct:
            banned.append("<|eot_id|>")
        if end_with_eos:
            eot_token = "<|eot_id|>" if is_instruct else tokenizer.eos_token
    elif "dream" in name.lower():
        banned = [tokenizer.eos_token, "<|im_start|>", "<|im_end|>"]
        if end_with_eos:
            eot_token = tokenizer.eos_token
    else:
        raise NotImplementedError(f"Token handling not defined for tokenizer {name}")
    return eos_token, eot_token, banned


def special_tokens(tokenizer):
    """Every special token of ``tokenizer`` (named ones and special added tokens), sorted."""
    special = set(tokenizer.all_special_tokens)
    special |= {t.content for t in getattr(tokenizer, "added_tokens_decoder", {}).values()
                if getattr(t, "special", False)}
    return sorted(special)


# ---- compiling a constraint ----


def compile_automaton(char_nfa, tokenizer, save_directory=None, cls=TokenAutomaton, device=None):
    """Re-alphabetise a character automaton to ``tokenizer``'s vocabulary.

    Returns it as a ``cls`` (a :class:`TokenAutomaton`, or a sampler class),
    saved to ``save_directory`` if given. ``device`` is where the automaton's
    tensors are built (default: the CPU).
    """
    graph = token_graph_from_char_nfa(char_nfa, tokenizer)
    automaton = cls.from_graph(graph, device=device)
    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)
        automaton.save_pretrained(save_directory)
    return automaton
