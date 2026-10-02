"""Re-alphabetising a constraint automaton from characters to tokens.

Grammars are written over characters, but generation happens over a
tokenizer's vocabulary, and a token spans several characters. Converting
between the two means walking each token's characters through the character
automaton and recording where it lands.
"""

from collections import deque

import numpy as np


def token_graph_from_char_nfa(nfa, tokenizer):
    """Convert ``nfa`` into a token-level automaton over ``tokenizer``'s vocabulary.

    Returns a graph dict with ``edges`` (``(from, to, mask)`` triples, where
    ``mask`` marks the token ids that take that edge), ``initial_state`` and
    ``accept_states``.
    """
    char_graph, symbol2id = _char_graph_from_char_nfa(nfa)
    return _token_graph_from_char_graph(char_graph, tokenizer, symbol2id=symbol2id)


def _char_graph_from_char_nfa(nfa):
    """Flatten ``nfa`` into a graph dict over integer symbol ids.

    Single-character symbols use their code point, so they coincide with the
    byte values a token decodes to. Longer symbols are indivisible: an
    end-of-sequence marker, say, gets its own id above the byte range.
    Returns the graph and the symbol-to-id mapping.
    """
    symbol2id = {}
    next_id = 256
    for symbol in nfa.input_symbols:
        if len(symbol) == 1:
            symbol2id[symbol] = ord(symbol)
        else:
            symbol2id[symbol] = next_id
            next_id += 1

    edges = []
    for source, target, symbols in nfa.all_edges():
        mask = np.zeros(next_id, dtype=bool)
        for symbol in symbols:
            mask[symbol2id[symbol]] = 1
        edges.append((source, target, mask))

    graph = {
        "edges": edges,
        "initial_state": nfa.initial_state,
        "accept_states": set(nfa.final_states),
    }
    return graph, symbol2id


def _token_graph_from_char_graph(char_graph, tokenizer, symbol2id=None):
    """Re-alphabetise ``char_graph`` from symbol ids to token ids.

    Explores states reachable from the initial one; from each, walks a byte
    trie of the vocabulary along the transitions the state allows, recording
    an edge for every token whose symbols can be consumed. A token whose text
    is one of the automaton's indivisible symbols (an end-of-sequence marker,
    say) is that symbol, not its bytes.
    """
    symbol2id = symbol2id or {}
    vocab_size = len(tokenizer)

    transitions = {}
    for source, target, mask in char_graph["edges"]:
        transitions[(source, target)] = mask
    symbol_index = _index_by_symbol(transitions)

    trie, texts = _vocab_trie(tokenizer)
    symbol_tokens = [(token_id, symbol2id[text]) for token_id, text in texts.items() if text in symbol2id]
    as_symbol = {token_id for token_id, _ in symbol_tokens}

    initial_state = char_graph["initial_state"]
    token_trans = {}
    reached = {initial_state}
    queue = deque([initial_state])

    while queue:
        state = queue.popleft()
        targets = {}  # target state -> token ids
        for token_id in trie.get(_TOKENS, ()):  # tokens with no text consume nothing
            if token_id not in as_symbol:
                targets.setdefault(state, []).append(token_id)
        stack = [(trie, (state,))]
        while stack:
            node, states = stack.pop()
            for byte, child in node.items():
                if byte == _TOKENS:
                    continue
                nxt = set()
                for s in states:
                    nxt.update(symbol_index.get((s, byte), ()))
                if not nxt:
                    continue
                for token_id in child.get(_TOKENS, ()):
                    if token_id not in as_symbol:
                        for target in nxt:
                            targets.setdefault(target, []).append(token_id)
                stack.append((child, tuple(nxt)))
        for token_id, symbol_id in symbol_tokens:
            for target in symbol_index.get((state, symbol_id), ()):
                targets.setdefault(target, []).append(token_id)
        for target, token_ids in targets.items():
            edge = token_trans.setdefault((state, target), np.zeros(vocab_size, dtype=bool))
            edge[token_ids] = 1
            if target not in reached:
                reached.add(target)
                queue.append(target)

    return {
        "edges": sorted((u, v, mask) for (u, v), mask in token_trans.items()),
        "initial_state": initial_state,
        "accept_states": [s for s in char_graph["accept_states"] if s in reached],
    }


#: Trie key under which a node lists the tokens whose bytes end there.
_TOKENS = -1
_TRIES = {}


def _vocab_trie(tokenizer):
    """A byte trie of the vocabulary (tokens by decoded UTF-8 bytes), and each token's text; cached per tokenizer."""
    key = (tokenizer.name_or_path, len(tokenizer), type(tokenizer).__name__)
    if key not in _TRIES:
        trie, texts = {}, {}
        for token_id in tokenizer.get_vocab().values():
            text = tokenizer.decode(token_id)
            texts[token_id] = text
            node = trie
            for byte in text.encode("utf-8"):
                node = node.setdefault(byte, {})
            node.setdefault(_TOKENS, []).append(token_id)
        _TRIES[key] = (trie, texts)
    return _TRIES[key]


def _index_by_symbol(transitions):
    """Index transitions as ``(state, symbol id) -> [target states]``."""
    symbol_index = {}
    for (source, target), mask in transitions.items():
        for symbol_id, present in enumerate(mask):
            if present:
                symbol_index.setdefault((source, symbol_id), []).append(target)
    return symbol_index


def _reachable_by(symbols, start_state, symbol_index):
    """States reachable from ``start_state`` by consuming ``symbols`` in order."""
    states = [start_state]
    for symbol_id in symbols:
        next_states = []
        for state in states:
            next_states.extend(symbol_index.get((state, symbol_id), ()))
        if not next_states:
            return []
        states = list(set(next_states))
    return states
