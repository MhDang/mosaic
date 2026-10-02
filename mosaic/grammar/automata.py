"""Automata for writing grammars.

``ConstraintNFA`` is an NFA whose operations do not introduce lambda
transitions: union, concatenation, option, Kleene star and separator-joined
repetition. ``literal``, ``any_of`` and ``symbol`` build the small automata
that those operations combine.
"""

import operator
from functools import reduce
from itertools import chain, count

from automata.fa.dfa import DFA
from automata.fa.nfa import NFA


def literal(s):
    """NFA accepting the single string ``s``, one state per symbol."""
    return ConstraintNFA.from_literal(s)


def any_of(s):
    """NFA accepting any one of the strings ``s``.

    A union of one ``literal`` per string, minimized so the shared prefixes of
    the alternatives collapse into one path.
    """
    return ConstraintNFA.union_all([literal(x) for x in s]).dfa_minify()


def symbol(s):
    """NFA accepting ``s`` as one indivisible symbol."""
    return ConstraintNFA.from_symbol(s)


class ConstraintNFA(NFA):
    """An NFA whose operations do not introduce lambda transitions.

    Supports union, concatenation, option, Kleene star and separator-joined
    repetition. Unlike ``automata-lib``'s versions these splice states
    directly, which keeps the state and edge counts down - it matters for the
    JSON grammars, where they would otherwise grow far too large.
    """

    # ---- construction ----

    @classmethod
    def from_literal(cls, keyword):
        """Construct an NFA accepting ``keyword``, one state per symbol."""
        n = len(keyword)
        states = set(range(n + 1))
        transitions = {i: {keyword[i]: {i + 1}} for i in range(n)}
        transitions[n] = {}

        nfa = cls(
            states=states,
            input_symbols=set(keyword),
            transitions=transitions,
            initial_state=0,
            final_states={n},
        )
        return nfa

    @classmethod
    def from_symbol(cls, keyword):
        """Construct an NFA accepting ``keyword`` as one indivisible symbol.

        For multi-character symbols that must not be matched piece by piece,
        such as a tokenizer's end-of-sequence marker.
        """
        nfa = cls(
            states={0, 1},
            input_symbols={keyword},
            transitions={0: {keyword: {1}}, 1: {}},
            initial_state=0,
            final_states={1},
        )
        return nfa

    @classmethod
    def from_nfa(cls, nfa: NFA):
        """Convert a standard NFA to a ConstraintNFA."""
        return cls(
            states=nfa.states,
            input_symbols=nfa.input_symbols,
            transitions=nfa.transitions,
            initial_state=nfa.initial_state,
            final_states=nfa.final_states,
        )

    @classmethod
    def from_regex(cls, pattern, input_symbols=None):
        """Create an NFA from a regular expression."""
        while "-" in pattern:
            idx = pattern.index("-")
            s, e = idx - 1, idx + 1
            assert s >= 0 and e < len(pattern), "invalid range in regex"
            sub_str = "|".join(
                chr(c) for c in range(ord(pattern[s]), ord(pattern[e]) + 1)
            )
            pattern = pattern[:s] + "(" + sub_str + ")" + pattern[e + 1 :]
        nfa = NFA.from_regex(pattern, input_symbols=input_symbols)
        return cls.from_nfa(nfa)

    # ---- combining these automata; all of these stay lambda-free ----

    def union(self, other: NFA):
        """Accept the union of this language and ``other``'s."""

        # Remove any edge that goes to the initial state
        self = self._remove_edge_to_initial()
        other = other._remove_edge_to_initial()

        # Starting at 1 because 0 is for the initial state
        (state_map_a, state_map_b) = self.__class__._get_state_maps(
            self.states, other.states, start=1
        )

        # state_map does not contain initial states
        state_map_a = {k: v for k, v in state_map_a.items() if k != self.initial_state}
        state_map_b = {k: v for k, v in state_map_b.items() if k != other.initial_state}


        new_states = frozenset(chain(state_map_a.values(), state_map_b.values(), [0]))
        new_transitions = {state: {} for state in new_states}

        # Connect new initial state to both branches
        new_trans = {}
        for symbol, next_states in self.transitions[self.initial_state].items():
            if symbol not in new_trans:
                new_trans[symbol] = set()
            for v in next_states:
                assert v != self.initial_state
                new_trans[symbol].add(state_map_a[v])
        for symbol, next_states in other.transitions[other.initial_state].items():
            if symbol not in new_trans:
                new_trans[symbol] = set()
            for v in next_states:
                assert v != other.initial_state
                new_trans[symbol].add(state_map_b[v])

        new_transitions[0] = new_trans

        # Transitions of self, remove transition from initial state
        new_transitions_a = dict(
            [
                (k, self.transitions[k])
                for k in self.transitions
                if k != self.initial_state
            ]
        )
        for k, v in new_transitions_a.items():
            for symbol, next_states in v.items():
                assert self.initial_state not in next_states
        self.__class__._load_new_transition_dict(
            state_map_a, new_transitions_a, new_transitions
        )

        # Transitions of other
        new_transitions_b = dict(
            [
                (k, other.transitions[k])
                for k in other.transitions
                if k != other.initial_state
            ]
        )
        for k, v in new_transitions_b.items():
            for symbol, next_states in v.items():
                assert other.initial_state not in next_states

        self.__class__._load_new_transition_dict(
            state_map_b, new_transitions_b, new_transitions
        )

        # Final states
        new_final_states = frozenset(
            chain(
                (
                    state_map_a[state]
                    for state in self.final_states
                    if state != self.initial_state
                ),
                (
                    state_map_b[state]
                    for state in other.final_states
                    if state != other.initial_state
                ),
            )
        )
        if (
            self.initial_state in self.final_states
            or other.initial_state in other.final_states
        ):
            new_final_states = new_final_states | {0}

        return self.__class__(
            states=new_states,
            input_symbols=self.input_symbols | other.input_symbols,
            transitions=new_transitions,
            initial_state=0,
            final_states=new_final_states,
        )

    @classmethod
    def union_all(cls, nfas):
        """Union of every NFA in ``nfas``."""
        return reduce(operator.or_, nfas)

    def concatenate(self, other: NFA):
        """Accept this language followed by ``other``'s."""
        other = other._remove_edge_to_initial()
        (state_map_a, state_map_b) = self.__class__._get_state_maps(
            self.states, other.states
        )

        state_map_b = {k: v for k, v in state_map_b.items() if k != other.initial_state}
        new_states = frozenset(chain(state_map_a.values(), state_map_b.values()))

        new_transitions = {state: {} for state in new_states}

        # Transitions of self
        self.__class__._load_new_transition_dict(
            state_map_a, self.transitions, new_transitions
        )
        # Transitions of other, remove initial state out edges
        other_transitions = dict(
            [
                (k, other.transitions[k])
                for k in other.transitions
                if k != other.initial_state
            ]
        )

        for k, v in other_transitions.items():
            for symbol, next_states in v.items():
                assert other.initial_state not in next_states

        self.__class__._load_new_transition_dict(
            state_map_b, other_transitions, new_transitions
        )

        # Transitions from self to other second states
        new_transitions_self = self._from_self_to_other_seconds(
            other, state_map_a, state_map_b
        )
        new_transitions = self._update_transition_left(
            new_transitions, new_transitions_self
        )

        # Final states of other
        new_final_states = frozenset(
            state_map_b[state]
            for state in other.final_states
            if state != other.initial_state
        )
        if other.initial_state in other.final_states:
            new_final_states = new_final_states | {
                state_map_a[state] for state in self.final_states
            }

        return self.__class__(
            states=new_states,
            input_symbols=self.input_symbols | other.input_symbols,
            transitions=new_transitions,
            initial_state=state_map_a[self.initial_state],
            final_states=new_final_states,
        )

    def recursive_concat(
        self,
        connect_str,
        ws=(" ", "\t", "\n"),
        ws_kleene="plus",
    ) -> "ConstraintNFA":
        """Accept ``X (sep X)*``: this language repeated, joined by a separator.

        ``connect_str`` is the separator, either a string or a list of
        alternative strings. ``ws`` lists the whitespace symbols allowed around
        it, and ``ws_kleene`` how many are required: ``"plus"`` for one or
        more, ``"star"`` for zero or more, ``"none"`` for no whitespace at all.
        """

        # build connector nfa
        ws_nfa = ConstraintNFA.union_all(
            ConstraintNFA.from_literal(c) for c in ws
        ).dfa_minify()
        if ws_kleene == "plus":
            ws_nfa = ws_nfa + ws_nfa.kleene_star().dfa_minify()
        elif ws_kleene == "star":
            ws_nfa = ws_nfa.kleene_star().dfa_minify()
        elif ws_kleene == "none":
            ws_nfa = None
        else:
            raise ValueError(
                f"Unknown ws_kleene {ws_kleene}, "
                "should be 'plus', 'star' or 'none'"
            )

        if isinstance(connect_str, list):
            str_nfa = ConstraintNFA.union_all(
                ConstraintNFA.from_literal(c) for c in connect_str
            ).dfa_minify()
        elif isinstance(connect_str, str):
            str_nfa = ConstraintNFA.from_literal(connect_str).dfa_minify()
        else:
            raise ValueError(
                f"Unknown connect_str type {type(connect_str)}, "
                "should be str or list of str"
            )

        if ws_nfa is None:
            connector_nfa_ = str_nfa
        else:
            connector_nfa_ = ws_nfa + str_nfa + ws_nfa

        connector_nfa_ = connector_nfa_.dfa_minify()

        connector_nfa = connector_nfa_._remove_edge_to_initial()
        connector_nfa = connector_nfa._remove_self_loop_edge_from_final()
        assert connector_nfa.dfa_minify() == connector_nfa_, "should be minimized"

        # kleene_plus operation with connector_nfa
        (state_map_a, state_map_b) = self.__class__._get_state_maps(
            self.states, connector_nfa.states
        )

        state_map_b = {
            k: v
            for k, v in state_map_b.items()
            if k != connector_nfa.initial_state and k not in connector_nfa.final_states
        }
        transitions_b = {
            k: {kk: vv - connector_nfa.final_states for (kk, vv) in v.items()}
            for k, v in connector_nfa.transitions.items()
            if k != connector_nfa.initial_state and k not in connector_nfa.final_states
        }
        new_states = frozenset(chain(state_map_a.values(), state_map_b.values()))
        new_initial_state = state_map_a[self.initial_state]

        new_transitions = {state: {} for state in new_states}

        self.__class__._load_new_transition_dict(
            state_map_a, self.transitions, new_transitions
        )

        self.__class__._load_new_transition_dict(
            state_map_b, transitions_b, new_transitions
        )
        for state in connector_nfa.final_states:
            state_map_b[state] = state_map_a[self.initial_state]

        # add a few transitions
        # from self final to connector_nfa second states
        new_transitions_self = self._from_self_to_other_seconds(
            connector_nfa, state_map_a, state_map_b
        )
        new_transitions = self._update_transition_left(
            new_transitions, new_transitions_self
        )

        # from connector last to final to self initial states
        new_transitions_connector = connector_nfa._from_last_to_final_to_other_initial(
            self, state_map_b, state_map_a
        )
        new_transitions = self._update_transition_left(
            new_transitions, new_transitions_connector
        )

        return self.__class__(
            states=new_states,
            input_symbols=self.input_symbols | connector_nfa.input_symbols,
            transitions=new_transitions,
            initial_state=new_initial_state,
            final_states=set(state_map_a[state] for state in self.final_states),
        )

    def option(self):
        """Also accept the empty string, i.e. the regular expression ``?``."""
        if self.initial_state in self.final_states:
            return self
        self = self._remove_edge_to_initial()
        return self.__class__(
            states=self.states,
            input_symbols=self.input_symbols,
            transitions=self.transitions,
            initial_state=self.initial_state,
            final_states=self.final_states | {self.initial_state},
        )

    def kleene_star(self):
        """Accept zero or more repetitions of this language, i.e. ``*``."""
        self = self._remove_edge_to_initial()
        state_map = {k: k for k in self.states}
        new_transitions = {
            state: dict(transition) for state, transition in self.transitions.items()
        }
        new_transitions_self = self._from_self_to_other_seconds(
            self, state_map, state_map
        )
        new_transitions = self._update_transition_left(
            new_transitions, new_transitions_self
        )
        return self.__class__(
            states=self.states,
            input_symbols=self.input_symbols,
            transitions=new_transitions,
            initial_state=self.initial_state,
            final_states=self.final_states | {self.initial_state},
        )

    # ---- inspection ----

    def has_lambda(self):
        """Check if the NFA has any lambda (ε) transitions."""
        for state in self.states:
            if "" in self.transitions.get(state, {}):
                return True
        return False

    def all_edges(self):
        """Return a list of all edges in the NFA."""
        edges = dict()
        for transition in self.iter_transitions():
            u, v, e = transition
            if (u, v) not in edges:
                edges[(u, v)] = []
            edges[(u, v)].append(e)

        edges = [(u, v, e) for ((u, v), e) in edges.items()]
        return edges

    def num_edges(self):
        """Return the number of edges in the NFA."""
        return len(self.all_edges())

    # ---- normalization ----

    def dfa_minify(self):
        """Minifies the NFA by converting it to a DFA and minimizing the DFA."""
        return self.__class__.from_dfa(DFA.from_nfa(self).minify())

    def canonicalize(self):
        """Relabel the reachable states 0, 1, ... in breadth-first order.

        The search starts at the initial state and follows symbols in sorted
        order. ``dfa_minify`` names states by the order its partition
        refinement happens to finish, which varies from run to run; after this
        relabeling a deterministic automaton's numbering depends only on its
        language, so the same constraint always compiles to the same tensors.
        """
        names = {self.initial_state: 0}
        order = [self.initial_state]
        for state in order:
            for sym in sorted(self.transitions.get(state, {})):
                for target in sorted(self.transitions[state][sym], key=str):
                    if target not in names:
                        names[target] = len(order)
                        order.append(target)

        transitions = {
            names[state]: {
                sym: {names[target] for target in targets}
                for sym, targets in self.transitions.get(state, {}).items()
            }
            for state in order
        }
        return self.__class__(
            states=set(range(len(order))),
            input_symbols=self.input_symbols,
            transitions=transitions,
            initial_state=0,
            final_states={names[s] for s in self.final_states if s in names},
        )

    # ---- internal: state splicing used by the operations above ----

    def _remove_edge_to_initial(self):
        """Remove any edge that goes to the initial state."""
        do_remove = False
        for u, v, e in self.iter_transitions():
            if v == self.initial_state:
                do_remove = True
                break

        if not do_remove:
            return self

        state_map = dict(zip(self.states, count(1)))

        new_transitions = {state: {} for state in self.states}
        for k, v in self.transitions.items():
            new_transition = dict()
            for symbol, next_states in v.items():
                new_transition[symbol] = set()
                new_transition[symbol] |= {
                    state_map[next_state]
                    for next_state in next_states
                    if next_state != self.initial_state
                }
                new_transition[symbol] |= {
                    0 for next_state in next_states if next_state == self.initial_state
                }
            new_transitions[state_map[k]] = new_transition

        new_transition_extra = dict()
        for symbol, next_states in self.transitions[self.initial_state].items():
            new_transition_extra[symbol] = set()
            new_transition_extra[symbol] |= {
                state_map[next_state]
                for next_state in next_states
                if next_state != self.initial_state
            }
            new_transition_extra[symbol] |= {
                0 for next_state in next_states if next_state == self.initial_state
            }
        new_transitions[0] = new_transition_extra

        final_states = {state_map[state] for state in self.final_states}

        if self.initial_state in self.final_states:
            final_states.add(0)

        return self.__class__(
            states=set(state_map.values()) | {0},
            input_symbols=self.input_symbols,
            transitions=new_transitions,
            initial_state=state_map[self.initial_state],
            final_states=final_states,
        )

    def _remove_self_loop_edge_from_final(self):
        """Remove self loop on final states"""
        do_remove = False
        for u, v, e in self.iter_transitions():
            if u in self.final_states:
                do_remove = True
                assert u == v, "self-loop on final state"
                break
        assert len(self.final_states) == 1, "assume only one final state"
        if not do_remove:
            return self

        state_map = dict(zip(self.states, count(1)))
        new_states = state_map.values()

        new_transitions = {state: {} for state in new_states}
        self.__class__._load_new_transition_dict(
            state_map, self.transitions, new_transitions
        )

        for u, v, e in self.iter_transitions():
            if v in self.final_states:
                new_transitions[state_map[u]][e] |= {0}

        final_states = {0}
        return self.__class__(
            states=set(new_states) | {0},
            input_symbols=self.input_symbols,
            transitions=new_transitions,
            initial_state=state_map[self.initial_state],
            final_states=final_states,
        )

    def _from_self_to_other_seconds(self, other, state_map_a, state_map_b):
        """Copy ``other``'s initial-state transitions onto ``self``'s accepting states.

        This is the lambda-free splice: instead of joining the two automata
        with an epsilon edge, every accepting state of ``self`` gains the edges
        that would have left ``other``'s initial state.
        """
        new_transitions = {state: {} for state in state_map_a.values()}
        for state in self.final_states:
            new_transition = new_transitions[state_map_a[state]]
            for symbol, next_states in other.transitions[other.initial_state].items():
                if symbol not in new_transition:
                    new_transition[symbol] = set()
                new_transition[symbol] |= {
                    state_map_b[next_state] for next_state in next_states
                }
            new_transitions[state_map_a[state]] = new_transition

        return new_transitions

    def _from_last_to_final_to_other_initial(
        self, other, state_map_self, state_map_other
    ):
        """Redirect edges landing on an accepting state to ``other``'s initial state.

        The mirror of :meth:`_from_self_to_other_seconds`: that method enters a
        successor automaton from accepting states, this one leaves the current
        automaton by rewiring its final edges into the successor's initial
        state. Used by :meth:`recursive_concat` to close the loop from the end
        of a separator back into the next repetition.
        """
        new_transitions = {state: {} for state in state_map_self.values()}
        for u, v, e in self.iter_transitions():
            if v in self.final_states:
                if u in state_map_self:
                    if e not in new_transitions[state_map_self[u]]:
                        new_transitions[state_map_self[u]][e] = set()
                    new_transitions[state_map_self[u]][e] |= {
                        state_map_other[other.initial_state]
                    }

        return new_transitions

    def _update_transition_left(self, transitions1, transitions2):
        """Merge ``transitions2`` into ``transitions1`` in place, unioning targets."""
        for k, v in transitions2.items():
            if k not in transitions1:
                transitions1[k] = v
            else:
                for symbol, next_states in v.items():
                    if symbol not in transitions1[k]:
                        transitions1[k][symbol] = next_states
                    else:
                        transitions1[k][symbol] |= next_states
        return transitions1
