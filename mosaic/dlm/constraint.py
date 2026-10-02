"""What the Dream and LLaDA loops share about the constraint.

Their constrained step is :meth:`~mosaic.matcher.ConstraintMatcher.propose_x0` over
the whole response: an x0 proposal for every generated position drawn exactly
from LM x automaton, and optionally each proposed token's constrained marginal
(the ``commit_by=constrained`` commit order). Already-committed tokens are
kept; still-masked positions are free.
"""

#: Where a constrained loop's commit rule reads each proposed token's confidence: its exact
#: constrained marginal (``constrained``), or the model's probability of it (``model``).
COMMIT_BY_OPTIONS = ("constrained", "model")


def resolve_commit_by(value):
    """Check the setting (``constrained`` / ``model``); returns it."""
    if value not in COMMIT_BY_OPTIONS:
        raise ValueError(f"commit_by must be one of {COMMIT_BY_OPTIONS}, got {value!r}")
    return value


def check_matcher(matcher, gen_length):
    """Check that ``matcher`` suits a loop that proposes all ``gen_length`` positions at once.

    The response is one block, so the matcher must be fresh (nothing accepted)
    and bounded to it: then every proposal ends in an accepting state.
    """
    if matcher is not None:
        assert matcher.num_accepted == 0 and matcher.budget() == gen_length, (
            "the matcher must be fresh and bounded to the response: "
            "ConstraintMatcher(compiled, max_new_tokens=<response length>)"
        )
