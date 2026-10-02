"""Parallel sampler: exact LM x automaton sampling in O(log L) depth, with a segment tree.

At one denoising step the diffusion LM gives an independent distribution per
position. Conditioning their product on the automaton accepting the sequence
is an HMM posterior. Instead of a sequential forward-backward, the per-position
transition matrices are multiplied up a balanced binary tree (log L levels);
sampling then walks down the tree, fixing the automaton state at each segment
midpoint, and finally draws each token given the edge it sits on. Per-position
marginals come from one pass up and one pass down a segment tree.

Any ``seq_len`` works: the tree is padded to a power of two with identity
leaves, which leave every product unchanged. Everything is in log space.

A call launches a hundred-odd small kernels, so on a GPU it is bound by launch
latency: :func:`sample_many` samples rows with different automata in one call.
"""

import logging

import torch
from einops import repeat

from .automaton import TokenAutomaton
from .layers import (
    matmul_a_logb,
    matmul_loga_logb,
)


class ParallelSampler(TokenAutomaton):
    """Samples from LM x automaton with a segment tree over positions.

    ``matmul_dtype`` is the precision of the tree's matrix products: None keeps
    the automaton's own dtype (fp32, the default), ``torch.float64`` computes
    them in fp64, which is much more robust to underflow on long or very
    peaked inputs but slower on GPUs with weak fp64.
    """

    matmul_dtype = None

    @torch.no_grad()
    def sample(self, logits, forced_tokens=None, temperature=1.0,
               return_marginal=False, start_nodes=None, end_log=None):
        """Draw from the LM x automaton product. Returns (tokens, marginal).

        logits: (bz, L, V) LM logits for the generated span.
        forced_tokens: (L,) or (bz, L); -1 marks a free position, anything
            else is an already-committed token.
        temperature: for the token drawn on each edge (0 = argmax); the state
            path is always sampled.
        return_marginal: also return, per position, the probability of the
            drawn token under the exact LM x automaton marginal (bz, L);
            otherwise the second value is None.
        start_nodes: (bz,) node each row starts in (default: the initial
            node), e.g. the state after an already-generated prefix.
        end_log: (N,) or (bz, N) log weight of ending in each node (default:
            0 on accepting nodes), e.g. ``finish_within_log(remaining)`` when
            more tokens follow the span.
        """
        bz, seq_len, _ = logits.shape
        forced_tokens = self._broadcast_forced(forced_tokens, bz)
        start_nodes, end_log = self._resolve_ends(bz, start_nodes, end_log)
        lm_logits = torch.log_softmax(logits, dim=-1)
        start_w, Q = self._backward_log(lm_logits, forced_tokens, start_nodes, end_log)
        nan_counts = []
        # drop the boundaries of the identity padding, if any
        boundary = self._forward_log(start_w, Q, nan_counts)[:, : seq_len + 1]
        tokens = self._sample_tokens_log(
            boundary, lm_logits, forced_tokens, start_nodes, temperature, nan_counts,
        )
        # The sampling tree is no longer needed; free it before the marginal
        # pass builds its own.
        del start_w, Q, boundary

        marginal = None
        if return_marginal:
            marginal = self._compute_marginal_log(
                lm_logits, forced_tokens, sampled_tokens=tokens,
                start_nodes=start_nodes, end_log=end_log,
            ).exp()
        warn_nan_counts(nan_counts)
        return tokens, marginal

    # ---- backward: build the tree bottom-up ----

    def _backward_log(self, lm_logits, forced_tokens, start_nodes, end_log):
        """Build the segment tree bottom-up (sum semiring).

        Returns start_w (bz, N), the first position's weights out of each row's
        start node, and Q, a list of (bz, n, N, N) levels: Q[0] holds the
        leaves (positions 1..L-1, identity padding up to a power of two, then
        the end weights on the diagonal), Q[-1] the root. Every node is shifted
        by its max, which sampling does not see.
        """
        e_logits = self._compute_emission_log(lm_logits, forced_tokens)  # (bz, L, E)
        flow = _edge_flow(e_logits, self.edge_lookup)  # (bz, L, N, N)
        return _build_tree(flow, start_nodes, end_log, self.matmul_dtype)

    # ---- forward: fix the state path top-down ----

    def _forward_log(self, start_w, Q, nan_counts):
        """Sample the node path top-down. Returns LongTensor (bz, seq_len + 1).

        boundary[:, t] is the node reached after token t. The root fixes the
        first and last nodes; each level then fixes the midpoint of every
        segment given its two ends.
        """
        return _forward_sample(start_w, Q, nan_counts)

    # ---- tokens given the path ----

    def _sample_tokens_log(self, boundary, lm_logits, forced_tokens, start_nodes, temperature, nan_counts):
        """Draw each token given the edge its position sits on. Returns (bz, seq_len)."""
        seq_len = boundary.shape[1] - 1
        src = torch.cat([start_nodes[:, None], boundary[:, : seq_len - 1]], dim=1)
        dst = boundary[:, :seq_len]
        edge_ids = self.edge_lookup[src, dst]
        log_emit = self.emission[edge_ids.clamp(min=0)].log()
        return _pick_tokens(log_emit, edge_ids, lm_logits, forced_tokens, temperature, nan_counts)

    # ---- exact per-position marginals ----

    def _compute_marginal_log(self, lm_logits, forced_tokens=None, sampled_tokens=None,
                              start_nodes=None, end_log=None):
        """Log LM x automaton marginal p(x_t = v | constraint) at every position.

        The per-position transition matrices are multiplied up a segment tree
        (without the sampling tree's shifts), then alpha / beta at every
        boundary are filled in one pass down it: at a node spanning
        [start, end) with midpoint mid,

            alpha[mid] = alpha[start] @ left_child
            beta[mid]  = right_child @ beta[end]

        so the depth is O(log L). Then

            p(x_t = v) ∝ p_LM(x_t = v) * sum_e alpha_t[src e] emission[e, v] beta_{t+1}[dst e].

        sampled_tokens=None -> (bz, L, V) log-marginals;
        sampled_tokens (bz, L) -> (bz, L) log-prob of those tokens.
        ``lm_logits`` must already be log-softmaxed. start_nodes / end_log as
        in :meth:`sample`.
        """
        bz, L, V = lm_logits.shape
        N = self.num_nodes
        device = self.device
        dtype = self.dtype
        start_nodes, end_log = self._resolve_ends(bz, start_nodes, end_log)
        bz_dim = torch.arange(bz, device=device)

        e_logits = self._compute_emission_log(lm_logits, forced_tokens)
        flow = _edge_flow(e_logits, self.edge_lookup)  # (bz, L, N, N)
        pad = _pad_to_power_of_two(L)
        if pad:  # identity leaves after the last position
            eye = torch.eye(N, device=device, dtype=dtype).log()
            flow = torch.cat([flow, eye.expand(bz, pad, N, N)], dim=1)
        P = L + pad

        # ---- up: levels[k][:, j] is the product over positions [j*2^k, (j+1)*2^k)
        levels = [flow]
        cur = flow
        while cur.shape[1] > 1:
            cur = matmul_loga_logb(cur[:, 0::2], cur[:, 1::2], self.matmul_dtype)
            levels.append(cur)
        depth = len(levels) - 1

        neg_inf = float("-inf")
        log_alpha = torch.full((bz, P + 1, N), neg_inf, device=device, dtype=dtype)
        log_beta = torch.full((bz, P + 1, N), neg_inf, device=device, dtype=dtype)
        log_alpha[bz_dim, 0, start_nodes] = log_alpha.new_zeros(())  # a device scalar: no host copy
        log_beta[:, P] = end_log

        # ---- down: fill alpha / beta at each node's midpoint
        for k in range(depth, 0, -1):
            chunk_size = 1 << k
            n_cells = P // chunk_size
            starts = torch.arange(n_cells, device=device) * chunk_size
            splits = starts + chunk_size // 2
            ends = starts + chunk_size

            left_chunks = levels[k - 1][:, 0::2]  # (bz, n_cells, N, N)
            right_chunks = levels[k - 1][:, 1::2]
            alpha_at_splits = matmul_loga_logb(
                log_alpha[:, starts, :].unsqueeze(-2), left_chunks, self.matmul_dtype
            ).squeeze(-2)  # (bz, n_cells, N)
            beta_at_splits = matmul_loga_logb(
                right_chunks, log_beta[:, ends, :].unsqueeze(-1), self.matmul_dtype
            ).squeeze(-1)  # (bz, n_cells, N)
            log_alpha[:, splits, :] = alpha_at_splits
            log_beta[:, splits, :] = beta_at_splits

        root = levels[depth][:, 0]  # (bz, N, N)
        log_Z = torch.logsumexp(root[bz_dim, start_nodes, :] + end_log, dim=-1)  # (bz,)

        # ---- per edge: alpha at its source + beta at its destination
        alpha_src = log_alpha[:, :L, :].gather(
            -1, self.edge_src[None, None, :].expand(bz, L, -1)
        )  # (bz, L, E)
        beta_dst = log_beta[:, 1 : L + 1, :].gather(
            -1, self.edge_index[None, None, :].expand(bz, L, -1)
        )  # (bz, L, E)
        ab = alpha_src + beta_dst

        if sampled_tokens is not None:
            emit_sampled = self.emission[:, sampled_tokens].permute(1, 2, 0)  # (bz, L, E)
            log_emit = emit_sampled.to(dtype).clamp_min(0).log()  # {0, -inf}
            log_sum = torch.logsumexp(ab + log_emit, dim=-1)  # (bz, L)
            lm_pick = lm_logits.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
            return lm_pick + log_sum - log_Z[:, None]

        flat_ab = ab.reshape(-1, ab.shape[-1])  # (bz*L, E)
        emit_T = self.emission.to(dtype).T.contiguous()  # (V, E)
        log_emit_sum = matmul_a_logb(emit_T, flat_ab.T)  # (V, bz*L)
        log_emit_sum = log_emit_sum.T.reshape(bz, L, V)
        return lm_logits + log_emit_sum - log_Z[:, None, None]

# ---- shared by ParallelSampler and sample_many: the tree and the token draw ----


def _edge_flow(e_logits, edge_lookup):
    """Per-position transition weights (bz, L, N, N) from edge weights (bz, L, E):
    flow[b, t, i, j] is the weight of the edge from i to j, -inf if there is none.

    An automaton has at most one edge between two nodes, so this gather equals the
    logsumexp over the edges into each cell (bit for bit). ``edge_lookup`` is (N, N),
    or (bz, N, N) for a different automaton per row.
    """
    bz, L, E = e_logits.shape
    N = edge_lookup.shape[-1]
    padded = torch.cat([e_logits, e_logits.new_full((bz, L, 1), float("-inf"))], dim=-1)
    index = edge_lookup.masked_fill(edge_lookup < 0, E)  # no edge: the -inf column
    if index.dim() == 2:
        return padded[:, :, index.view(-1)].view(bz, L, N, N)
    return padded.gather(2, index.view(bz, 1, N * N).expand(bz, L, N * N)).view(bz, L, N, N)


def _forward_sample(start_w, Q, nan_counts):
    """:meth:`ParallelSampler._forward_log` for any tree ``Q`` (its automaton's padding
    included). Each NaN fallback appends its count to ``nan_counts`` (see
    :func:`warn_nan_counts`)."""
    bz, seq_len, N, _ = Q[0].shape
    device = Q[0].device
    bz_dim = torch.arange(bz, device=device)

    def _decode(logits):
        probs = torch.softmax(logits.float(), dim=-1)
        bad = probs.isnan().any(dim=-1)
        # uniform over the nodes where the weights are NaN; torch.where, not an
        # `if bad.any()`, so that no level waits on the GPU
        probs = torch.where(bad[:, None], 1.0 / N, probs)
        nan_counts.append(bad.sum())
        return torch.multinomial(probs, 1).squeeze(-1)

    root = Q[-1]
    z_1 = _decode(root[:, 0].logsumexp(dim=-1) + start_w)
    z_end = _decode(root[bz_dim, 0, z_1])

    boundary = torch.zeros(bz, seq_len + 1, dtype=torch.long, device=device)
    boundary[:, 0] = z_1
    boundary[:, seq_len] = z_end

    level = len(Q) - 2
    while level >= 0:
        Q_level = Q[level]
        left = Q_level[:, 0::2]
        right = Q_level[:, 1::2]
        num_seg = Q_level.shape[1] // 2
        seg_len = seq_len // num_seg
        begin = torch.arange(num_seg, device=device) * seg_len
        end = begin + seg_len
        mid = begin + seg_len // 2
        seg_idx = torch.arange(num_seg, device=device)[None, :]

        src_nodes = boundary[:, begin]
        dst_nodes = boundary[:, end]
        logits_left = left[bz_dim[:, None], seg_idx, src_nodes]
        logits_right = right[bz_dim[:, None], seg_idx, :, dst_nodes]

        z_mid = _decode((logits_left + logits_right).reshape(-1, N))
        boundary[:, mid] = z_mid.reshape(bz, num_seg)
        level -= 1

    return boundary


def _build_tree(flow, start_nodes, end_log, matmul_dtype=None):
    """The segment tree above per-position transition weights ``flow`` (bz, L, N, N).

    Returns (start_w, Q) as :meth:`ParallelSampler._backward_log` describes.
    """
    bz, seq_len, N, _ = flow.shape
    device, dtype = flow.device, flow.dtype
    flow_max = flow.flatten(2).amax(2)[:, :, None, None]
    flow_max.masked_fill_(flow_max.isinf(), 0.0)
    flow.sub_(flow_max)

    start_w = flow[torch.arange(bz, device=device), 0, start_nodes]  # (bz, N)
    flow = flow[:, 1:]  # (bz, L-1, N, N)

    eye = repeat(
        torch.eye(N, device=device, dtype=dtype), "N1 N2 -> bz 1 N1 N2", bz=bz
    ).log()
    end_leaf = eye + end_log[:, None, None, :]  # fold the end weights into the last leaf
    pad = _pad_to_power_of_two(seq_len)
    leaves = [flow, eye.expand(bz, pad, N, N), end_leaf] if pad else [flow, end_leaf]
    flow = torch.cat(leaves, dim=1)

    Q = [flow]
    while flow.shape[1] > 1:
        flow = matmul_loga_logb(flow[:, 0::2], flow[:, 1::2], matmul_dtype)
        flow_max = flow.flatten(2).amax(2)[:, :, None, None]
        flow_max.masked_fill_(flow_max.isinf(), 0.0)
        flow.sub_(flow_max)
        Q.append(flow)

    return start_w, Q


def _pick_tokens(log_emit, edge_ids, lm_logits, forced_tokens, temperature, nan_counts):
    """Each position's token given its edge: ``log_emit`` (bz, L, V) is the edge's
    0/-inf token mask (garbage where ``edge_ids`` < 0, masked here). The NaN
    fallback appends its count to ``nan_counts``."""
    bz, seq_len, V = lm_logits.shape
    log_emit = log_emit.masked_fill(edge_ids.unsqueeze(-1) < 0, float("-inf"))

    token_logits = (log_emit + lm_logits).reshape(-1, V)
    if temperature == 0:
        flat_tokens = token_logits.argmax(dim=-1)
    else:
        flat_probs = torch.softmax(token_logits.float() / temperature, dim=-1)
        bad = flat_probs.isnan().any(dim=-1)
        flat_probs = torch.where(bad[:, None], 1.0 / V, flat_probs)  # no host sync
        flat_tokens = torch.multinomial(flat_probs, 1).squeeze(-1)
        nan_counts.append(bad.sum())

    tokens = flat_tokens.reshape(bz, seq_len)
    if forced_tokens is not None:  # keep the committed tokens (torch.where: no host sync)
        tokens = torch.where(forced_tokens != -1, forced_tokens, tokens)
    return tokens

# ---- rows with different automata, in one pass ----


@torch.no_grad()
def sample_many(samplers, logits, forced_tokens=None, temperature=1.0, start_nodes=None,
                end_log=None, nan_counts=None):
    """:meth:`ParallelSampler.sample` (without marginals) for rows with different automata.

    The tree, the state path and the token draw run once for the whole batch
    (a call is a hundred-odd small kernels, so one call instead of one per row);
    the emission products stay per row. Every automaton is padded to the
    largest state and edge counts: padded states have no edges in or out and
    padded edges allow no token, so they carry no weight and are never drawn.
    Each row is distributed exactly as its own ``sample`` draws it, though not
    bit for bit (padded products may sum in another order; the random draws
    are shared by the rows).

    samplers: one ParallelSampler per row (same vocabulary, device, dtype and matmul_dtype).
    logits (bz, L, V); forced_tokens (bz, L) or None; start_nodes (bz,) or None
    (each row's initial node); end_log: a list of per-row (N_b,) log end
    weights, or None (each row's accepting nodes). Returns tokens (bz, L).

    Rows are packed into passes by automaton size, so that no pass pads beyond
    ``_SHARED_MAX_LEAF_ELEMENTS``: a large automaton goes alone rather than
    padding every other row to its size.

    nan_counts: None, to check the NaN fallbacks here (a host sync), or a list the
    call appends its counts to (device tensors) for the caller to check
    (:func:`warn_nan_counts`): then the call itself needs no host sync.
    """
    if nan_counts is None:
        nan_counts = []
        tokens = sample_many(samplers, logits, forced_tokens, temperature, start_nodes, end_log,
                             nan_counts=nan_counts)
        warn_nan_counts(nan_counts)
        return tokens
    bz, seq_len, _ = logits.shape
    device = logits.device
    forced_tokens = TokenAutomaton._broadcast_forced(forced_tokens, bz)
    if start_nodes is None:
        start_nodes = [s.initial_node_id for s in samplers]
    if not torch.is_tensor(start_nodes):
        start_nodes = to_device_async(start_nodes, device)
    if end_log is None:
        end_log = [s.accept_node.log() for s in samplers]

    passes = _pack_passes(samplers, seq_len)
    if len(passes) == 1:  # every row: no copies
        return _sample_stacked(samplers, logits, forced_tokens, temperature, start_nodes, end_log, nan_counts)
    tokens = torch.empty(bz, seq_len, dtype=torch.long, device=device)
    for rows in passes:
        idx = to_device_async(rows, device)
        tokens[idx] = _sample_stacked(
            [samplers[b] for b in rows], logits[idx],
            None if forced_tokens is None else forced_tokens[idx], temperature,
            start_nodes[idx], [end_log[b] for b in rows], nan_counts,
        )
    return tokens


def warn_nan_counts(nan_counts):
    """Check the NaN-fallback counts the samplers left in ``nan_counts`` (one host sync)
    and warn if any draw fell back to uniform."""
    if nan_counts:
        total = int(torch.stack(nan_counts).sum())
        if total:
            logging.warning("NaN probabilities (fp32 underflow; matmul_dtype=float64 avoids it): "
                            "%d draws fell back to uniform", total)


def to_device_async(values, device):
    """A list of ints as a long tensor on ``device``, copied without a host sync (a pinned
    buffer; ``torch.tensor(values, device=...)`` would wait for the queued GPU work)."""
    if torch.device(device).type != "cuda":
        return torch.tensor(values, dtype=torch.long, device=device)
    return torch.tensor(values, dtype=torch.long, pin_memory=True).to(device, non_blocking=True)


#: Rows sampled in one :func:`sample_many` pass pad to at most this many
#: (bz, L, N, N) leaf elements (64 MB per such fp32 tensor).
_SHARED_MAX_LEAF_ELEMENTS = 1 << 24


def _pack_passes(samplers, seq_len):
    """Row groups for :func:`sample_many` (each in row order): by automaton size, each
    within the padding bound."""
    order = sorted(range(len(samplers)), key=lambda b: samplers[b].num_nodes)
    passes, rows = [], []
    for b in order:  # the largest so far pads the pass
        if rows and (len(rows) + 1) * seq_len * samplers[b].num_nodes ** 2 > _SHARED_MAX_LEAF_ELEMENTS:
            passes.append(sorted(rows))
            rows = []
        rows.append(b)
    passes.append(sorted(rows))
    return passes


def _sample_stacked(samplers, logits, forced_tokens, temperature, start_nodes, end_log, nan_counts):
    """One :func:`sample_many` pass: these rows' automata padded and stacked."""
    stacked = _StackedAutomata(samplers)
    seq_len = logits.shape[1]
    end_log = stacked.pad_end_log(end_log)
    lm_logits = torch.log_softmax(logits, dim=-1)
    start_w, Q = stacked.backward_log(lm_logits, forced_tokens, start_nodes, end_log)
    boundary = _forward_sample(start_w, Q, nan_counts)[:, : seq_len + 1]
    return stacked.sample_tokens_log(boundary, lm_logits, forced_tokens, start_nodes, temperature, nan_counts)


class _StackedAutomata:
    """Automata padded to common state / edge counts and stacked along the batch.

    Work that depends on the automaton is done once per distinct automaton, on
    all of its rows at once.
    """

    def __init__(self, samplers):
        distinct = list({id(s): s for s in samplers}.values())
        first = distinct[0]
        assert all((s.vocab_size, s.dtype, s.matmul_dtype, s.device)
                   == (first.vocab_size, first.dtype, first.matmul_dtype, first.device) for s in distinct)
        self.distinct = distinct
        position = {id(s): d for d, s in enumerate(distinct)}
        row_group = [position[id(s)] for s in samplers]  # row -> index into distinct
        self.device, self.dtype, self.vocab_size = first.device, first.dtype, first.vocab_size
        self.matmul_dtype = first.matmul_dtype
        self.num_nodes = N = max(s.num_nodes for s in distinct)
        self.num_edges = max(s.num_edges for s in distinct)
        edge_lookup = torch.full((len(distinct), N, N), -1, device=self.device, dtype=torch.long)
        for d, s in enumerate(distinct):
            edge_lookup[d, : s.num_nodes, : s.num_nodes] = s.edge_lookup
        self.edge_lookup = edge_lookup[to_device_async(row_group, self.device)]  # (bz, N, N)
        self._rows = [to_device_async([b for b, g in enumerate(row_group) if g == d], self.device)
                      for d in range(len(distinct))]  # each distinct automaton's rows

    def pad_end_log(self, end_log):
        """The per-row (N_b,) end weights as (bz, N), -inf on each row's padding."""
        end = torch.nn.utils.rnn.pad_sequence([e.to(self.dtype) for e in end_log], batch_first=True,
                                              padding_value=float("-inf"))
        if end.shape[1] < self.num_nodes:
            end = torch.nn.functional.pad(end, (0, self.num_nodes - end.shape[1]), value=float("-inf"))
        return end

    def backward_log(self, lm_logits, forced_tokens, start_nodes, end_log):
        """:meth:`ParallelSampler._backward_log`, each row with its own automaton."""
        bz, seq_len, _ = lm_logits.shape
        e_logits = torch.full((bz, seq_len, self.num_edges), float("-inf"),
                              device=self.device, dtype=self.dtype)
        for s, rows in zip(self.distinct, self._rows):
            forced = None if forced_tokens is None else forced_tokens[rows]
            e_logits[rows, :, : s.num_edges] = s._compute_emission_log(lm_logits[rows], forced)
        flow = _edge_flow(e_logits, self.edge_lookup)  # (bz, L, N, N)
        return _build_tree(flow, start_nodes, end_log, self.matmul_dtype)

    def sample_tokens_log(self, boundary, lm_logits, forced_tokens, start_nodes, temperature, nan_counts):
        """:meth:`ParallelSampler._sample_tokens_log`, each row with its own edges and emission."""
        bz, seq_len = boundary.shape[0], boundary.shape[1] - 1
        src = torch.cat([start_nodes[:, None], boundary[:, : seq_len - 1]], dim=1)
        dst = boundary[:, :seq_len]
        edge_ids = self.edge_lookup[torch.arange(bz, device=self.device)[:, None], src, dst]
        log_emit = torch.empty(bz, seq_len, self.vocab_size, device=self.device, dtype=self.dtype)
        for s, rows in zip(self.distinct, self._rows):
            log_emit[rows] = s.emission[edge_ids[rows].clamp(min=0)]
        return _pick_tokens(log_emit.log_(), edge_ids, lm_logits, forced_tokens, temperature, nan_counts)


def _pad_to_power_of_two(n):
    """How many identity leaves bring ``n`` leaves up to a power of two."""
    return (1 << max(n - 1, 0).bit_length()) - n
