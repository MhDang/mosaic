"""Sequential sampler: exact LM x automaton sampling by forward-backward.

The same posterior as ``parallel.py``, computed the HMM way: a backward pass
right-to-left over positions accumulates, per edge, the LM-weighted mass of
accepting completions; a forward pass left-to-right then draws one token per
position from (LM x automaton) marginalized over edges, and advances the
automaton state by the drawn token. O(L) sequential depth, any ``seq_len``, and
O(bz x L x E) memory instead of the parallel sampler's O(bz x L x N^2).
"""

import logging

import torch

from .automaton import TokenAutomaton
from .layers import matmul_a_logb, matmul_loga_b, scatter_logsumexp


class SequentialSampler(TokenAutomaton):
    """Samples from LM x automaton with a sequential forward-backward over edges."""

    def _setup(self):
        """Also register the transposed transition (E, N) and emission (V, E)."""
        super()._setup()
        self.register_buffer("transition_T", self.transition.T.contiguous())
        self.register_buffer("emission_T", self.emission.T.contiguous())

    @torch.no_grad()
    def sample(self, logits, forced_tokens=None, temperature=1.0,
               return_marginal=False, start_nodes=None, end_log=None):
        """Draw from the LM x automaton product. Returns (tokens, marginal).

        logits: (bz, L, V) LM logits for the generated span; any L.
        forced_tokens: (L,) or (bz, L); -1 marks a free position.
        temperature: for each drawn token (0 = argmax of the conditional).
        return_marginal: also return, per position, the probability of the
            drawn token under the exact LM x automaton marginal (bz, L);
            otherwise the second value is None.
        start_nodes / end_log: per-row start node and log end weights, as in
            :meth:`ParallelSampler.sample`.
        """
        bz = logits.shape[0]
        lm_logits = torch.log_softmax(logits, dim=-1)
        forced_tokens = self._broadcast_forced(forced_tokens, bz)
        start_nodes, end_log = self._resolve_ends(bz, start_nodes, end_log)
        back_y = self._backward_log(lm_logits, forced_tokens, end_log)
        tokens = self._forward_sample_log(
            back_y, lm_logits, forced_tokens, start_nodes,
            temperature=temperature,
        )
        del back_y

        marginal = None
        if return_marginal:
            marginal = self._compute_marginal_log(
                lm_logits, forced_tokens, sampled_tokens=tokens,
                start_nodes=start_nodes, end_log=end_log,
            ).exp()
        return tokens, marginal

    # ---- backward ----

    def _backward_log(self, lm_logits, forced_tokens, end_log):
        """Right-to-left pass in edge space.

        Returns back_y, a list of (bz, E) tensors: back_y[k][b, e] is the log
        LM-weighted mass of accepting completions of positions k..L-1 given the
        automaton sits at dest(e) before position k (up to a per-row constant,
        which cancels in sampling). Storing it before position k's emission lets
        the forward pass form the per-token weight in one matmul.
        """
        bz, L, _ = lm_logits.shape
        dtype = lm_logits.dtype

        # p_emit[b, k, e] = log sum_v emission[e, v] exp(lm_logits[b, k, v])
        p_emit_all = matmul_loga_b(lm_logits.reshape(bz * L, -1), self.emission_T)
        p_emit_all = p_emit_all.reshape(bz, L, -1)  # (bz, L, E)

        y = end_log[:, self.edge_index].to(dtype).clone()  # (bz, E): end weight of dest(e)

        back_y = [None] * L
        for k in range(L - 1, -1, -1):
            back_y[k] = y

            p_emit = p_emit_all[:, k]  # (bz, E)
            if forced_tokens is not None:
                forced_mask = forced_tokens[:, k] != -1  # (bz,)
                if forced_mask.any():
                    forced_ids = forced_tokens[forced_mask, k].clamp(min=0)
                    emit_mask = self.emission_T[forced_ids].bool()  # (n_forced, E)
                    p_emit = p_emit.clone()
                    p_emit[forced_mask] = emit_mask.new_zeros(
                        emit_mask.shape, dtype=p_emit.dtype
                    ).masked_fill(~emit_mask, float("-inf"))  # 0 or -inf

            h = p_emit + y  # (bz, E)
            node_val = matmul_loga_b(h, self.transition_T)  # (bz, N)
            y = node_val[:, self.edge_index]  # (bz, E)

            m = y.amax(dim=-1, keepdim=True)
            y = y - m
            y.nan_to_num_(nan=float("-inf"))

        return back_y

    # ---- forward ----

    def _forward_sample_log(self, back_y, lm_logits, forced_tokens, start_nodes,
                            temperature=1.0):
        """Left-to-right ancestral sampling. Returns tokens (bz, L).

        token_weight[b, v] = lm[b, k, v] * sum_e x[b, e] back_y[k][b, e] emission[e, v]
        """
        bz, L, _ = lm_logits.shape
        device, dtype = self.device, lm_logits.dtype
        N = self.num_nodes

        tokens = torch.zeros(bz, L, dtype=torch.long, device=device)

        # x[b, e]: log forward mass on edge e
        x = self.transition_T[:, start_nodes].T.to(dtype).log().contiguous()  # (bz, E)

        def _sample(logits):
            if temperature == 0:
                return torch.softmax(logits.float(), dim=-1).argmax(dim=-1)
            scaled = logits.float() / temperature
            probs = torch.softmax(scaled, dim=-1)
            bad = probs.isnan().any(dim=-1)
            if bad.any():
                logging.warning("NaN probabilities: %d of %d draws fell back to uniform",
                                bad.sum().item(), bad.numel())
                probs[bad] = 1.0 / N
            return torch.multinomial(probs, 1).squeeze(-1)

        for k in range(L):
            if forced_tokens is not None:
                forced_mask = forced_tokens[:, k] != -1  # (bz,)
                free_mask = ~forced_mask
            else:
                free_mask = torch.ones(bz, dtype=torch.bool, device=device)
                forced_mask = ~free_mask

            xb = x + back_y[k]  # (bz, E)

            if free_mask.any():
                nfa_w = matmul_loga_b(xb[free_mask], self.emission)  # (n_free, V)
                token_w = nfa_w + lm_logits[free_mask, k]
                tokens[free_mask, k] = _sample(token_w)

            if forced_mask.any():
                tokens[forced_mask, k] = forced_tokens[forced_mask, k]

            # advance: keep edges that allow the token, collect at their
            # destination node, then step out along that node's edges
            token_ids = tokens[:, k]
            emit_mask = self.emission_T[token_ids].bool()  # (bz, E)
            x_weighted = x.masked_fill(~emit_mask, float("-inf"))
            node_val = scatter_logsumexp(x_weighted, self.edge_index, N)  # (bz, N)
            x = matmul_loga_b(node_val, self.transition)

        return tokens

    # ---- exact per-position marginals ----

    def _compute_marginal_log(self, lm_logits, forced_tokens=None, sampled_tokens=None,
                              start_nodes=None, end_log=None):
        """Log LM x automaton marginal p(x_t = v | constraint), by O(L) forward-backward.

        alpha[b, t, s]: log mass of paths from the start node to s after t
        tokens; beta[b, t, s]: log mass of accepting completions from s over
        positions t..L-1 (LM integrated at free positions, forced positions
        reduced to their 0/1 indicator). Z = beta[b, 0, start].

            p(x_t = v) ∝ p_LM(x_t = v) * sum_e alpha_t[src e] emission[e, v] beta_{t+1}[dst e]

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

        e_logits = self._compute_emission_log(lm_logits, forced_tokens)  # (bz, L, E)
        leaf = e_logits[:, :, None, :]
        leaf = leaf.masked_fill(self.transition[None, None, :, :] == 0, float("-inf"))
        flow = scatter_logsumexp(leaf, self.edge_index, N)  # (bz, L, N_src, N_dst)

        neg_inf = float("-inf")
        log_alpha = torch.full((bz, L + 1, N), neg_inf, device=device, dtype=dtype)
        log_alpha[bz_dim, 0, start_nodes] = 0.0
        for t in range(L):
            log_alpha[:, t + 1] = torch.logsumexp(
                log_alpha[:, t, :, None] + flow[:, t, :, :], dim=-2
            )

        log_beta = torch.full((bz, L + 1, N), neg_inf, device=device, dtype=dtype)
        log_beta[:, L] = (
            end_log.to(dtype)
        )
        for t in range(L - 1, -1, -1):
            log_beta[:, t] = torch.logsumexp(
                flow[:, t, :, :] + log_beta[:, t + 1, None, :], dim=-1
            )

        log_Z = log_beta[bz_dim, 0, start_nodes]  # (bz,)

        alpha_src = log_alpha[:, :L, :].gather(
            -1, self.edge_src[None, None, :].expand(bz, L, -1)
        )  # (bz, L, E)
        beta_dst = log_beta[:, 1:, :].gather(
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
