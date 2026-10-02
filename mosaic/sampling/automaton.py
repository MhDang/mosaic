"""A token-level automaton as tensors: the constraint the samplers condition on.

Viewed as an HMM, the automaton's states are the hidden states, its edges are
the transitions and each edge emits the tokens it allows. Buffers:

    transition   (N, E)  0/1; transition[n, e] = 1 if edge e leaves node n
    emission     (E, V)  0/1; emission[e, v] = 1 if edge e allows token v
    edge_index   (E,)    destination node of each edge
    accept_node  (N,)    1 if the node is accepting
    edge_src     (E,)    source node of each edge (derived)
    edge_lookup  (N, N)  edge id between two nodes, -1 if none (derived)

and ``initial_node_id``. The samplers in ``sequential.py`` and ``parallel.py``
subclass this and add ``sample``.
"""

import json
import os

import numpy as np
import torch

from .layers import matmul_a_logb

#: steps_to_accept() value for nodes that cannot reach an accepting node.
UNREACHABLE = 1 << 30


class TokenAutomaton(torch.nn.Module):
    """Tensors of a token-level automaton, its (de)serialization and acceptance."""

    def __init__(self, num_nodes, num_edges, vocab_size):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.vocab_size = vocab_size
        self.initial_node_id = -1

        self.register_buffer("transition", torch.zeros(num_nodes, num_edges))
        self.register_buffer("emission", torch.zeros(num_edges, vocab_size))
        self.register_buffer("edge_index", torch.zeros(num_edges, dtype=torch.long))
        self.register_buffer("accept_node", torch.zeros(num_nodes))

    # ---- construction ----

    @classmethod
    def from_graph(cls, graph, device=None):
        """Build from a token graph (``edges``, ``initial_state``, ``accept_states``).

        ``mosaic.grammar.tokenize.token_graph_from_char_nfa`` produces the graph; each
        edge is ``(from, to, mask)`` with ``mask`` a boolean array over the
        vocabulary. ``device``: where the tensors are built.
        """
        edges = graph["edges"]
        initial_state = graph["initial_state"]
        accept_states = graph["accept_states"]
        vocab_size = len(edges[0][2])

        node_cnt, edge_cnt = 0, 0
        state2idx, edge2idx = {}, {}
        for u, v, _ in edges:
            if u not in state2idx:
                state2idx[u] = node_cnt
                node_cnt += 1
        for u, v, _ in edges:
            assert v in state2idx, "each state must have an outgoing edge"
            uid, vid = state2idx[u], state2idx[v]
            assert (uid, vid) not in edge2idx
            edge2idx[(uid, vid)] = edge_cnt
            edge_cnt += 1

        initial_id = state2idx[initial_state]
        accept_ids = [state2idx[s] for s in accept_states]

        transition = torch.zeros(node_cnt, edge_cnt)
        accept_node = torch.zeros(node_cnt)
        accept_node[accept_ids] = 1.0
        edge_index = torch.full((edge_cnt,), -1, dtype=torch.long)

        for u, v, _ in edges:
            uid, vid = state2idx[u], state2idx[v]
            eid = edge2idx[(uid, vid)]
            transition[uid, eid] = 1.0
            edge_index[eid] = vid
        # Edges are numbered in list order, so the emission's rows are the masks.
        emission = torch.from_numpy(np.stack([mask for _, _, mask in edges]))

        with torch.device(device or "cpu"):
            model = cls(node_cnt, edge_cnt, vocab_size)
        with torch.no_grad():
            model.transition.copy_(transition)
            model.emission.copy_(emission)
            model.edge_index.copy_(edge_index)
            model.accept_node.copy_(accept_node)
            model.initial_node_id = initial_id
        model._setup()
        return model

    def _setup(self):
        """Register the derived buffers; subclasses extend this."""
        N, E = self.num_nodes, self.num_edges
        device = self.transition.device
        edge_src = self.transition.argmax(dim=0)  # (E,)
        edge_lookup = torch.full((N, N), -1, dtype=torch.long, device=device)
        edge_lookup[edge_src, self.edge_index] = torch.arange(E, device=device)
        self.register_buffer("edge_lookup", edge_lookup)
        self.register_buffer("edge_src", edge_src)

    # ---- serialization ----

    def save_pretrained(self, save_directory):
        """Save the (sparse-packed) tensors and config to a directory."""
        os.makedirs(save_directory, exist_ok=True)
        torch.save(
            {
                "transition": _pack(self.transition),
                "emission": _pack(self.emission),
                "edge_index": self.edge_index.cpu(),
                "accept_node": _pack(self.accept_node),
            },
            os.path.join(save_directory, f"pytorch_model.bin.tmp{os.getpid()}"),
        )
        os.replace(os.path.join(save_directory, f"pytorch_model.bin.tmp{os.getpid()}"),
                   os.path.join(save_directory, "pytorch_model.bin"))
        # config.json last, and atomically: readers take it to mean the entry is complete
        config = os.path.join(save_directory, "config.json")
        with open(config + f".tmp{os.getpid()}", "w") as f:
            json.dump(
                {
                    "num_nodes": self.num_nodes,
                    "num_edges": self.num_edges,
                    "vocab_size": self.vocab_size,
                    "initial_node_id": self.initial_node_id,
                },
                f,
            )
        os.replace(config + f".tmp{os.getpid()}", config)

    @classmethod
    def from_pretrained(cls, directory, device=None):
        """Load an automaton saved by :meth:`save_pretrained`.

        With ``device`` the tensors are built there, straight from the saved
        sparse indices: much faster than building the dense (E, V) emission on
        the CPU and copying it over.
        """
        with open(os.path.join(directory, "config.json")) as f:
            cfg = json.load(f)
        d = torch.load(os.path.join(directory, "pytorch_model.bin"), map_location="cpu")

        with torch.device(device or "cpu"):
            model = cls(cfg["num_nodes"], cfg["num_edges"], cfg["vocab_size"])
        model.initial_node_id = cfg["initial_node_id"]
        with torch.no_grad():
            _unpack_into(model.transition, d["transition"])
            _unpack_into(model.emission, d["emission"])
            model.edge_index.copy_(d["edge_index"])
            _unpack_into(model.accept_node, d["accept_node"])
        model._setup()
        return model

    @staticmethod
    def is_saved(directory):
        """Whether ``directory`` holds a saved automaton."""
        return os.path.isfile(os.path.join(directory, "config.json"))

    def _apply(self, fn, *args, **kwargs):
        # .to() / .double() / ...: derived caches hold the old tensors or device
        for name in ("_steps_to_accept", "_max_steps_to_accept", "_finish_within"):
            self.__dict__.pop(name, None)
        return super()._apply(fn, *args, **kwargs)

    # ---- properties ----

    @property
    def dtype(self):
        return self.transition.dtype

    @property
    def device(self):
        return self.transition.device

    # ---- acceptance ----

    @torch.no_grad()
    def accepts(self, input_ids):
        """BoolTensor (bz,): whether each row of ``input_ids`` (bz, L) is accepted.

        Any length. A position that is out of vocabulary or negative (e.g. a
        still-masked -1) matches every edge that allows some token.
        """
        bz, seq_len = input_ids.shape
        N = self.num_nodes
        allowed = (self.emission > 0).to(self.dtype)  # (E, V) support only
        any_token = allowed.sum(dim=-1).clamp(max=1.0)  # (E,)

        cur = torch.zeros(bz, N, device=self.device, dtype=self.dtype)
        cur[:, self.initial_node_id] = 1.0
        for t in range(seq_len):
            ids = input_ids[:, t]
            missing = (ids < 0) | (ids >= self.vocab_size)
            emit = allowed[:, ids.clamp(0, self.vocab_size - 1)].T  # (bz, E)
            emit[missing] = any_token
            active = cur[:, self.edge_src] * emit  # (bz, E)
            cur = torch.zeros_like(cur).index_add_(1, self.edge_index, active)
            cur = (cur > 0).to(self.dtype)
        return (cur * self.accept_node).sum(dim=-1) > 0

    @torch.no_grad()
    def advance(self, nodes, tokens):
        """The nodes reached from ``nodes`` (bz,) after reading ``tokens`` (bz, L).

        -1 marks a row that has been rejected (or started at -1). Every compiled
        constraint is deterministic, so each (node, token) has at most one edge.
        """
        nodes = torch.as_tensor(nodes, dtype=torch.long, device=self.device)
        allowed = self.allowed_edges(tokens).cpu().numpy()
        return torch.from_numpy(self.walk(nodes.cpu().numpy(), allowed)).to(self.device)

    def allowed_edges(self, tokens):
        """(bz, L, E) bool on the device: whether each edge allows each token of ``tokens`` (bz, L).

        One gather of just these columns of the emission; :meth:`walk` then
        follows them on the host.
        """
        tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        in_vocab = (tokens >= 0) & (tokens < self.vocab_size)
        allowed = self.emission[:, tokens.clamp(0, self.vocab_size - 1)].permute(1, 2, 0) > 0
        return allowed & in_vocab[..., None]

    def walk(self, nodes, allowed):
        """NumPy: the nodes reached from ``nodes`` (bz,) through :meth:`allowed_edges`' result."""
        edge_src, edge_index = self._edges_numpy()
        cur = np.asarray(nodes, dtype=np.int64)
        for t in range(allowed.shape[1]):
            match = (edge_src[None, :] == cur[:, None]) & allowed[:, t]  # (bz, E)
            ok = match.any(axis=-1) & (cur >= 0)
            cur = np.where(ok, edge_index[match.argmax(axis=-1)], -1)
        return np.asarray(cur, dtype=np.int64)

    def _edges_numpy(self):
        """(edge_src, edge_index) as NumPy arrays, cached: the automaton never changes."""
        cached = self.__dict__.get("_edges_np")
        if cached is None:
            cached = self._edges_np = (self.edge_src.cpu().numpy(), self.edge_index.cpu().numpy())
        return cached

    def steps_to_accept(self):
        """(N,) LongTensor: the fewest tokens from each node to an accepting node.

        Nodes that cannot reach acceptance get ``UNREACHABLE``. Computed once by
        breadth-first search over reversed edges and cached.
        """
        cached = self.__dict__.get("_steps_to_accept")
        if cached is not None and cached.device == self.device:
            return cached
        src, dst = self.edge_src.tolist(), self.edge_index.tolist()
        preds = [[] for _ in range(self.num_nodes)]
        for s, d in zip(src, dst):
            preds[d].append(s)
        dist = [UNREACHABLE] * self.num_nodes
        frontier = [n for n, a in enumerate(self.accept_node.tolist()) if a > 0]
        for n in frontier:
            dist[n] = 0
        while frontier:
            nxt = []
            for d in frontier:
                for s in preds[d]:
                    if dist[s] == UNREACHABLE:
                        dist[s] = dist[d] + 1
                        nxt.append(s)
            frontier = nxt
        self._steps_to_accept = torch.tensor(dist, dtype=torch.long, device=self.device)
        self._max_steps_to_accept = max([d for d in dist if d < UNREACHABLE], default=0)
        return self._steps_to_accept

    def finish_within_log(self, remaining):
        """(N,) log end weights: 0 where acceptance is reachable within ``remaining`` tokens.

        As the end weights of a sampler call, this conditions a block on the
        output still being completable in the tokens left after it. Cached per
        ``remaining`` (all values past the farthest finite distance give the same
        weights); do not modify the result in place.
        """
        steps = self.steps_to_accept()
        remaining = min(max(int(remaining), 0), self._max_steps_to_accept)
        cache = self.__dict__.setdefault("_finish_within", {})
        if remaining not in cache:
            cache[remaining] = torch.zeros(self.num_nodes, device=self.device, dtype=self.dtype).masked_fill(
                ~(steps <= remaining), float("-inf")
            )
        return cache[remaining]

    # ---- helpers for samplers ----

    def _resolve_ends(self, bz, start_nodes=None, end_log=None):
        """Per-row start nodes (bz,) and log end weights (bz, N).

        Defaults: the initial node, and 0 on accepting nodes (-inf elsewhere).
        """
        if start_nodes is None:
            start_nodes = torch.full((bz,), self.initial_node_id, dtype=torch.long, device=self.device)
        else:
            start_nodes = torch.as_tensor(start_nodes, dtype=torch.long, device=self.device)
            start_nodes = start_nodes.reshape(-1).expand(bz) if start_nodes.numel() == 1 else start_nodes
        if end_log is None:
            end_log = self.accept_node.log().unsqueeze(0).expand(bz, -1)
        else:
            end_log = torch.as_tensor(end_log, device=self.device).to(self.dtype)
            end_log = end_log.unsqueeze(0).expand(bz, -1) if end_log.dim() == 1 else end_log
        return start_nodes, end_log


    def _compute_emission_log(self, lm_logits, forced_tokens=None):
        """e_logits[b, t, e] = log sum_v emission[e, v] * exp(lm_logits[b, t, v]).

        One (E, V) x (V, bz*L) matmul; a forced position reduces to the 0/1
        indicator of its token instead.
        """
        bz, L, V = lm_logits.shape
        # The automaton is kept in fp32; logits may arrive in bf16/fp16.
        flat = lm_logits.reshape(bz * L, V).to(self.emission.dtype)  # (bz*L, V)
        out = matmul_a_logb(self.emission, flat.T)  # (E, bz*L)
        e_logits = out.T.reshape(bz, L, -1)  # (bz, L, E)

        if forced_tokens is not None:  # torch.where over a gather: no host sync
            forced_log = self.emission[:, forced_tokens.clamp(min=0)].log().permute(1, 2, 0)  # (bz, L, E)
            e_logits = torch.where((forced_tokens != -1)[..., None], forced_log, e_logits)

        return e_logits


    @staticmethod
    def _broadcast_forced(forced_tokens, bz):
        """Broadcast a (seq_len,) ``forced_tokens`` to (bz, seq_len); None stays None."""
        if forced_tokens is None:
            return None
        if forced_tokens.dim() == 1:
            return forced_tokens.unsqueeze(0).expand(bz, -1)
        return forced_tokens


# ---- storage: nonzero indices (+ values when not 0/1) ----


def _pack(x):
    """Store a tensor as its nonzero indices, plus values if it is not 0/1.

    Emissions are sparse, so this is several times smaller than a bitmask.
    """
    idx = x.nonzero(as_tuple=False)
    is_binary = torch.equal(x, (x != 0).to(x.dtype))
    out = {"indices": idx.to(torch.int32).cpu(), "shape": tuple(x.shape)}
    if not is_binary and idx.numel() > 0:
        out["values"] = x[tuple(idx[:, i] for i in range(idx.shape[1]))].clone().cpu()
    return out


def _unpack_into(x, obj):
    """Inverse of :func:`_pack`: write it into the zero tensor ``x``, on x's device."""
    assert tuple(x.shape) == tuple(obj["shape"]), (tuple(x.shape), obj["shape"])
    idx = obj["indices"].to(x.device, torch.long)
    vals = obj.get("values", None)
    if idx.numel() > 0:
        coords = tuple(idx[:, i] for i in range(idx.shape[1]))
        x[coords] = vals.to(x.device, x.dtype) if vals is not None else 1.0
    return x
