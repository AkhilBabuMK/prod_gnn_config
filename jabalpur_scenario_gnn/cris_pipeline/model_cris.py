# -*- coding: utf-8 -*-
"""
cris_pipeline/model_cris.py
===========================
Corridor next-event model.

Same skeleton as the previous NextEventGNN — heterogeneous message passing,
monotone quantile heads, pinball loss — with one substantive addition and one
correction.

ADDITION — JourneyHistoryEncoder
    The old model could only see `delay_min` (now) and a one-tick
    `delay_velocity`. Journey history reached it solely through a GRU memory
    that was (a) keyed by SERVICE rather than by run, (b) stepped once per
    CLOCK TICK rather than per halt, and (c) truncated at TBPTT_K=12 ticks —
    about 60 minutes against a 3-5 hour traverse. It was therefore structurally
    unable to represent how delay had propagated along the run.

    JourneyHistoryEncoder runs a GRU over the stops the train has ALREADY
    completed on this run, carrying the per-leg delay delta, dwell and run-over.

    WHERE IT ENTERS — the two signals are kept on separate paths and joined at
    the head:

        h_gnn   = network state: how other trains, sections and stations
                  constrain this train right now
        h_hist  = own dynamics: how this run's delay has propagated so far
        head([h_gnn, h_hist]) -> next-event quantiles

    History is deliberately NOT mixed into the node input by default. Feeding
    it pre-GNN would (a) force the signal through four conv layers and four
    LayerNorms before it reaches the loss, which is a lot of routing to learn
    from 19 training days, and (b) let the model satisfy the objective by
    reading its own history while message passing atrophies — the same
    delay-persistence failure as before, but hidden inside the GNN instead of
    visible in the features.

    `history_to_gnn=True` restores the pre-GNN injection so the two can be
    compared rather than assumed: a neighbour's TREND is arguably informative
    (a train taking +5 min every leg will keep blocking; one recovering from a
    single 40-min hit will clear).

CORRECTION — the edge set matches what the corridor dataset actually builds.
    The old model declared `train->approaching->station` edges that the
    corridor snapshots never produce.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv

from scenario_events import CONFLICT_EDGE_DIM

# Single source of truth for the graph schema: `make_conv` builds a module per
# entry and `_edge_index` feeds edges per entry, so the two cannot disagree.
EDGE_TYPES = [
    ("station", "track", "station"),
    ("train", "at", "station"),
    ("station", "has", "train"),
    ("train", "approaching", "station"),
    ("station", "awaits", "train"),
    ("section", "connects", "station"),
    ("station", "borders", "section"),
    ("train", "traverses", "section"),
    ("section", "carries", "train"),
    ("train", "conflicts", "train"),
]

NUM_CLASSES = 15
NUM_GROUPS = 4


class JourneyHistoryEncoder(nn.Module):
    """GRU over the halts already completed on this run.

    Input  : [T, L, F] oldest-first, zero-padded at the end
             [T]       true length per train
    Output : [T, out_dim]
    """

    def __init__(self, feat_dim: int, hidden: int = 64, out_dim: int = 48):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden, out_dim), nn.ReLU(), nn.LayerNorm(out_dim))
        self.out_dim = out_dim

    def forward(self, hist: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if hist.numel() == 0:
            return hist.new_zeros((hist.shape[0], self.out_dim))

        out, _ = self.gru(hist)                        # [T, L, H]
        # Gather the step at each train's true final position. Using the packed
        # final state would be equivalent, but gathering keeps this safe when a
        # length is clamped and avoids sorting the batch every call.
        idx = (lengths.clamp(min=1) - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))
        last = out.gather(1, idx).squeeze(1)            # [T, H]
        return self.out(last)


class CorridorNextEventGNN(nn.Module):
    def __init__(
        self,
        station_in_dim: int,
        train_in_dim: int,
        section_in_dim: int,
        hist_in_dim: int,
        num_services: int,
        num_stations: int,
        num_sections: int,
        hidden: int = 128,
        memory_dim: int = 80,
        embedding_dim: int = 20,
        hist_dim: int = 48,
        dropout: float = 0.20,
        history_to_gnn: bool = False,
        service_dropout: float = 0.0,
        hurdle: bool = False,
    ):
        super().__init__()
        self.hidden = hidden
        self.memory_dim = memory_dim
        self.num_stations = num_stations
        self.num_sections = num_sections
        self.num_services = num_services
        self.history_to_gnn = history_to_gnn
        # Probability of hiding a train's SERVICE IDENTITY during training.
        #
        # `service_emb` gives every service its own learned row, but only 48 of
        # 301 services run near-daily. Measured on the 19-day training split:
        # 42 rows are never touched by gradient at all and still sit at their
        # random init (norm 0.092 against 0.24 for trained rows), and those
        # services carry 10.9% of TEST supervision -- 23% counting services
        # seen on only 1-2 days.
        #
        # So on roughly a quarter of the test set the model meets a condition
        # it has NEVER trained under: no usable service identity. Hiding the
        # embedding at random makes that condition in-distribution instead.
        #
        # Zero is the right stand-in rather than a learned UNK row: an
        # untrained row is already a near-null vector at inference, so training
        # against zero matches what the model will actually receive, and no
        # inference-side change is needed.
        #
        # MEASURED AT 0.25 (v5) — IT DOES NOT WORK. Leave at 0.0.
        # Median journey error by training exposure, v3 -> v5:
        #     never trained  17.32 -> 17.33   no effect at all
        #     1-2 days        8.24 ->  9.55   WORSE, and this is the group it
        #                                     was designed for
        #     3-5 days        7.47 ->  6.68
        #     6+ days         5.73 ->  5.42
        #
        # The diagnosis behind it was wrong. Rare services are not a case of
        # the model OVER-RELYING on identity it has plenty of; they are a case
        # of having almost no identity signal to begin with. Dropout rations a
        # signal that is already starved, so it makes the thin case thinner.
        # That is a data-volume problem and this is the wrong instrument for it.
        self.service_dropout = float(service_dropout)

        e, eh = embedding_dim, embedding_dim // 2

        self.class_emb = nn.Embedding(NUM_CLASSES, e)
        self.group_emb = nn.Embedding(NUM_GROUPS, eh)
        self.service_emb = nn.Embedding(num_services, e)

        self.history_encoder = JourneyHistoryEncoder(hist_in_dim, out_dim=hist_dim)

        train_proj_in = train_in_dim + memory_dim + e + e + eh
        if history_to_gnn:
            train_proj_in += hist_dim
        station_proj_in = station_in_dim + memory_dim
        section_proj_in = section_in_dim + memory_dim

        def proj(d):
            return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.LayerNorm(hidden))

        self.station_proj = proj(station_proj_in)
        self.train_proj = proj(train_proj_in)
        self.section_proj = proj(section_proj_in)

        def make_conv():
            return HeteroConv({
                ("station", "track", "station"):
                    SAGEConv((hidden, hidden), hidden, aggr="mean"),
                ("train", "at", "station"):
                    SAGEConv((hidden, hidden), hidden, aggr="sum"),
                ("station", "has", "train"):
                    SAGEConv((hidden, hidden), hidden, aggr="sum"),
                # The station a train is APPROACHING. Congestion there is what
                # actually creates delay on this corridor (trains held at
                # platforms), and without this edge it reached the train only
                # as an undirected blend of both neighbours. `mean` on the
                # inbound direction so a train reads the state of the platform
                # ahead; `sum` on the reverse so a station accumulates the
                # pressure of everything converging on it.
                ("train", "approaching", "station"):
                    SAGEConv((hidden, hidden), hidden, aggr="sum"),
                ("station", "awaits", "train"):
                    SAGEConv((hidden, hidden), hidden, aggr="mean"),
                ("section", "connects", "station"):
                    SAGEConv((hidden, hidden), hidden, aggr="mean"),
                ("station", "borders", "section"):
                    SAGEConv((hidden, hidden), hidden, aggr="mean"),
                ("train", "traverses", "section"):
                    SAGEConv((hidden, hidden), hidden, aggr="sum"),
                ("section", "carries", "train"):
                    SAGEConv((hidden, hidden), hidden, aggr="mean"),
                ("train", "conflicts", "train"):
                    GATConv((hidden, hidden), hidden, heads=4, concat=False,
                            edge_dim=CONFLICT_EDGE_DIM, add_self_loops=False,
                            dropout=0.1),
            }, aggr="mean")

        self.convs = nn.ModuleList([make_conv() for _ in range(4)])
        self.lns = nn.ModuleList([
            nn.ModuleDict({
                "s": nn.LayerNorm(hidden),
                "t": nn.LayerNorm(hidden),
                "sec": nn.LayerNorm(hidden),
            }) for _ in range(4)
        ])
        self.drop = nn.Dropout(dropout)

        # Heads see BOTH paths: GNN network state and own journey dynamics.
        head_in = hidden + hist_dim

        # Plain regression, ONE number: the change in delay at the next stop.
        #
        # This was a 3-quantile head trained with pinball loss. Pinball at
        # q=0.5 is exactly L1, which fits the MEDIAN. Per-leg delay change is
        # strongly right-skewed (delays blow out much further than they
        # recover), so the median sits below the mean, and feeding a median
        # back into itself hop after hop produced the flat forecasts seen in
        # rollout: train 22688 predicted +285 -> +300 across 16 stations while
        # the truth ran +292 -> +396, a -20 min bias at the 2-4 h horizon.
        # A squared-error-shaped head fits the MEAN and carries that drift.
        def reghead():
            return nn.Sequential(
                nn.Linear(head_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, 1))

        self.next_arr_head = reghead()
        self.next_dep_head = reghead()

        # HOW MUCH LONGER WILL THIS TRAIN STAND HERE?
        #
        # The only head that asks about the PRESENT. `next_arr` and `next_dep`
        # both describe the next station; nothing described the platform the
        # train is on right now, which is why a hold could never lengthen
        # inside a rollout: a stop's departure is written while the train is
        # still one stop back and is never revisited.
        #
        # This head is queried at every tick while a train stands, and its
        # answer extends that departure. The train then stays on the platform
        # because the model keeps saying it will -- which is what generates a
        # queue behind it, and hence a cascade.
        #
        # Trained on 70,353 standing observations against the 295 large delay
        # events the spike work has been starved by. A spike IS a long stand,
        # so this re-poses our rarest-event regression as our densest duration
        # problem.
        #
        # softplus so the answer is always a non-negative number of minutes:
        # a negative wait is not a thing, and clamping after the fact would
        # leave a dead gradient wherever the raw output went negative.
        self.remaining_dwell_head = reghead()

        # ── HURDLE HEADS ──────────────────────────────────────────────────────
        #
        # `next_arr_head` above must emit ONE number for a target that is
        # near-zero 98.6% of the time and +50/+80/+120 the other 1.4%. Under
        # any average-error loss the minimiser of that mixture sits near the
        # common case, which is why 84% of spike predictions come out too low.
        # That is not a tuning failure -- it is what a single point estimate on
        # a mixture HAS to do.
        #
        # Three experiments confirmed the head is the constraint, not the
        # knob: v4 (recovery weight 3->10), v5 (service dropout) and v6 (full
        # scheduled sampling) each pushed that one number in a different
        # direction and all three landed statistically tied with v3. v6 made
        # the tension explicit -- it cut runaways 3->1 AND dropped spike recall
        # 86.0% -> 73.6%, because "never run away" and "commit to big rises"
        # are opposite demands on the same output.
        #
        # So split them:
        #   spike_logit  P(|delta| > SPIKE_THRESHOLD)  -- a probability, so it
        #                CANNOT run away however bold it is, and detection is
        #                the part we already do well (AUC 0.864).
        #   spike_mag    the SIGNED delta given that it is large, trained only
        #                on those events so the quiet majority cannot wash it
        #                out. Signed, so one pair of heads covers both spikes
        #                and recoveries.
        #
        # `next_arr_head` keeps its existing target and training signal, so
        # with the decision threshold at 1.0 this model degenerates EXACTLY to
        # v6. The change is strictly additive and that degeneracy is testable.
        # OFF BY DEFAULT. Measured in v7 and it does not work, in two separate
        # ways:
        #
        #   1. Used at inference, it HURTS. On the events where it fired it was
        #      worse even on the TRUE large changes (19.01 -> 26.62), because
        #      it was trained on per-hop jumps (median +46) while the rollout
        #      scores CUMULATIVE rises -- a train losing 12 min at each of five
        #      stops is a spike operationally and invisible to a per-hop
        #      classifier. Firing on consecutive hops then adds a full
        #      event-sized jump each time.
        #
        #   2. Merely TRAINING them damages the shared representation. v7
        #      scored with the hurdle DISABLED still went from 1 runaway
        #      journey to 7, and lost on every summary statistic against v6.
        #      Auxiliary heads are not free even when unused.
        #
        # Kept behind a flag rather than deleted so the negative result stays
        # reproducible from the release directories.
        self.hurdle = bool(hurdle)
        if self.hurdle:
            self.spike_logit_head = nn.Sequential(
                nn.Linear(head_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, 1))
            self.spike_mag_head = reghead()

        # REMOVED — `station_critical_head`. It emitted a logit per station on
        # every forward pass and NOTHING supervised it: train_cris.py has no
        # loss term for it, so its 8,321 parameters stayed at initialisation
        # for all nine training runs while diagnose_model.py explicitly
        # filtered it out of every analysis. An output that looks like a
        # prediction and is pure noise is worse than no output.
        #
        # Checkpoints v3-v10 still carry its weights; every loader here reads
        # them with strict=False and reports the discarded keys, so nothing
        # older breaks. `d["station"].y_critical` stays in the dataset — the
        # LABEL is real and cheap to keep, and re-adding a supervised head
        # later should not need a rebuild.
        self.station_memory_updater = nn.GRUCell(hidden, memory_dim)
        self.service_memory_updater = nn.GRUCell(hidden, memory_dim)
        self.section_memory_updater = nn.GRUCell(hidden, memory_dim)

        self.register_buffer("station_memory", torch.zeros(num_stations, memory_dim))
        self.register_buffer("service_memory", torch.zeros(num_services, memory_dim))
        self.register_buffer("section_memory", torch.zeros(num_sections, memory_dim))

        self.reset_parameters()

    # ── Init ──────────────────────────────────────────────────────────────────

    def reset_parameters(self) -> None:
        """Explicit init.

        PyTorch defaults leave Linear layers at Kaiming-uniform for a
        leaky_relu(a=sqrt(5)) gain, which under-scales genuine ReLU stacks.
        Every Linear here is followed by ReLU or feeds a head, so use the
        matching ReLU gain and zero the biases. GRU weights get orthogonal
        recurrent matrices so the history encoder does not vanish over 19 steps.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, (nn.GRU, nn.GRUCell)):
                for name, p in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(p)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

        # Start the regression heads near zero so the first prediction is
        # "delay does not change" (persistence) rather than noise, which makes
        # early-epoch loss directly interpretable against the persistence
        # baseline.
        #
        # SCALE DOWN, do not zero. With W == 0 the backward pass gives
        # dL/dh = W^T delta = 0, so the conv stack, history encoder and every
        # embedding receive exactly zero gradient — the whole network is dead
        # at step 1. A small non-zero init keeps predictions near persistence
        # while leaving the gradient path intact.
        for head in (self.next_arr_head, self.next_dep_head,
                     self.remaining_dwell_head):
            last = [m for m in head if isinstance(m, nn.Linear)][-1]
            with torch.no_grad():
                last.weight.mul_(0.01)
            nn.init.zeros_(last.bias)

    # ── Memory ────────────────────────────────────────────────────────────────

    def reset_memory(self) -> None:
        dev = self.station_memory.device
        self.station_memory = torch.zeros(self.num_stations, self.memory_dim, device=dev)
        self.service_memory = torch.zeros(self.num_services, self.memory_dim, device=dev)
        self.section_memory = torch.zeros(self.num_sections, self.memory_dim, device=dev)

    def detach_memory(self) -> None:
        self.station_memory = self.station_memory.detach()
        self.service_memory = self.service_memory.detach()
        self.section_memory = self.section_memory.detach()

    # ── Forward ───────────────────────────────────────────────────────────────

    def _encode(self, data, hist):
        srv = data["train"].service_idx
        svc = self.service_emb(srv)
        if self.training and self.service_dropout > 0.0:
            # Per TRAIN-NODE, not per feature: the model must learn to run with
            # this train's identity absent while its neighbours keep theirs,
            # which is exactly the test-time situation.
            keep = (torch.rand(svc.shape[0], 1, device=svc.device)
                    >= self.service_dropout).float()
            svc = svc * keep
        parts = [
            data["train"].x,
            self.service_memory[srv],
            svc,
            self.class_emb(data["train"].class_idx),
            self.group_emb(data["train"].group_idx),
        ]
        if self.history_to_gnn:
            parts.append(hist)
        h_t = self.train_proj(torch.cat(parts, dim=-1))
        h_s = self.station_proj(
            torch.cat([data["station"].x, self.station_memory], dim=-1))
        h_sec = self.section_proj(
            torch.cat([data["section"].x, self.section_memory], dim=-1))
        return h_s, h_t, h_sec

    @staticmethod
    def _edge_index(data) -> dict:
        # Derived from EDGE_TYPES, never re-listed. This was a hardcoded copy
        # of the conv keys, so adding ("train","approaching","station") built
        # the conv modules but never passed them any edges — they received
        # exactly zero gradient and would have trained as dead weight for the
        # whole run without changing a single number. Caught by
        # diagnose_model's no-gradient check; keep them sharing one list.
        return {k: data[k].edge_index for k in EDGE_TYPES}

    def _gnn(self, h_s, h_t, h_sec, data):
        ei = self._edge_index(data)
        ea = {("train", "conflicts", "train"):
              data["train", "conflicts", "train"].edge_attr}

        for i, (conv, ln) in enumerate(zip(self.convs, self.lns)):
            x = conv({"station": h_s, "train": h_t, "section": h_sec}, ei, ea)
            last = (i == len(self.convs) - 1)
            def step(new, old, key):
                z = F.relu(ln[key](new))
                return (z if last else self.drop(z)) + old
            h_s = step(x["station"], h_s, "s")
            h_t = step(x["train"], h_t, "t")
            h_sec = step(x["section"], h_sec, "sec")
        return h_s, h_t, h_sec

    def forward(self, data, update_memory: bool = True,
                ablate_history: bool = False,
                ablate_network: bool = False) -> dict:
        """`ablate_*` zero one input path at a time so the diagnostics can
        measure how much of the prediction each is actually carrying."""
        hist = self.history_encoder(data["train"].history,
                                    data["train"].history_len)
        if ablate_history:
            hist = torch.zeros_like(hist)

        h_s, h_t, h_sec = self._encode(data, hist)
        h_s4, h_t4, h_sec4 = self._gnn(h_s, h_t, h_sec, data)

        h_pred = h_t4
        if ablate_network:
            h_pred = torch.zeros_like(h_pred)
        joint = torch.cat([h_pred, hist], dim=-1)

        out = {
            # Predicted CHANGE in delay, in minutes, at the next stop.
            "next_arr": self.next_arr_head(joint).squeeze(-1),
            "next_dep": self.next_dep_head(joint).squeeze(-1),
            # minutes until this train leaves the platform it is on now.
            # softplus keeps it non-negative with a live gradient everywhere.
            "remaining_dwell": F.softplus(
                self.remaining_dwell_head(joint).squeeze(-1)),
            "h_train": h_t4,
            "h_history": hist,
            "h_station": h_s4,
            "h_section": h_sec4,
        }
        if self.hurdle:
            # Hurdle pair: "is this a large change" and "how large, signed".
            out["spike_logit"] = self.spike_logit_head(joint).squeeze(-1)
            out["spike_mag"] = self.spike_mag_head(joint).squeeze(-1)

        if update_memory:
            srv = data["train"].service_idx
            self.station_memory = self.station_memory_updater(h_s4, self.station_memory)
            self.section_memory = self.section_memory_updater(h_sec4, self.section_memory)
            if srv.numel() > 0:
                new_srv = self.service_memory.clone()
                new_srv[srv] = self.service_memory_updater(
                    h_t4, self.service_memory[srv])
                self.service_memory = new_srv
        return out


# Weights an OLD checkpoint may carry that the CURRENT model no longer has.
# Retiring a head must never make historical checkpoints unscoreable — that
# comparability is the only reason the release directories are worth keeping.
RETIRED_PREFIXES = ("station_critical_head",)

# Weights a NEW model has that pre-v13 checkpoints cannot supply. Loading an
# older model leaves this head at init; the rollout refuses to use it unless
# it was actually trained, so an untrained head can never silently drive a
# forecast.
DWELL_PREFIXES = ("remaining_dwell_head",)

# Weights the CURRENT model may have that an OLD checkpoint does not. These
# stay at initialisation and the caller is told, so it can refuse to use them.
OPTIONAL_PREFIXES = ("spike_logit_head", "spike_mag_head")


def check_feature_dims(ck_config: dict, ds) -> None:
    """Fail loudly when a checkpoint and the current dataset disagree.

    `BASE_TRAIN_FEAT_DIM` in dataset_cris.py decides how many dims
    `materialise` emits, and it has moved between versions (36 for v2-v4, 38
    for v5-v7 and v9-v10, 44 for v8). Score a checkpoint against the wrong
    value and torch raises a shape error from four frames inside a Linear,
    which reads like a bug in the model rather than a mismatched constant.

    Worse is the near miss: if two versions ever happen to agree on the TOTAL
    while differing on WHICH dims, nothing raises at all and the run silently
    scores a model on features it never saw. This check is the only thing
    standing between that and a published number.
    """
    want = {"train": (ck_config["train_in_dim"], ds.train_feat_dim),
            "station": (ck_config["station_in_dim"], ds.station_feat_dim),
            "section": (ck_config["section_in_dim"], ds.section_feat_dim),
            "history": (ck_config["hist_in_dim"], ds.hist_feat_dim)}
    bad = {k: v for k, v in want.items() if v[0] != v[1]}
    if not bad:
        return
    lines = [f"    {k:8} checkpoint {a}  dataset {b}" for k, (a, b) in bad.items()]
    raise SystemExit(
        "checkpoint was trained on a different feature set:\n"
        + "\n".join(lines)
        + "\n  Set BASE_TRAIN_FEAT_DIM in cris_pipeline/dataset_cris.py to "
          f"{ck_config['train_in_dim'] - 8} to score this checkpoint, or use a "
          "checkpoint trained on the current features.")


def load_weights(model: nn.Module, state_dict: dict,
                 verbose: bool = True) -> bool:
    """Load a checkpoint across model versions.

    Returns (hurdle_ok, dwell_ok) -- whether each optional head was actually
    present in the checkpoint. Raises on any genuine
    mismatch — a silently half-loaded model scores as a broken model, which is
    the one failure mode that looks like a research result.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    retired = [k for k in unexpected if k.startswith(RETIRED_PREFIXES)]
    # v7/v8 were trained WITH the hurdle heads but their checkpoints predate
    # the `hurdle` config key, so the model rebuilt from that config has no
    # heads to receive them. Dropping the weights is exactly equivalent to
    # scoring at --hurdle-tau 1.0, which is how they were measured anyway.
    orphan_hurdle = [k for k in unexpected if k.startswith(OPTIONAL_PREFIXES)]
    unexpected = [k for k in unexpected
                  if not k.startswith(RETIRED_PREFIXES + OPTIONAL_PREFIXES)]

    hurdle_ok = not any(k.startswith(OPTIONAL_PREFIXES) for k in missing)
    dwell_ok = not any(k.startswith(DWELL_PREFIXES) for k in missing)
    missing = [k for k in missing
               if not k.startswith(OPTIONAL_PREFIXES + DWELL_PREFIXES)]

    if missing or unexpected:
        raise SystemExit("checkpoint does not match the model: "
                         f"missing={missing} unexpected={unexpected}")
    if verbose:
        if retired:
            print(f"  checkpoint carries {len(retired)} retired tensors "
                  f"({RETIRED_PREFIXES[0]}, never trained) — discarded")
        if orphan_hurdle:
            print(f"  checkpoint's hurdle heads discarded (config predates the "
                  f"flag) — equivalent to hurdle-tau 1.0")
    return hurdle_ok, dwell_ok


# Huber is quadratic within delta and LINEAR beyond, so delta caps the gradient
# on large errors. At 20 that capped it on exactly the events we need most: a
# missed spike (actual +40, predicted +5) has error 35, and squared error would
# push 3.5x harder than Huber does there. Choosing 20 to tame outliers quietly
# re-created the median bias that switching away from pinball loss was meant to
# remove. 60 keeps essentially the whole realistic delta range (h1 |delta| p99
# is ~48 min) in the quadratic, mean-seeking region while still bounding the
# rare 120+ min event.
HUBER_DELTA = 60.0


def regression_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor, weight: torch.Tensor | None = None,
                    delta: float = HUBER_DELTA) -> torch.Tensor:
    """Huber loss in LINEAR minutes on the change in delay.

    pred [N], target [N], mask [N] in {0,1}.

    Replaces the pinball loss. Pinball at q=0.5 is L1, which fits the MEDIAN;
    per-leg delay change is right-skewed, so the median under-runs the mean and
    an autoregressive rollout built on it flattens out (measured: -20 min bias
    at the 2-4 h horizon, forecasts frozen at +300 while truth reached +396).

    Huber is quadratic within `delta` and linear beyond, so it fits the MEAN
    over the normal operating range — which is what has to be right for errors
    that accumulate hop after hop — while a single 400-minute outlier still
    cannot dominate the gradient the way plain MSE would let it.
    """
    if mask.sum() < 1:
        return pred.sum() * 0.0

    diff = pred - target
    a = diff.abs()
    loss = torch.where(a <= delta, 0.5 * diff * diff, delta * (a - 0.5 * delta))

    w = mask if weight is None else mask * weight
    return (loss * w).sum() / w.sum().clamp(min=1.0)
