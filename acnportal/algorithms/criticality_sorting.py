# coding=utf-8
"""
Dependency-criticality ordering for sorting-based scheduling algorithms.

The score follows the Centrality-Entropy Index (CEI):

    score_i = alpha * C_i + beta * H_i + gamma * R_i

    C_i: composite centrality of the session's EVSE in the graph that links each
        EVSE to the infrastructure constraints it appears in (0.35 PageRank,
        0.30 betweenness, 0.20 degree, 0.15 closeness), divided by the largest
        value over the EVSEs.
    H_i: normalized Shannon entropy of the EVSE's recent utilization (last
        applied pilot over the EVSE's maximum pilot), over a sliding window.
    R_i: an operator-declared criticality in [0, 1] for the session, for
        example 1 for a fleet or emergency vehicle and 0 otherwise.

Sessions with higher scores are placed first. Pass an instance as the sort_fn of
SortedSchedulingAlgo or RoundRobin.

Reference: P. Pokharel, Centrality-Entropy Index (CEI) for governance-aware
resource allocation, Future Technologies Conference (FTC) 2026.
"""
from collections import deque
from typing import Deque, Dict, List, Mapping, Optional, Tuple

import numpy as np

from acnportal.acnsim.interface import SessionInfo, InfrastructureInfo, Interface

try:
    import networkx as nx
except ImportError:  # pragma: no cover - exercised only without networkx
    nx = None

CRITICALITY_CLASSES: Tuple[Tuple[str, float], ...] = (
    ("critical", 0.75),
    ("elevated", 0.50),
    ("moderate", 0.25),
)


def criticality_class(score: float) -> str:
    """ Return the class name of a dependency-criticality score.

    Args:
        score (float): Score in [0, 1].

    Returns:
        str: "critical", "elevated", "moderate" or "low".
    """
    for name, threshold in CRITICALITY_CLASSES:
        if score >= threshold:
            return name
    return "low"


def station_centrality(infrastructure: InfrastructureInfo) -> Dict[str, float]:
    """ Composite centrality of each EVSE in the EVSE-constraint graph.

    Each EVSE and each constraint is a node. An EVSE is linked to every
    constraint in which its coefficient is non-zero.

    Args:
        infrastructure (InfrastructureInfo): Description of the electrical
            infrastructure.

    Returns:
        Dict[str, float]: Centrality of each station in [0, 1], divided by the
            largest value over the stations.
    """
    if nx is None:
        raise ImportError(
            "networkx is required for the centrality term. Please install it "
            "using\n\tpip install networkx"
        )
    graph = nx.Graph()
    stations = [("station", s) for s in infrastructure.station_ids]
    graph.add_nodes_from(stations)
    matrix = np.asarray(infrastructure.constraint_matrix)
    for j, constraint_id in enumerate(infrastructure.constraint_ids):
        for i, station in enumerate(stations):
            if abs(matrix[j, i]) > 1e-12:
                graph.add_edge(station, ("constraint", constraint_id))
    try:
        pagerank = nx.pagerank(graph, alpha=0.85)
    except nx.PowerIterationFailedConvergence:
        pagerank = {n: 1 / graph.number_of_nodes() for n in graph}
    betweenness = nx.betweenness_centrality(graph)
    degree = nx.degree_centrality(graph)
    closeness = nx.closeness_centrality(graph)
    composite = {
        s: 0.35 * pagerank[s]
        + 0.30 * betweenness[s]
        + 0.20 * degree[s]
        + 0.15 * closeness[s]
        for s in stations
    }
    top = max(composite.values()) if composite else 0.0
    return {
        s[1]: (round(v / top, 12) if top > 0 else 1.0) for s, v in composite.items()
    }


def utilization_entropy(history: List[float], bins: int = 10) -> float:
    """ Normalized Shannon entropy of utilization values in [0, 1].

    Args:
        history (List[float]): Utilization samples.
        bins (int): Number of equal-width bins on [0, 1].

    Returns:
        float: Entropy divided by log2(bins), in [0, 1]. 0 for an empty history.
    """
    if len(history) == 0:
        return 0.0
    values = np.clip(np.asarray(history, dtype=float), 0.0, 1.0)
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    p = counts[counts > 0] / counts.sum()
    return float(min(1.0, -(p * np.log2(p)).sum() / np.log2(bins)))


class DependencyCriticalitySort:
    """ Sort function that orders sessions by dependency-criticality score.

    Args:
        alpha (float): Weight of the centrality term.
        beta (float): Weight of the utilization-entropy term.
        gamma (float): Weight of the declared-criticality term.
        criticality (Mapping[str, float]): Declared criticality in [0, 1] by
            session_id. Sessions not listed get 0.
        history_length (int): Number of periods of utilization kept per EVSE.
        bins (int): Number of bins for the entropy term.
    """

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.35,
        gamma: float = 0.25,
        criticality: Optional[Mapping[str, float]] = None,
        history_length: int = 12,
        bins: int = 10,
    ) -> None:
        if min(alpha, beta, gamma) < 0:
            raise ValueError("Weights alpha, beta and gamma must be non-negative.")
        if history_length < 1:
            raise ValueError("history_length must be at least 1.")
        if alpha > 0 and nx is None:
            raise ImportError(
                "networkx is required when alpha > 0. Please install it using\n"
                "\tpip install networkx"
            )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.criticality = dict(criticality) if criticality is not None else {}
        self.history_length = history_length
        self.bins = bins
        self._history: Dict[str, Deque[float]] = {}
        self._last_recorded: Optional[int] = None
        self._centrality: Optional[Dict[str, float]] = None
        self._centrality_key: Optional[Tuple[str, ...]] = None

    def _station_centrality(self, infrastructure: InfrastructureInfo) -> Dict[str, float]:
        key = tuple(infrastructure.station_ids) + tuple(infrastructure.constraint_ids)
        if self._centrality is None or key != self._centrality_key:
            self._centrality = station_centrality(infrastructure)
            self._centrality_key = key
        return self._centrality

    def _record_utilization(self, iface: Interface) -> None:
        """ Append last period's utilization of every EVSE, once per time step. """
        if self.beta == 0 or iface.current_time == self._last_recorded:
            return
        try:
            pilots = iface.last_applied_pilot_signals
        except NotImplementedError:
            return
        infrastructure = iface.infrastructure_info()
        session_station = {s.session_id: s.station_id for s in iface.active_sessions()}
        utilization = {station_id: 0.0 for station_id in infrastructure.station_ids}
        for session_id, pilot in pilots.items():
            station_id = session_station.get(session_id)
            if station_id is not None:
                max_pilot = iface.max_pilot_signal(station_id)
                utilization[station_id] = pilot / max_pilot if max_pilot > 0 else 0.0
        for station_id, value in utilization.items():
            self._history.setdefault(
                station_id, deque(maxlen=self.history_length)
            ).append(value)
        self._last_recorded = iface.current_time

    def scores(self, evs: List[SessionInfo], iface: Interface) -> Dict[str, float]:
        """ Return the dependency-criticality score of each session.

        Args:
            evs (List[SessionInfo]): Sessions to score.
            iface (Interface): Interface object.

        Returns:
            Dict[str, float]: Score by session_id.
        """
        centrality: Dict[str, float] = {}
        if self.alpha > 0:
            centrality = self._station_centrality(iface.infrastructure_info())
        result = {}
        for ev in evs:
            c = centrality.get(ev.station_id, 0.0)
            h = 0.0
            if self.beta > 0:
                h = utilization_entropy(
                    list(self._history.get(ev.station_id, [])), self.bins
                )
            r = float(self.criticality.get(ev.session_id, 0.0))
            result[ev.session_id] = self.alpha * c + self.beta * h + self.gamma * r
        return result

    def __call__(self, evs: List[SessionInfo], iface: Interface) -> List[SessionInfo]:
        """ Sort sessions by score in decreasing order.

        Scores within 1e-9 of each other are treated as equal. Ties are broken by
        arrival time, then session_id, so the order does not depend on floating
        point rounding.

        Args:
            evs (List[SessionInfo]): List of sessions to be sorted.
            iface (Interface): Interface object.

        Returns:
            List[SessionInfo]: Sessions sorted by decreasing score.
        """
        self._record_utilization(iface)
        scores = self.scores(evs, iface)
        return sorted(
            evs,
            key=lambda ev: (-round(scores[ev.session_id], 9), ev.arrival, ev.session_id),
        )
