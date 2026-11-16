# coding=utf-8
"""
Tests for the dependency-criticality sort function.
"""
import unittest

import numpy as np

from acnportal.algorithms import (
    SortedSchedulingAlgo,
    DependencyCriticalitySort,
    criticality_class,
)
from acnportal.algorithms.criticality_sorting import (
    station_centrality,
    utilization_entropy,
)
from acnportal.algorithms.tests.generate_test_cases import (
    session_generator,
    three_phase_balanced_network,
)
from acnportal.algorithms.tests.testing_interface import TestingInterface
from acnportal.algorithms.utils import infrastructure_constraints_feasible


def _interface(
    num_sessions=3,
    arrivals=None,
    pod=False,
    pilots=None,
    current_time=0,
    limit=1000,
):
    network = three_phase_balanced_network(1, limit)
    if pod:
        network["constraint_matrix"] = np.vstack(
            [network["constraint_matrix"], [1, 0, 0]]
        )
        network["constraint_limits"] = np.append(network["constraint_limits"], limit)
        network["constraint_ids"] = network["constraint_ids"] + ["pod"]
    sessions = session_generator(
        num_sessions=num_sessions,
        arrivals=arrivals if arrivals is not None else [0] * num_sessions,
        departures=[12] * num_sessions,
        requested_energy=[10] * num_sessions,
        remaining_energy=[10] * num_sessions,
        max_rates=[32] * num_sessions,
    )
    data = {
        "active_sessions": sessions,
        "infrastructure_info": network,
        "current_time": current_time,
        "period": 5,
    }
    if pilots is not None:
        data["last_applied_pilot_signals"] = pilots
    return TestingInterface(data)


def _order(sort_fn, iface):
    return [s.session_id for s in sort_fn(iface.active_sessions(), iface)]


class TestCriticalityClass(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(criticality_class(0.75), "critical")
        self.assertEqual(criticality_class(0.7499), "elevated")
        self.assertEqual(criticality_class(0.50), "elevated")
        self.assertEqual(criticality_class(0.25), "moderate")
        self.assertEqual(criticality_class(0.2499), "low")


class TestUtilizationEntropy(unittest.TestCase):
    def test_empty_history(self):
        self.assertEqual(utilization_entropy([]), 0.0)

    def test_constant_history(self):
        self.assertEqual(utilization_entropy([0.5] * 12), 0.0)

    def test_uniform_history(self):
        values = [0.05 + 0.1 * k for k in range(10)]
        self.assertAlmostEqual(utilization_entropy(values), 1.0)

    def test_values_clipped(self):
        self.assertEqual(utilization_entropy([-1.0, 2.0]), utilization_entropy([0.0, 1.0]))


class TestStationCentrality(unittest.TestCase):
    def test_symmetric_network(self):
        iface = _interface()
        centrality = station_centrality(iface.infrastructure_info())
        self.assertEqual(set(centrality.values()), {1.0})

    def test_extra_constraint_raises_centrality(self):
        iface = _interface(pod=True)
        centrality = station_centrality(iface.infrastructure_info())
        self.assertEqual(centrality["0"], 1.0)
        self.assertLess(centrality["1"], 1.0)
        self.assertLess(centrality["2"], 1.0)


class TestDependencyCriticalitySort(unittest.TestCase):
    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            DependencyCriticalitySort(alpha=-0.1)

    def test_history_length_rejected(self):
        with self.assertRaises(ValueError):
            DependencyCriticalitySort(history_length=0)

    def test_declared_criticality_goes_first(self):
        sort_fn = DependencyCriticalitySort(
            alpha=0, beta=0, gamma=1, criticality={"2": 1.0}
        )
        self.assertEqual(_order(sort_fn, _interface()), ["2", "0", "1"])

    def test_ties_break_by_arrival_then_id(self):
        sort_fn = DependencyCriticalitySort(alpha=0, beta=0, gamma=1)
        iface = _interface(arrivals=[0, 0, 0])
        iface.data["active_sessions"][0]["arrival"] = 0
        iface.data["active_sessions"][1]["arrival"] = 0
        self.assertEqual(_order(sort_fn, iface), ["0", "1", "2"])
        late_first = _interface()
        late_first.data["current_time"] = 2
        late_first.data["active_sessions"][0]["arrival"] = 2
        self.assertEqual(_order(sort_fn, late_first), ["1", "2", "0"])

    def test_centrality_orders_pod_station_first(self):
        sort_fn = DependencyCriticalitySort(alpha=1, beta=0, gamma=0)
        iface = _interface(pod=True)
        self.assertEqual(_order(sort_fn, iface)[0], "0")

    def test_scores_combine_terms(self):
        sort_fn = DependencyCriticalitySort(
            alpha=0.4, beta=0.35, gamma=0.25, criticality={"1": 1.0}
        )
        iface = _interface()
        scores = sort_fn.scores(iface.active_sessions(), iface)
        self.assertAlmostEqual(scores["0"], 0.4)
        self.assertAlmostEqual(scores["1"], 0.65)

    def test_entropy_recorded_once_per_step(self):
        sort_fn = DependencyCriticalitySort(alpha=0, beta=1, gamma=0)
        iface = _interface(pilots={"0": 32.0, "1": 0.0, "2": 16.0}, current_time=1)
        sort_fn(iface.active_sessions(), iface)
        sort_fn(iface.active_sessions(), iface)
        self.assertEqual(len(sort_fn._history["0"]), 1)
        iface.data["current_time"] = 2
        iface.data["last_applied_pilot_signals"] = {"0": 0.0, "1": 0.0, "2": 16.0}
        order = _order(sort_fn, iface)
        self.assertEqual(len(sort_fn._history["0"]), 2)
        # Station 0 went from full to idle, so it has the only non-zero entropy.
        self.assertEqual(order[0], "0")

    def test_history_window(self):
        sort_fn = DependencyCriticalitySort(alpha=0, beta=1, gamma=0, history_length=2)
        iface = _interface(pilots={"0": 32.0, "1": 0.0, "2": 0.0})
        for t in range(1, 5):
            iface.data["current_time"] = t
            sort_fn(iface.active_sessions(), iface)
        self.assertEqual(len(sort_fn._history["0"]), 2)

    def test_missing_pilot_history_is_ignored(self):
        sort_fn = DependencyCriticalitySort(alpha=0, beta=1, gamma=0)
        iface = _interface()
        self.assertEqual(_order(sort_fn, iface), ["0", "1", "2"])
        self.assertEqual(sort_fn._history, {})

    def test_schedule_feasible_when_constrained(self):
        algo = SortedSchedulingAlgo(
            DependencyCriticalitySort(criticality={"1": 1.0})
        )
        iface = _interface(limit=40)
        algo.register_interface(iface)
        schedule = algo.run()
        rates = np.array([schedule[s][0] for s in ["0", "1", "2"]])
        self.assertTrue(
            infrastructure_constraints_feasible(rates, iface.infrastructure_info())
        )
        # The declared-critical session is served first and gets its full rate.
        self.assertEqual(schedule["1"][0], 32)


if __name__ == "__main__":
    unittest.main()
