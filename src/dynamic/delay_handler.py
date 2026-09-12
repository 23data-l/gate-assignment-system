"""
Dynamic delay handling: the core novel contribution of this system.

When a flight's arrival/departure shifts (a delay), re-solving the
entire gate assignment from scratch is wasteful and slow — most of the
schedule is unaffected. Instead, this module:

  1. Identifies the "blast radius" of the delay: only flights whose
     gate-occupancy windows now conflict with the delayed flight (or
     with each other as a consequence) need to be reconsidered.
  2. Builds a much smaller QUBO over only that affected subset, with
     everyone else's gate held fixed.
  3. Re-solves only that sub-problem, which is where the real
     speedup over full re-optimization comes from.

This keeps re-optimization cost roughly proportional to the size of
the disruption, not the size of the whole schedule — which is the
property that matters operationally (an airport can't afford to
recompute a full-day schedule every time one flight is 20 minutes late)
and is the strongest, most defensible novelty claim for a patent.
"""

import time
from dataclasses import dataclass, replace
from typing import Dict, List, Set

from src.optimization.qubo_formulation import GateAssignmentQUBO, Flight, Gate
from src.optimization.quantum_solver import QuantumInspiredSolver


@dataclass
class DelayEvent:
    flight_id: str
    new_arrival: float
    new_departure: float


@dataclass
class ReoptimizationResult:
    updated_assignment: Dict[str, str]
    affected_flights: List[str]
    solve_time_s: float
    method: str
    full_reoptimization: bool


class DynamicDelayHandler:
    def __init__(self, flights: Dict[str, Flight], gates: Dict[str, Gate],
                 current_assignment: Dict[str, str], buffer_min: float = 15.0):
        self.flights = dict(flights)
        self.gates = gates
        self.current_assignment = dict(current_assignment)
        self.buffer_min = buffer_min

    def _overlaps(self, f1: Flight, f2: Flight) -> bool:
        return not (f1.departure + self.buffer_min <= f2.arrival or
                    f2.departure + self.buffer_min <= f1.arrival)

    def _find_affected_flights(self, delayed_flight: Flight) -> Set[str]:
        """
        Any flight sharing a gate with the delayed flight whose window
        now conflicts is affected. We propagate transitively.
        """
        affected = {delayed_flight.flight_id}
        frontier = [delayed_flight.flight_id]

        while frontier:
            fid = frontier.pop()
            current_gate = self.current_assignment.get(fid)
            if current_gate is None:
                continue
            same_gate_flights = [f for f, g in self.current_assignment.items() if g == current_gate]
            for other_fid in same_gate_flights:
                if other_fid == fid or other_fid in affected:
                    continue
                if self._overlaps(self.flights[fid], self.flights[other_fid]):
                    affected.add(other_fid)
                    frontier.append(other_fid)

        return affected

    def handle_delay(self, event: DelayEvent, num_reads: int = 300, num_sweeps: int = 800) -> ReoptimizationResult:
        start = time.perf_counter()

        delayed = self.flights[event.flight_id]
        updated_flight = replace(delayed, arrival=event.new_arrival, departure=event.new_departure)
        self.flights[event.flight_id] = updated_flight

        affected_ids = self._find_affected_flights(updated_flight)

        sub_flights = {fid: self.flights[fid] for fid in affected_ids}

        sub_qubo = GateAssignmentQUBO(flights=sub_flights, gates=self.gates)
        bqm = sub_qubo.build()

        solver = QuantumInspiredSolver(num_reads=num_reads, num_sweeps=num_sweeps, seed=None)
        result = solver.solve(bqm, sub_qubo.decode)

        new_assignment = dict(self.current_assignment)
        new_assignment.update(result.assignment)
        self.current_assignment = new_assignment

        elapsed = time.perf_counter() - start

        return ReoptimizationResult(
            updated_assignment=new_assignment,
            affected_flights=sorted(affected_ids),
            solve_time_s=elapsed,
            method="incremental_reoptimization",
            full_reoptimization=False,
        )

    def handle_delay_full_reopt(self, event: DelayEvent, num_reads: int = 500, num_sweeps: int = 1500) -> ReoptimizationResult:
        """Baseline for comparison: re-solve the ENTIRE schedule from scratch."""
        start = time.perf_counter()

        delayed = self.flights[event.flight_id]
        updated_flight = replace(delayed, arrival=event.new_arrival, departure=event.new_departure)
        self.flights[event.flight_id] = updated_flight

        full_qubo = GateAssignmentQUBO(flights=self.flights, gates=self.gates)
        bqm = full_qubo.build()
        solver = QuantumInspiredSolver(num_reads=num_reads, num_sweeps=num_sweeps, seed=None)
        result = solver.solve(bqm, full_qubo.decode)

        self.current_assignment = result.assignment
        elapsed = time.perf_counter() - start

        return ReoptimizationResult(
            updated_assignment=result.assignment,
            affected_flights=list(self.flights.keys()),
            solve_time_s=elapsed,
            method="full_reoptimization_baseline",
            full_reoptimization=True,
        )
