"""
QUBO (Quadratic Unconstrained Binary Optimization) formulation of the
Airport Gate Assignment Problem (AGAP).

Decision variable:
    x[f, g] = 1  if flight f is assigned to gate g, else 0

Objective (to MINIMIZE):
    1. Passenger connection / walking cost      (soft, weighted)
    2. Gate-size mismatch penalty                (soft, weighted)
    3. Constraint: each flight gets exactly one gate      (hard, penalty)
    4. Constraint: no two overlapping flights share a gate (hard, penalty)

All constraints are folded into the QUBO as quadratic penalty terms,
since QUBO solvers (quantum annealers and their classical simulators)
only accept unconstrained quadratic objectives.
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Tuple

import dimod


@dataclass
class Flight:
    flight_id: str
    arrival: float          # minutes since midnight (or epoch minutes)
    departure: float        # when the flight vacates the gate
    size_class: int         # 0=small (narrow-body), 1=medium, 2=large (wide-body)
    passengers: int = 150


@dataclass
class Gate:
    gate_id: str
    size_class: int          # max size_class of aircraft it can host
    terminal: str = "A"
    walk_cost_to_hub: float = 1.0   # relative walking distance to baggage/hub


@dataclass
class GateAssignmentQUBO:
    flights: Dict[str, Flight]
    gates: Dict[str, Gate]

    # Penalty weights — hard constraints must dominate soft ones.
    HARD_PENALTY: float = 100.0
    SIZE_MISMATCH_PENALTY: float = 20.0
    WALK_COST_WEIGHT: float = 1.0

    def _var(self, flight_id: str, gate_id: str) -> str:
        return f"x_{flight_id}__{gate_id}"

    def _overlaps(self, f1: Flight, f2: Flight) -> bool:
        # Two flights conflict if their gate-occupancy windows overlap,
        # including a buffer for turnaround/pushback.
        buffer = 15.0  # minutes
        return not (f1.departure + buffer <= f2.arrival or
                    f2.departure + buffer <= f1.arrival)

    def build(self) -> dimod.BinaryQuadraticModel:
        bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)

        flight_ids = list(self.flights.keys())
        gate_ids = list(self.gates.keys())

        # ---- Soft cost terms: size mismatch + walking distance ----
        for fid in flight_ids:
            flight = self.flights[fid]
            for gid in gate_ids:
                gate = self.gates[gid]
                var = self._var(fid, gid)

                cost = 0.0
                # Penalize assigning a large aircraft to a small gate
                # (infeasible) or a small aircraft to a much larger gate
                # (wasted capacity).
                if gate.size_class < flight.size_class:
                    cost += self.SIZE_MISMATCH_PENALTY * 5  # infeasible, heavy penalty
                else:
                    cost += self.SIZE_MISMATCH_PENALTY * 0.1 * (gate.size_class - flight.size_class)

                # Passenger-weighted walking cost
                cost += self.WALK_COST_WEIGHT * gate.walk_cost_to_hub * (flight.passengers / 100.0)

                bqm.add_linear(var, cost)

        # ---- Hard constraint 1: each flight assigned exactly one gate ----
        # Penalty: HARD_PENALTY * (sum_g x[f,g] - 1)^2
        for fid in flight_ids:
            vars_for_flight = [self._var(fid, gid) for gid in gate_ids]
            # Expand (sum x_i - 1)^2 = sum x_i^2 + 2*sum_{i<j} x_i x_j - 2*sum x_i + 1
            # x_i^2 = x_i for binary variables.
            for v in vars_for_flight:
                bqm.add_linear(v, self.HARD_PENALTY * (1 - 2))  # x_i term: +1 -2 = -1 coefficient * HARD_PENALTY
            for v1, v2 in combinations(vars_for_flight, 2):
                bqm.add_quadratic(v1, v2, self.HARD_PENALTY * 2)
            bqm.offset += self.HARD_PENALTY * 1

        # ---- Hard constraint 2: no two overlapping flights share a gate ----
        for gid in gate_ids:
            for f1, f2 in combinations(flight_ids, 2):
                if self._overlaps(self.flights[f1], self.flights[f2]):
                    v1 = self._var(f1, gid)
                    v2 = self._var(f2, gid)
                    bqm.add_quadratic(v1, v2, self.HARD_PENALTY * 2)

        return bqm

    def decode(self, sample: Dict[str, int]) -> Dict[str, str]:
        """Convert a QUBO solution sample back into {flight_id: gate_id}."""
        assignment = {}
        for fid in self.flights:
            for gid in self.gates:
                var = self._var(fid, gid)
                if sample.get(var, 0) == 1:
                    assignment[fid] = gid
        return assignment
