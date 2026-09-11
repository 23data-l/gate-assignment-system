"""
Classical baseline solver for the Gate Assignment Problem, using
Google OR-Tools CP-SAT (a state-of-the-art classical constraint
programming solver). This gives an honest, strong classical
comparison point for the QUBO/simulated-annealing solver — exactly
what a paper's evaluation section and a patent's "advantages over
prior art" section both need.
"""

import time
from dataclasses import dataclass
from typing import Dict

from ortools.sat.python import cp_model

from src.optimization.qubo_formulation import Flight, Gate


@dataclass
class ClassicalResult:
    assignment: Dict[str, str]
    objective: float
    solve_time_s: float
    method: str
    status: str


class ClassicalILPSolver:
    """
    Same objective and constraints as the QUBO formulation, expressed
    natively as a CP-SAT model (no penalty terms needed — CP-SAT
    supports hard constraints directly).
    """

    def __init__(self, flights: Dict[str, Flight], gates: Dict[str, Gate],
                 size_mismatch_penalty: float = 20.0, walk_cost_weight: float = 1.0,
                 buffer_min: float = 15.0, time_limit_s: float = 10.0):
        self.flights = flights
        self.gates = gates
        self.size_mismatch_penalty = size_mismatch_penalty
        self.walk_cost_weight = walk_cost_weight
        self.buffer_min = buffer_min
        self.time_limit_s = time_limit_s

    def _overlaps(self, f1: Flight, f2: Flight) -> bool:
        return not (f1.departure + self.buffer_min <= f2.arrival or
                    f2.departure + self.buffer_min <= f1.arrival)

    def solve(self) -> ClassicalResult:
        model = cp_model.CpModel()
        flight_ids = list(self.flights.keys())
        gate_ids = list(self.gates.keys())

        x = {}
        for fid in flight_ids:
            for gid in gate_ids:
                x[fid, gid] = model.NewBoolVar(f"x_{fid}_{gid}")

        # Hard: each flight exactly one gate, only among size-feasible gates
        for fid in flight_ids:
            flight = self.flights[fid]
            feasible_vars = []
            for gid in gate_ids:
                gate = self.gates[gid]
                if gate.size_class < flight.size_class:
                    model.Add(x[fid, gid] == 0)  # infeasible, forbid
                else:
                    feasible_vars.append(x[fid, gid])
            model.Add(sum(x[fid, gid] for gid in gate_ids) == 1)
            if not feasible_vars:
                raise RuntimeError(f"No feasible gate for flight {fid} (size_class={flight.size_class})")

        # Hard: no two overlapping flights share a gate
        for gid in gate_ids:
            for i, f1 in enumerate(flight_ids):
                for f2 in flight_ids[i + 1:]:
                    if self._overlaps(self.flights[f1], self.flights[f2]):
                        model.Add(x[f1, gid] + x[f2, gid] <= 1)

        # Objective: minimize size-mismatch waste + passenger-weighted walk cost
        # (scaled to integers for CP-SAT)
        SCALE = 100
        terms = []
        for fid in flight_ids:
            flight = self.flights[fid]
            for gid in gate_ids:
                gate = self.gates[gid]
                cost = 0.0
                if gate.size_class >= flight.size_class:
                    cost += self.size_mismatch_penalty * 0.1 * (gate.size_class - flight.size_class)
                cost += self.walk_cost_weight * gate.walk_cost_to_hub * (flight.passengers / 100.0)
                terms.append(int(round(cost * SCALE)) * x[fid, gid])
        model.Minimize(sum(terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_s
        start = time.perf_counter()
        status = solver.Solve(model)
        elapsed = time.perf_counter() - start

        assignment = {}
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for fid in flight_ids:
                for gid in gate_ids:
                    if solver.Value(x[fid, gid]) == 1:
                        assignment[fid] = gid

        return ClassicalResult(
            assignment=assignment,
            objective=solver.ObjectiveValue() / SCALE if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else float("inf"),
            solve_time_s=elapsed,
            method="classical_cp_sat_ilp",
            status=solver.StatusName(status),
        )
