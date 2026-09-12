"""
Quantum-inspired solver for the Gate Assignment QUBO.

Uses simulated quantum annealing (neal.SimulatedAnnealingSampler), which
models the same energy-minimization physics as a real quantum annealer
(e.g. D-Wave) but runs on classical hardware. This is the standard,
honest framing for a "quantum-inspired" system: it borrows the annealing
algorithm and energy-landscape formulation from quantum annealing theory,
without requiring physical quantum hardware.

Swapping this module for `dwave.system.DWaveSampler` later (if real QPU
access is available) requires no change to the QUBO formulation — that
portability is itself part of the system's value proposition.
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional

import dimod
import neal


@dataclass
class SolverResult:
    assignment: Dict[str, str]
    energy: float
    solve_time_s: float
    num_reads: int
    method: str


class QuantumInspiredSolver:
    def __init__(self, num_reads: int = 200, num_sweeps: int = 1000, seed: Optional[int] = None):
        self.sampler = neal.SimulatedAnnealingSampler()
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.seed = seed

    def solve(self, bqm: dimod.BinaryQuadraticModel, decode_fn) -> SolverResult:
        start = time.perf_counter()
        sampleset = self.sampler.sample(
            bqm,
            num_reads=self.num_reads,
            num_sweeps=self.num_sweeps,
            seed=self.seed,
        )
        elapsed = time.perf_counter() - start

        best = sampleset.first
        assignment = decode_fn(best.sample)

        return SolverResult(
            assignment=assignment,
            energy=best.energy,
            solve_time_s=elapsed,
            num_reads=self.num_reads,
            method="simulated_quantum_annealing",
        )
