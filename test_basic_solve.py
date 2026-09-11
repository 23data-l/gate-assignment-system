import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimization.qubo_formulation import GateAssignmentQUBO, Flight, Gate
from src.optimization.quantum_solver import QuantumInspiredSolver


def make_toy_scenario():
    flights = {
        "AI101": Flight("AI101", arrival=0,   departure=60,  size_class=1, passengers=180),
        "AI202": Flight("AI202", arrival=30,  departure=90,  size_class=0, passengers=120),
        "AI303": Flight("AI303", arrival=100, departure=160, size_class=2, passengers=300),
        "AI404": Flight("AI404", arrival=45,  departure=105, size_class=1, passengers=150),
    }
    gates = {
        "G1": Gate("G1", size_class=2, terminal="A", walk_cost_to_hub=1.0),
        "G2": Gate("G2", size_class=1, terminal="A", walk_cost_to_hub=1.5),
        "G3": Gate("G3", size_class=1, terminal="B", walk_cost_to_hub=3.0),
    }
    return flights, gates


def check_feasible(assignment, flights, gates):
    # each flight assigned
    assert len(assignment) == len(flights), f"Not all flights assigned: {assignment}"
    # size feasibility
    for fid, gid in assignment.items():
        assert gates[gid].size_class >= flights[fid].size_class, \
            f"{fid} (size {flights[fid].size_class}) assigned to too-small gate {gid}"
    # no overlap conflicts on same gate
    by_gate = {}
    for fid, gid in assignment.items():
        by_gate.setdefault(gid, []).append(fid)
    for gid, fids in by_gate.items():
        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                f1, f2 = flights[fids[i]], flights[fids[j]]
                buffer = 15.0
                overlap = not (f1.departure + buffer <= f2.arrival or f2.departure + buffer <= f1.arrival)
                assert not overlap, f"Conflict on gate {gid}: {fids[i]} & {fids[j]}"
    print("All hard constraints satisfied.")


if __name__ == "__main__":
    flights, gates = make_toy_scenario()
    qubo = GateAssignmentQUBO(flights=flights, gates=gates)
    bqm = qubo.build()
    print(f"QUBO built: {len(bqm.variables)} variables, {len(bqm.quadratic)} quadratic terms")

    solver = QuantumInspiredSolver(num_reads=500, seed=42)
    result = solver.solve(bqm, qubo.decode)

    print(f"\nMethod: {result.method}")
    print(f"Solve time: {result.solve_time_s:.4f}s over {result.num_reads} reads")
    print(f"Best energy: {result.energy:.2f}")
    print("Assignment:")
    for fid, gid in result.assignment.items():
        print(f"  {fid} -> {gid}")

    check_feasible(result.assignment, flights, gates)
