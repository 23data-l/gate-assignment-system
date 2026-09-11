import time
from src.data.airport_data import build_daily_schedule, build_gate_config
from src.optimization.qubo_formulation import GateAssignmentQUBO
from src.optimization.quantum_solver import QuantumInspiredSolver
from src.optimization.classical_baseline import ClassicalILPSolver


def compute_qubo_objective(assignment, flights, gates, size_mismatch_penalty=20.0, walk_cost_weight=1.0):
    """Evaluate the true (unpenalized) cost of an assignment for fair comparison."""
    total = 0.0
    for fid, gid in assignment.items():
        flight, gate = flights[fid], gates[gid]
        if gate.size_class >= flight.size_class:
            total += size_mismatch_penalty * 0.1 * (gate.size_class - flight.size_class)
        else:
            total += 1e6  # infeasible, shouldn't happen
        total += walk_cost_weight * gate.walk_cost_to_hub * (flight.passengers / 100.0)
    return total


def run_benchmark(num_flights=40, num_gates=12, num_reads=500):
    flights = build_daily_schedule(num_flights=num_flights)
    gates = build_gate_config(num_gates=num_gates)

    print(f"Scenario: {len(flights)} real BLR flights, {len(gates)} gates\n")

    # --- Quantum-inspired ---
    qubo = GateAssignmentQUBO(flights=flights, gates=gates)
    bqm = qubo.build()
    q_solver = QuantumInspiredSolver(num_reads=num_reads, seed=42)
    q_result = q_solver.solve(bqm, qubo.decode)
    q_cost = compute_qubo_objective(q_result.assignment, flights, gates) if len(q_result.assignment) == len(flights) else float("inf")

    # --- Classical baseline ---
    c_solver = ClassicalILPSolver(flights=flights, gates=gates, time_limit_s=15.0)
    c_result = c_solver.solve()

    print("=" * 60)
    print(f"{'Metric':<30}{'Quantum-Inspired':<18}{'Classical ILP':<15}")
    print("=" * 60)
    print(f"{'Flights assigned':<30}{len(q_result.assignment)}/{len(flights):<17}{len(c_result.assignment)}/{len(flights)}")
    print(f"{'Solve time (s)':<30}{q_result.solve_time_s:<18.4f}{c_result.solve_time_s:<15.4f}")
    print(f"{'Objective (lower=better)':<30}{q_cost:<18.2f}{c_result.objective:<15.2f}")
    print(f"{'Status':<30}{'complete' if len(q_result.assignment)==len(flights) else 'incomplete':<18}{c_result.status:<15}")
    print("=" * 60)

    if q_cost != float("inf") and c_result.objective > 0:
        gap = (q_cost - c_result.objective) / c_result.objective * 100
        print(f"\nQuantum-inspired solution is {gap:+.1f}% vs optimal classical ILP objective")
        print(f"Quantum-inspired solve time is {q_result.solve_time_s / c_result.solve_time_s:.2f}x classical solve time")

    return q_result, c_result


if __name__ == "__main__":
    run_benchmark()
