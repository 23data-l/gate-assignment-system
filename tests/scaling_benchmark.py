from src.data.airport_data import build_daily_schedule, build_gate_config
from src.optimization.qubo_formulation import GateAssignmentQUBO
from src.optimization.quantum_solver import QuantumInspiredSolver
from src.optimization.classical_baseline import ClassicalILPSolver
from tests.benchmark import compute_qubo_objective


def run_scaling_test(sizes=((20, 8), (40, 12), (70, 16), (110, 20))):
    results = []
    for num_flights, num_gates in sizes:
        flights = build_daily_schedule(num_flights=num_flights, day_length_min=24 * 60)
        gates = build_gate_config(num_gates=num_gates)

        qubo = GateAssignmentQUBO(flights=flights, gates=gates)
        bqm = qubo.build()
        q_solver = QuantumInspiredSolver(num_reads=1000, num_sweeps=1500, seed=42)
        q_result = q_solver.solve(bqm, qubo.decode)
        q_complete = len(q_result.assignment) == len(flights)
        q_cost = compute_qubo_objective(q_result.assignment, flights, gates) if q_complete else float("inf")

        c_solver = ClassicalILPSolver(flights=flights, gates=gates, time_limit_s=8.0)
        c_result = c_solver.solve()

        row = {
            "flights": num_flights, "gates": num_gates,
            "q_time": q_result.solve_time_s, "q_cost": q_cost, "q_complete": q_complete,
            "c_time": c_result.solve_time_s, "c_cost": c_result.objective, "c_status": c_result.status,
        }
        results.append(row)
        print(f"n={num_flights:4d} flights, {num_gates:3d} gates | "
              f"QI: {row['q_time']:6.2f}s cost={q_cost:7.1f} | "
              f"ILP: {row['c_time']:6.2f}s cost={c_result.objective:7.1f} status={c_result.status}")

    return results


if __name__ == "__main__":
    print("Scaling comparison: Quantum-Inspired (SA) vs Classical ILP (CP-SAT)\n")
    run_scaling_test()
