from src.data.airport_data import build_daily_schedule, build_gate_config
from src.optimization.qubo_formulation import GateAssignmentQUBO
from src.optimization.quantum_solver import QuantumInspiredSolver
from src.dynamic.delay_handler import DynamicDelayHandler, DelayEvent


def test_delay_handling():
    flights = build_daily_schedule(num_flights=70, day_length_min=24 * 60)
    gates = build_gate_config(num_gates=16)

    # Establish an initial baseline assignment
    qubo = GateAssignmentQUBO(flights=flights, gates=gates)
    bqm = qubo.build()
    solver = QuantumInspiredSolver(num_reads=800, num_sweeps=1500, seed=42)
    initial = solver.solve(bqm, qubo.decode)
    print(f"Initial full solve: {initial.solve_time_s:.2f}s for {len(flights)} flights\n")

    # Pick a flight and delay it by 90 minutes
    delayed_flight_id = list(flights.keys())[10]
    original = flights[delayed_flight_id]
    event = DelayEvent(
        flight_id=delayed_flight_id,
        new_arrival=original.arrival + 90,
        new_departure=original.departure + 90,
    )
    print(f"Delaying {delayed_flight_id}: {original.arrival:.0f}-{original.departure:.0f} -> "
          f"{event.new_arrival:.0f}-{event.new_departure:.0f}\n")

    # Incremental re-optimization
    handler_inc = DynamicDelayHandler(flights, gates, initial.assignment)
    inc_result = handler_inc.handle_delay(event)
    print(f"INCREMENTAL: {inc_result.solve_time_s:.3f}s, "
          f"{len(inc_result.affected_flights)}/{len(flights)} flights touched: {inc_result.affected_flights}")

    # Full re-optimization baseline
    handler_full = DynamicDelayHandler(flights, gates, initial.assignment)
    full_result = handler_full.handle_delay_full_reopt(event)
    print(f"FULL RE-SOLVE: {full_result.solve_time_s:.3f}s, "
          f"{len(full_result.affected_flights)}/{len(flights)} flights touched")

    speedup = full_result.solve_time_s / inc_result.solve_time_s
    print(f"\nIncremental re-optimization is {speedup:.1f}x faster than full re-solve")
    print(f"Only {len(inc_result.affected_flights)/len(flights)*100:.1f}% of flights needed to be touched")


if __name__ == "__main__":
    test_delay_handling()
