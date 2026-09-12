"""
FastAPI backend for the Quantum-Inspired Adaptive Gate Assignment System.

Endpoints:
  GET  /api/health
  GET  /api/schedule        -> current flights, gates, and assignment
  POST /api/solve           -> run a fresh full solve (quantum-inspired)
  POST /api/solve/classical -> run the classical ILP baseline for comparison
  POST /api/delay           -> trigger a delay event, get incremental re-optimization
  POST /api/reset           -> reset to a fresh random scenario
"""

from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.data.airport_data import build_daily_schedule, build_gate_config
from src.optimization.qubo_formulation import GateAssignmentQUBO, Flight, Gate
from src.optimization.quantum_solver import QuantumInspiredSolver
from src.optimization.classical_baseline import ClassicalILPSolver
from src.dynamic.delay_handler import DynamicDelayHandler, DelayEvent

app = FastAPI(title="Quantum-Inspired Adaptive Gate Assignment System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class State:
    flights: Dict[str, Flight] = {}
    gates: Dict[str, Gate] = {}
    assignment: Dict[str, str] = {}
    last_solve_info: dict = {}


state = State()


def _flight_to_dict(f: Flight) -> dict:
    return {
        "flight_id": f.flight_id, "arrival": f.arrival, "departure": f.departure,
        "size_class": f.size_class, "passengers": f.passengers,
    }


def _gate_to_dict(g: Gate) -> dict:
    return {
        "gate_id": g.gate_id, "size_class": g.size_class,
        "terminal": g.terminal, "walk_cost_to_hub": g.walk_cost_to_hub,
    }


class DelayRequest(BaseModel):
    flight_id: str
    delay_minutes: float


class ScenarioRequest(BaseModel):
    num_flights: Optional[int] = 40
    num_gates: Optional[int] = 12
    seed: Optional[int] = 7


def _init_scenario(num_flights=40, num_gates=12, seed=7):
    state.flights = build_daily_schedule(num_flights=num_flights, day_length_min=24 * 60, seed=seed)
    state.gates = build_gate_config(num_gates=num_gates)
    state.assignment = {}
    state.last_solve_info = {}


_init_scenario()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/reset")
def reset_scenario(req: ScenarioRequest):
    _init_scenario(req.num_flights, req.num_gates, req.seed)
    return {"message": "Scenario reset", "num_flights": len(state.flights), "num_gates": len(state.gates)}


@app.get("/api/schedule")
def get_schedule():
    return {
        "flights": [_flight_to_dict(f) for f in state.flights.values()],
        "gates": [_gate_to_dict(g) for g in state.gates.values()],
        "assignment": state.assignment,
        "last_solve_info": state.last_solve_info,
    }


@app.post("/api/solve")
def solve():
    if not state.flights:
        raise HTTPException(400, "No scenario loaded")
    qubo = GateAssignmentQUBO(flights=state.flights, gates=state.gates)
    bqm = qubo.build()
    solver = QuantumInspiredSolver(num_reads=600, num_sweeps=1200, seed=42)
    result = solver.solve(bqm, qubo.decode)
    state.assignment = result.assignment
    state.last_solve_info = {
        "method": result.method, "solve_time_s": result.solve_time_s,
        "energy": result.energy, "complete": len(result.assignment) == len(state.flights),
    }
    return {"assignment": state.assignment, "info": state.last_solve_info}


@app.post("/api/solve/classical")
def solve_classical():
    if not state.flights:
        raise HTTPException(400, "No scenario loaded")
    solver = ClassicalILPSolver(flights=state.flights, gates=state.gates, time_limit_s=15.0)
    result = solver.solve()
    return {
        "assignment": result.assignment,
        "info": {"method": result.method, "solve_time_s": result.solve_time_s,
                  "objective": result.objective, "status": result.status},
    }


@app.post("/api/delay")
def trigger_delay(req: DelayRequest):
    if req.flight_id not in state.flights:
        raise HTTPException(404, f"Flight {req.flight_id} not found")
    if not state.assignment:
        raise HTTPException(400, "No initial assignment — call /api/solve first")

    flight = state.flights[req.flight_id]
    event = DelayEvent(
        flight_id=req.flight_id,
        new_arrival=flight.arrival + req.delay_minutes,
        new_departure=flight.departure + req.delay_minutes,
    )

    handler = DynamicDelayHandler(state.flights, state.gates, state.assignment)
    result = handler.handle_delay(event)

    state.flights = handler.flights
    state.assignment = result.updated_assignment

    return {
        "assignment": state.assignment,
        "affected_flights": result.affected_flights,
        "solve_time_s": result.solve_time_s,
        "method": result.method,
        "updated_flight": _flight_to_dict(state.flights[req.flight_id]),
    }


# Serve the interactive dashboard (mounted last so /api routes take priority)
app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")
