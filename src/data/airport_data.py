"""
Builds a realistic daily flight schedule for a chosen airport (default:
BLR - Kempegowda International, Bangalore) from real OpenFlights route
data, and pairs it with a synthetic-but-plausible gate configuration.

Why synthetic gates: airports do not publish machine-readable gate
layouts (bay counts, sizes, terminal adjacency) as open data — this is
operationally sensitive information. So we use real, verifiable flight
demand (which airlines/routes actually serve this airport, at what
volume) and layer a gate configuration on top that is sized and
structured to match the airport's known public facts (terminal count,
approximate contact-gate count), which is standard practice in academic
gate-assignment literature.

BLR facts used (public, from AAI/BIAL sources): 2 passenger terminals,
T1 (older) and T2 (newer, opened 2022); roughly 40+ contact stands
combined. We model a representative subset for a tractable demo.
"""

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from src.optimization.qubo_formulation import Flight, Gate

WIDE_BODY_EQUIPMENT = {"777", "787", "747", "330", "340", "350", "380", "767", "L15"}


def _load_routes_for_airport(airport_code: str, routes_file: Path) -> List[dict]:
    routes = []
    with open(routes_file, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 9:
                continue
            airline, _, src, _, dst, _, _, stops, equipment = row[:9]
            if src == airport_code or dst == airport_code:
                routes.append({
                    "airline": airline,
                    "src": src,
                    "dst": dst,
                    "equipment": equipment.split()[0] if equipment else "",
                })
    return routes


def _size_class_for_equipment(equipment: str) -> int:
    for code in WIDE_BODY_EQUIPMENT:
        if code in equipment:
            return 2  # wide-body / large
    if equipment in ("CR2", "CRJ", "E90", "AT7", "DH4", "SF3"):
        return 0  # regional / small
    return 1  # default: narrow-body / medium (most common: A320/737 family)


def build_daily_schedule(
    airport_code: str = "BLR",
    num_flights: int = 40,
    day_length_min: int = 20 * 60,   # model a 20-hour operating day
    seed: int = 7,
) -> Dict[str, Flight]:
    """Sample real routes serving this airport and lay them out across a day."""
    rng = random.Random(seed)
    data_dir = Path(__file__).parent.parent.parent / "data"
    routes = _load_routes_for_airport(airport_code, data_dir / "routes.dat")

    if not routes:
        raise RuntimeError(f"No routes found for airport code {airport_code}")

    sampled = rng.sample(routes, k=min(num_flights, len(routes)))

    flights: Dict[str, Flight] = {}
    for i, route in enumerate(sampled):
        arrival = rng.uniform(0, day_length_min - 90)
        turnaround = rng.uniform(35, 75)  # minutes at gate (typical narrow-body turnaround)
        departure = arrival + turnaround
        size_class = _size_class_for_equipment(route["equipment"])
        passengers = {0: rng.randint(50, 90), 1: rng.randint(120, 189), 2: rng.randint(220, 380)}[size_class]

        flight_id = f"{route['airline']}{100+i}_{route['src']}-{route['dst']}"
        flights[flight_id] = Flight(
            flight_id=flight_id,
            arrival=round(arrival, 1),
            departure=round(departure, 1),
            size_class=size_class,
            passengers=passengers,
        )

    return flights


def build_gate_config(num_gates: int = 12) -> Dict[str, Gate]:
    """
    Representative gate configuration modeled on BLR's two-terminal
    structure: a mix of contact stands across T1 and T2, with a range
    of size classes and walking-distance-to-hub values reflecting
    typical pier layouts (near-hub gates cheaper to reach than
    far-pier gates).
    """
    rng = random.Random(3)
    gates: Dict[str, Gate] = {}
    terminals = ["T1", "T2"]
    for i in range(num_gates):
        terminal = terminals[i % 2]
        # Size distribution: mostly medium (narrow-body), some large (wide-body capable), a few small
        size_class = rng.choices([0, 1, 2], weights=[0.15, 0.65, 0.20])[0]
        # Gates further down the pier have higher walk cost
        pier_position = i // 2
        walk_cost = 1.0 + pier_position * 0.4 + rng.uniform(0, 0.3)
        gate_id = f"{terminal}-G{i+1}"
        gates[gate_id] = Gate(
            gate_id=gate_id,
            size_class=size_class,
            terminal=terminal,
            walk_cost_to_hub=round(walk_cost, 2),
        )
    return gates


if __name__ == "__main__":
    flights = build_daily_schedule()
    gates = build_gate_config()
    print(f"Loaded {len(flights)} flights (from real BLR routes) and {len(gates)} gates\n")
    print("Sample flights:")
    for fid, f in list(flights.items())[:5]:
        print(f"  {fid}: arr={f.arrival:.0f}min dep={f.departure:.0f}min size={f.size_class} pax={f.passengers}")
    print("\nGate config:")
    for gid, g in list(gates.items())[:5]:
        print(f"  {gid}: terminal={g.terminal} size={g.size_class} walk_cost={g.walk_cost_to_hub}")

    size_dist = {}
    for f in flights.values():
        size_dist[f.size_class] = size_dist.get(f.size_class, 0) + 1
    print(f"\nFlight size distribution: {size_dist}")
