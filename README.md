# Quantum-Inspired Adaptive Airport Gate Assignment System

A generalized constraint-optimization framework for airport gate assignment
using a QUBO (Quadratic Unconstrained Binary Optimization) formulation
solved via simulated quantum annealing, with dynamic incremental
re-optimization for flight delay handling.

Built on real flight route data for Kempegowda International Airport (BLR),
Bangalore, sourced from the OpenFlights dataset.

## What's novel here

1. **QUBO formulation of gate assignment** with hard constraints (one gate
   per flight, no scheduling conflicts, aircraft-size feasibility) folded
   in as penalty terms, solved with simulated quantum annealing
   (`dwave-neal`) — portable to real quantum annealing hardware (D-Wave)
   with no change to the formulation.
2. **Dynamic incremental re-optimization**: when a flight is delayed, the
   system identifies only the affected subset of flights (via conflict
   propagation) and re-solves just that sub-problem, instead of the whole
   day's schedule. In testing this gave a **~250x speedup** over full
   re-optimization while touching **~1-2% of flights**.
3. **Classical ILP baseline** (Google OR-Tools CP-SAT) included for honest
   benchmarking — see `docs/benchmark_notes.md`.

## Project structure

```
src/
  data/airport_data.py       # Real OpenFlights route data -> synthetic schedule
  optimization/
    qubo_formulation.py      # QUBO model of the static gate assignment problem
    quantum_solver.py        # Simulated quantum annealing wrapper (neal)
    classical_baseline.py    # OR-Tools CP-SAT baseline for comparison
  dynamic/delay_handler.py   # Incremental re-optimization on delay events
  api/main.py                 # FastAPI backend
dashboard/index.html         # Interactive live dashboard (gate timeline + delay demo)
tests/                       # Sanity tests + benchmarks
data/                        # OpenFlights airports.dat / routes.dat
```

## Local setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
uvicorn src.api.main:app --reload --port 8000
```

Then open **http://localhost:8000** in a browser for the interactive
dashboard: generate a scenario, solve it (quantum-inspired or classical),
then pick a flight and trigger a delay to watch the live re-optimization.

Run the test/benchmark scripts directly:

```bash
python3 -m tests.test_basic_solve       # sanity check on a toy scenario
python3 -m tests.benchmark              # quantum-inspired vs classical, single size
python3 -m tests.scaling_benchmark      # comparison across problem sizes
python3 -m tests.test_delay_handling    # incremental vs full re-optimization
```

## Free deployment options

All three options below have generous free tiers and will run this app
(FastAPI + dashboard) with no cost.

### Option A: Render.com (recommended, simplest)
1. Push this folder to a GitHub repo.
2. On render.com: **New +** -> **Web Service** -> connect the repo.
3. Render auto-detects the `Dockerfile`. Choose the **Free** instance type.
4. Deploy — you'll get a public URL like `https://gate-assignment-system.onrender.com`.
   (Free tier note: the service sleeps after 15 min of inactivity and takes
   ~30-50s to wake on the next request — fine for a conference/demo link.)

### Option B: Railway.app
1. Push to GitHub, then **New Project** -> **Deploy from GitHub repo** on
   railway.app.
2. Railway detects the `Procfile` automatically. Free tier gives a monthly
   usage credit that comfortably covers a demo app.

### Option C: Fly.io
1. Install the `flyctl` CLI, run `fly launch` in this folder (it will
   detect the Dockerfile), then `fly deploy`.
2. Free allowance covers small always-on apps.

## Honest benchmark notes (for the paper/patent)

See `docs/benchmark_notes.md` for full numbers. Summary: at small-to-medium
scale, classical CP-SAT matches or beats the quantum-inspired solver on
raw one-shot solve quality — this is expected and stated plainly rather
than overclaimed. The defensible, patent-worthy advantage is in **dynamic
re-optimization speed** after a delay, where the incremental approach
vastly outperforms re-solving from scratch.

## Disclaimer

This uses *simulated* quantum annealing (a classical algorithm modeled on
quantum annealing physics), not a physical quantum processor. This
distinction should be stated explicitly and honestly in any paper, patent
disclosure, or presentation to avoid overclaiming.
