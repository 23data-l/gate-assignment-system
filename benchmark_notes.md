# Benchmark Notes

Real BLR (Kempegowda International, Bangalore) route data from OpenFlights,
paired with a synthetic-but-realistic gate configuration (see README for
why gate layouts are synthetic — this data isn't publicly released by
airports).

## 1. Static solve: Quantum-Inspired (SA) vs Classical ILP (CP-SAT)

| Flights | Gates | QI time (s) | QI cost | ILP time (s) | ILP cost | ILP status |
|---------|-------|--------------|---------|----------------|----------|------------|
| 20      | 8     | 5.12         | 46.9    | 0.01           | 35.9     | OPTIMAL    |
| 40      | 12    | 16.60        | 122.4   | 8.00           | 78.5     | FEASIBLE (capped) |
| 70      | 16    | 41.52        | 267.6   | 8.00           | 181.7    | FEASIBLE (capped) |
| 110     | 20    | 88.53        | 485.0   | 8.00           | 347.7    | FEASIBLE (capped) |

**Honest finding:** classical CP-SAT wins on both time and solution
quality at these scales, and the gap does not close as problem size
grows with this implementation. This should be reported plainly, not
worked around. It also motivates *why* the dynamic re-optimization
capability (not raw one-shot solve quality) is the paper/patent's real
contribution — see below.

## 2. Dynamic delay re-optimization: Incremental vs Full re-solve

Scenario: 70 flights, 16 gates. One flight delayed by 90 minutes.

| Method                  | Time (s) | Flights touched |
|--------------------------|----------|------------------|
| Incremental re-optimization | 0.086    | 1 / 70 (1.4%)   |
| Full re-solve from scratch  | 21.92    | 70 / 70 (100%)  |

**Speedup: ~256x.** This is the core, defensible novelty: bounding
re-optimization cost to the size of the disruption rather than the size
of the whole schedule. This is the property an operational airport
system actually needs (delays are frequent; full-day re-solves on every
delay are not viable), and it is a strong, specific technical claim for
a patent (an incremental/localized re-optimization method for QUBO-based
dynamic scheduling problems).

## Recommendations for the paper's evaluation section

- Report both results honestly. Don't claim static-solve superiority.
- Frame the contribution as: "a QUBO-based gate assignment framework
  whose dynamic re-optimization scales with disruption size, not
  schedule size" — supported by the ~256x number.
- If time permits, repeat the delay experiment across multiple flights/
  delay magnitudes/schedule densities to report a distribution, not a
  single data point (reviewers will ask for this).
