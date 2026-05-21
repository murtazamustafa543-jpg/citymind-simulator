# CityMind — AI-Powered Smart City Simulator

A multi-algorithm city planning and emergency management simulation built in Python. CityMind models a 10×10 grid city and applies five distinct AI/optimization challenges in sequence, all sharing a live graph — so decisions made in one challenge directly affect the next.

---

## Features at a Glance

| Challenge | Problem | Algorithm |
|-----------|---------|-----------|
| C1 — City Layout | Place hospitals, schools, industrial zones, etc. with zoning rules | CSP + Backtracking + Forward Checking + MRV |
| C2 — Road Network | Find the minimum-cost connected road network with redundancy | Genetic Algorithm + Kruskal's MST + Max-Flow |
| C3 — Ambulance Placement | Optimally position 3 ambulances to minimise worst-case response time | Simulated Annealing + Dijkstra |
| C4 — Emergency Routing | Route a rescue team to civilians under dynamic flood events | A\* + Dynamic Replanning |
| C5 — Crime Risk Prediction | Predict and visualise node-level crime risk; deploy police | K-Means Clustering + KNN Classification |

A Pygame dashboard (`ui.py`) ties all five challenges together in a real-time animated interface.

---

## Project Structure

```
citymind/
├── city_graph.py               # Shared graph: nodes, edges, crime risk, road costs
├── challenge1csp.py            # C1: CSP city layout solver
├── challenge2road.py           # C2: Genetic algorithm road optimiser
├── challenge3ambulance.py      # C3: Simulated annealing ambulance placement
├── challenge4emergencyroute.py # C4: A* emergency router with dynamic replanning
├── challenge5crime.py          # C5: Crime prediction pipeline (K-Means + KNN)
└── ui.py                       # Pygame dashboard
```

---

## How It Works

### Shared City Graph (`city_graph.py`)
All challenges operate on a single `CityGraph` instance — a 10×10 grid of `Node` objects connected by `Edge` objects. Node types, road states (blocked/active), and crime risk multipliers all live here and propagate instantly across challenges.

### C1 — City Layout Planning
A CSP solver places special node types (Hospitals, Schools, Industrial zones, Power Plants, Ambulance Depots) while enforcing hard constraints:
- **C1**: Industrial zones cannot be adjacent to Schools or Hospitals.
- **C2** *(soft)*: Residential cells should be within 3 hops of a hospital.
- **C3**: Power plants must be within 2 hops of an industrial zone.

Uses MRV heuristic, forward checking, backtracking, and a smart hospital pre-seeding strategy. Falls back to a minimum-violation layout if the grid is too constrained.

### C2 — Road Network Optimisation
A Genetic Algorithm evolves a subset of city roads that:
- Keeps all nodes connected (verified with Union-Find).
- Minimises total road construction cost.
- Guarantees at least 2 independent paths between the primary Hospital and Ambulance Depot (verified with Ford-Fulkerson max-flow).

The MST (Kruskal's algorithm) seeds the initial population to give the GA a strong starting point.

### C3 — Ambulance Placement
Simulated Annealing places 3 ambulances across accessible nodes to minimise the **worst-case distance** from any populated residential node to its nearest ambulance. Re-runs automatically when crime risk weights change (after C5).

### C4 — Emergency Routing
An A\* planner routes a rescue team from the hospital to multiple civilian locations. Flood events dynamically block roads mid-simulation, triggering live replanning. Civilians that become temporarily unreachable are deferred and retried later.

### C5 — Crime Risk Prediction
A two-stage ML pipeline predicts crime risk for every node:
1. **K-Means** clusters nodes by population density, industrial proximity, and zone type.
2. **KNN** (k=5, distance-weighted) trains on synthetic labelled data and predicts `LOW`, `MEDIUM`, or `HIGH` risk for each node.

Risk levels update edge travel costs (×1.0 / ×1.2 / ×1.5), which feeds back into C3 and C4.

---

## Dashboard (UI)

Run `ui.py` to launch the Pygame dashboard. All five challenges execute in sequence at startup, then the simulation steps forward in real time.

**Controls:**

| Key / Action | Effect |
|---|---|
| `Space` | Pause / Resume |
| `← →` | Decrease / Increase simulation speed |
| `R` | Reset and re-run full pipeline |
| `1` / `2` / `3` / `4` | Toggle overlays: Roads · Coverage · Crime · Agents |
| Mouse wheel | Scroll the event log |

**Overlays:**
- **Roads** — green = fast, red = blocked.
- **Coverage** — green = close to an ambulance, red = far.
- **Crime** — amber tint = Medium risk, red tint = High risk; blue badge = police count.
- **Agents** — `M` = rescue team, `+` = ambulances, `C` = civilians.

---

## Requirements

```
python >= 3.10
pygame
numpy
scikit-learn
```

Install dependencies:
```bash
pip install pygame numpy scikit-learn
```

---

## Running

### Full dashboard
```bash
python ui.py
```

### Individual challenges (standalone)
```bash
python challenge1csp.py
python challenge2road.py
python challenge3ambulance.py
python challenge4emergencyroute.py
python challenge5crime.py
```

Each file has a self-contained `__main__` block that builds a test city graph and prints results to the terminal.

---

## Algorithm Details

### Genetic Algorithm (C2)
- Population: 60 chromosomes (each a bitmask over all candidate edges)
- Selection: Tournament (size 5)
- Crossover: Two-point (rate 0.80)
- Mutation: Bit-flip (rate 0.001 per bit)
- Elitism: Top 4 preserved each generation
- Fitness: `total_cost + penalty × max(0, 2 − max_flow)`

### Simulated Annealing (C3)
- Initial temperature: 100.0, cooling rate: 0.995
- 3,000 iterations per restart, 3 restarts
- Neighbour moves: swap one ambulance to an adjacent or random accessible node

### A\* (C4)
- Heuristic: Euclidean distance to goal
- Edge cost: `edge.effective_cost` (base cost × crime multiplier)
- On road block: immediate replan from current position

### Crime Pipeline (C5)
- Features: population density, industrial proximity, zone type flags, proximity to safe zones
- K-Means: k=3, 10 initialisations
- KNN: k=5, distance-weighted, Euclidean metric
- 10% label noise injected into synthetic training set for robustness

---
University project — AI/Algorithms course.
