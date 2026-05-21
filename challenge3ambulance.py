"""
CityMind — Challenge 3: Ambulance Placement (Simulated Annealing)
Author    : Hadia Sajjad (24i-0552)
"""

import math
import random
import heapq
from collections import defaultdict

from city_graph import CityGraph



def extract_adjacency(city_graph: CityGraph) -> dict:
    
    adj = defaultdict(list)
    for key, edge in city_graph.edges.items():
        if edge.is_blocked:
            continue
        a, b = edge.node_a, edge.node_b
        if not city_graph.nodes[a].accessibility:
            continue
        if not city_graph.nodes[b].accessibility:
            continue
        cost = edge.effective_cost
        adj[a].append((b, cost))
        adj[b].append((a, cost))
    return dict(adj)



def dijkstra(source: tuple, graph: dict) -> dict:
    
    dist = {source: 0.0}
    heap = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        for v, cost in graph.get(u, []):
            nd = d + cost
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return dist



def worst_case_response(
    ambulance_positions: list,     
    populated_nodes    : list,     
    graph              : dict,     
) -> float:
    
    if not populated_nodes:
        return 0.0

    all_dists = [dijkstra(a, graph) for a in ambulance_positions]

    max_dist = 0.0
    for r in populated_nodes:
        min_dist = min(d.get(r, math.inf) for d in all_dists)
        if min_dist > max_dist:
            max_dist = min_dist

    return max_dist



class AmbulanceSA:
    

    NUM_AMBULANCES = 3
    INITIAL_TEMP   = 100.0
    COOLING_RATE   = 0.995      
    MIN_TEMP       = 0.01
    ITERATIONS     = 3_000      
    RESTARTS       = 3          

    def __init__(
        self,
        city_graph : CityGraph,
        seed       : int = 42,
    ):
        self.city_graph = city_graph
        random.seed(seed)

        self._refresh_from_graph()


    def _refresh_from_graph(self):
        self.graph = extract_adjacency(self.city_graph)

        reachable = set(self.graph.keys())

        self.all_nodes = [
            nid for nid, node in self.city_graph.nodes.items()
            if node.accessibility and nid in reachable
        ]

        self.populated_nodes = [
            nid for nid, node in self.city_graph.nodes.items()
            if node.population_density > 0
            and node.accessibility
            and nid in reachable
        ]

        self.neighbours = defaultdict(list)
        for node, edges in self.graph.items():
            for nb, _ in edges:
                self.neighbours[node].append(nb)


    def _random_state(self) -> tuple:
        if len(self.all_nodes) < self.NUM_AMBULANCES:
            raise ValueError(
                f"Not enough accessible nodes ({len(self.all_nodes)}) "
                f"to place {self.NUM_AMBULANCES} ambulances."
            )
        return tuple(random.sample(self.all_nodes, self.NUM_AMBULANCES))

    def _neighbour_state(self, state: tuple) -> tuple:
        
        state = list(state)
        idx   = random.randrange(self.NUM_AMBULANCES)
        curr  = state[idx]
        nbs   = [n for n in self.neighbours[curr] if n not in state]

        if nbs:
            state[idx] = random.choice(nbs)
        else:
            candidates = [n for n in self.all_nodes if n not in state]
            if candidates:
                state[idx] = random.choice(candidates)

        return tuple(state)

    def _objective(self, state: tuple) -> float:
        return worst_case_response(list(state), self.populated_nodes, self.graph)


    def _run_once(self, verbose: bool) -> tuple:
        current_state = self._random_state()
        current_score = self._objective(current_state)
        best_state    = current_state
        best_score    = current_score
        temp          = self.INITIAL_TEMP

        for _ in range(self.ITERATIONS):
            if temp < self.MIN_TEMP:
                break

            candidate       = self._neighbour_state(current_state)
            candidate_score = self._objective(candidate)
            delta           = candidate_score - current_score

            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_state = candidate
                current_score = candidate_score

            if current_score < best_score:
                best_score = current_score
                best_state = current_state

            temp *= self.COOLING_RATE

        return best_state, best_score


    def run(self, verbose: bool = True) -> dict:
        
        overall_best_state = None
        overall_best_score = math.inf

        for restart in range(self.RESTARTS):
            state, score = self._run_once(verbose=verbose)
            if verbose:
                print(f"  Restart {restart + 1}/{self.RESTARTS} | "
                      f"best_score={score:.4f} | positions={state}")
            if score < overall_best_score:
                overall_best_score = score
                overall_best_state = state

        if overall_best_state is None:
            print("[C3] Warning: SA could not find a valid placement. "
                  "Falling back to first available nodes.")
            fallback = self.all_nodes[:self.NUM_AMBULANCES]
            while len(fallback) < self.NUM_AMBULANCES:
                fallback.append(fallback[0])
            overall_best_state = tuple(fallback)
            overall_best_score = self._objective(overall_best_state)

        coverage = {
            amb: dijkstra(amb, self.graph)
            for amb in overall_best_state
        }

        return {
            "positions"       : list(overall_best_state),
            "worst_case_dist" : overall_best_score,
            "coverage"        : coverage,   
        }


def rerun_placement(sa: AmbulanceSA, verbose: bool = False) -> dict:
    
    sa._refresh_from_graph()
    return sa.run(verbose=verbose)


if __name__ == "__main__":
    from city_graph import CityGraph, NodeType, CrimeRisk
    from challenge1csp import CityLayoutCSP

    print("=" * 60)
    print("CityMind — Challenge 3: Ambulance Placement (SA)")
    print("=" * 60)

    graph = CityGraph(grid_size=6)
    csp   = CityLayoutCSP(graph, seed=7)
    csp.solve()

    print(f"\n{graph}\n")

    sa     = AmbulanceSA(city_graph=graph, seed=42)
    result = sa.run(verbose=True)

    print(f"\n{'─'*60}")
    print(f"[Initial Placement Result]")
    print(f"  Ambulance positions : {result['positions']}")
    print(f"  Worst-case distance : {result['worst_case_dist']:.4f}")
    print(f"\n  Coverage (ambulance → distance to each populated node):")
    for amb, dists in result["coverage"].items():
        pop_dists = {
            r: round(dists.get(r, math.inf), 3)
            for r in sa.populated_nodes
        }
        print(f"    Amb @ {amb}: {pop_dists}")

    print(f"\n{'─'*60}")
    print("[Simulation Step 5 — crime weights updated by Challenge 5]")

    sample_node = list(graph.nodes.keys())[10]
    graph.update_crime_risk(sample_node, CrimeRisk.HIGH)
    print(f"  Node {sample_node} → HIGH crime risk (1.5x cost)")

    result2 = rerun_placement(sa, verbose=True)
    print(f"\n[Re-placement Result]")
    print(f"  Ambulance positions : {result2['positions']}")
    print(f"  Worst-case distance : {result2['worst_case_dist']:.4f}")
    print("=" * 60)