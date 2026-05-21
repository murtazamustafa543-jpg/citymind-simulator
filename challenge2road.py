"""
CityMind — Challenge 2: Road Network Optimization
"""

import random
import heapq
from collections import defaultdict, deque
from typing import Optional


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def _bfs_path(graph: dict, source: int, sink: int, parent: dict) -> bool:
    visited = {source}
    queue   = deque([source])
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in visited and graph[u][v] > 0:
                visited.add(v)
                parent[v] = u
                if v == sink:
                    return True
                queue.append(v)
    return False


def max_flow(nodes: list, active_edges: list, source: int, sink: int) -> int:
    graph = defaultdict(lambda: defaultdict(int))
    for u, v in active_edges:
        graph[u][v] += 1
        graph[v][u] += 1

    flow = 0
    while True:
        parent = {}
        if not _bfs_path(graph, source, sink, parent):
            break
        path_flow = float('inf')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, graph[u][v])
            v = u
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = u
        flow += path_flow
    return flow


def kruskal_mst(num_nodes: int, all_edges: list) -> list:
    uf          = UnionFind(num_nodes)
    mst_indices = []
    for idx, (cost, u, v) in enumerate(all_edges):
        if uf.union(u, v):
            mst_indices.append(idx)
            if len(mst_indices) == num_nodes - 1:
                break
    return mst_indices



class RoadNetworkGA:
    
    PENALTY          = 500   
    POPULATION_SIZE  = 60
    GENERATIONS      = 150
    CROSSOVER_RATE   = 0.80
    MUTATION_RATE    = 0.001   
    ELITE_SIZE       = 4       
    TOURNAMENT_SIZE  = 5

    def __init__(
        self,
        num_nodes  : int,
        all_edges  : list,          
        hospital_id: int,
        depot_id   : int,
        seed       : int = 42,
    ):
        self.num_nodes   = num_nodes
        self.all_edges   = all_edges   
        self.n_edges     = len(all_edges)
        self.hospital_id = hospital_id
        self.depot_id    = depot_id
        random.seed(seed)


        self.mst_indices = set(kruskal_mst(num_nodes, all_edges))
        self.mst_chrom   = [1 if i in self.mst_indices else 0
                            for i in range(self.n_edges)]


    def _decode(self, chrom: list) -> list:
        return [(self.all_edges[i][1], self.all_edges[i][2])
                for i, bit in enumerate(chrom) if bit == 1]

    def _is_connected(self, chrom: list) -> bool:
        active = self._decode(chrom)
        if not active:
            return False
        adj = defaultdict(set)
        for u, v in active:
            adj[u].add(v)
            adj[v].add(u)
        visited = set()
        stack   = [0]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adj[node] - visited)
        return len(visited) == self.num_nodes

    def _fitness(self, chrom: list) -> float:
        active      = self._decode(chrom)
        total_cost  = sum(self.all_edges[i][0]
                          for i, bit in enumerate(chrom) if bit == 1)

        if not self._is_connected(chrom):
            return total_cost + self.PENALTY * self.num_nodes

        flow        = max_flow(list(range(self.num_nodes)),
                               active,
                               self.hospital_id,
                               self.depot_id)
        flow_penalty = self.PENALTY * max(0, 2 - flow)

        return total_cost + flow_penalty


    def _random_chrom(self) -> list:
        chrom = [1] * self.n_edges  
        for i in range(self.n_edges):
            if random.random() < 0.1:  
                chrom[i] = 0
        return chrom

    def _init_population(self) -> list:
        pop = [self.mst_chrom[:]]                          
        for _ in range(min(10, self.POPULATION_SIZE - 1)):
            c = self.mst_chrom[:]
            extra = random.randint(0, self.n_edges - 1)
            c[extra] = 1
            pop.append(c)
        while len(pop) < self.POPULATION_SIZE:
            pop.append(self._random_chrom())
        return pop


    def _tournament(self, pop: list, fitnesses: list) -> list:
        contestants = random.sample(range(len(pop)), self.TOURNAMENT_SIZE)
        winner      = min(contestants, key=lambda i: fitnesses[i])
        return pop[winner][:]


    def _crossover(self, p1: list, p2: list):
        if random.random() > self.CROSSOVER_RATE:
            return p1[:], p2[:]
        pt1, pt2 = sorted(random.sample(range(1, self.n_edges), 2))
        c1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
        c2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]
        return c1, c2


    def _mutate(self, chrom: list) -> list:
        return [bit ^ 1 if random.random() < self.MUTATION_RATE else bit
                for bit in chrom]


    def run(self, verbose: bool = True) -> dict:
        pop       = self._init_population()
        fitnesses = [self._fitness(c) for c in pop]

        best_chrom   = min(pop, key=lambda c: self._fitness(c))
        best_fitness = self._fitness(best_chrom)

        for gen in range(self.GENERATIONS):
            elite_idx = sorted(range(len(pop)), key=lambda i: fitnesses[i])[:self.ELITE_SIZE]
            new_pop   = [pop[i][:] for i in elite_idx]

            while len(new_pop) < self.POPULATION_SIZE:
                p1 = self._tournament(pop, fitnesses)
                p2 = self._tournament(pop, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.POPULATION_SIZE:
                    new_pop.append(self._mutate(c2))

            pop       = new_pop
            fitnesses = [self._fitness(c) for c in pop]

            gen_best     = min(pop, key=lambda c: self._fitness(c))
            gen_best_fit = self._fitness(gen_best)
            if gen_best_fit < best_fitness:
                best_fitness = gen_best_fit
                best_chrom   = gen_best[:]

            if verbose and (gen + 1) % 25 == 0:
                flow = max_flow(
                    list(range(self.num_nodes)),
                    self._decode(best_chrom),
                    self.hospital_id,
                    self.depot_id,
                )
                active_count = sum(best_chrom)
                total_cost   = sum(self.all_edges[i][0]
                                   for i, b in enumerate(best_chrom) if b == 1)
                print(f"  Gen {gen+1:>4} | fitness={best_fitness:.2f} | "
                      f"edges={active_count} | cost={total_cost:.2f} | "
                      f"flow={flow}")

        active_edges = self._decode(best_chrom)
        total_cost   = sum(self.all_edges[i][0]
                           for i, b in enumerate(best_chrom) if b == 1)
        final_flow   = max_flow(
            list(range(self.num_nodes)),
            active_edges,
            self.hospital_id,
            self.depot_id,
        )
        connected    = self._is_connected(best_chrom)

        return {
            "chromosome"   : best_chrom,
            "active_edges" : active_edges,   
            "total_cost"   : total_cost,
            "flow"         : final_flow,
            "connected"    : connected,
            "fitness"      : best_fitness,
        }


def apply_road_network(city_graph, result: dict) -> None:
    
    size = city_graph.grid_size

    def int_to_tuple(n: int) -> tuple:
        return (n // size, n % size)


    active_set = set()
    for u, v in result["active_edges"]:
        active_set.add(frozenset([int_to_tuple(u), int_to_tuple(v)]))

    for key, edge in city_graph.edges.items():
        fs = frozenset([edge.node_a, edge.node_b])
        if fs in active_set:
            edge.is_blocked = False   
        else:
            edge.is_blocked = True    



if __name__ == "__main__":
    import math

    print("=" * 60)
    print("CityMind — Challenge 2: Road Network Optimization (GA)")
    print("=" * 60)

    NUM_NODES   = 10
    HOSPITAL_ID = 0
    DEPOT_ID    = 9

    positions = {
        0: (0, 0), 
        1: (1, 0),
        2: (2, 0),
        3: (0, 1),
        4: (1, 1),
        5: (2, 1),
        6: (0, 2),
        7: (1, 2),
        8: (2, 2),
        9: (3, 2),   
    }

    all_edges = []
    for u in range(NUM_NODES):
        for v in range(u + 1, NUM_NODES):
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dist <= 1.5:
                cost = round(dist, 3)
                all_edges.append((cost, u, v))

    print(f"\nCity: {NUM_NODES} nodes | {len(all_edges)} candidate edges")
    print(f"Hospital node: {HOSPITAL_ID}  |  Depot node: {DEPOT_ID}\n")

    mst_idx   = kruskal_mst(NUM_NODES, all_edges)
    mst_cost  = sum(all_edges[i][0] for i in mst_idx)
    mst_edges = [(all_edges[i][1], all_edges[i][2]) for i in mst_idx]
    mst_flow  = max_flow(list(range(NUM_NODES)), mst_edges, HOSPITAL_ID, DEPOT_ID)

    print(f"[Kruskal MST]  edges={len(mst_idx)}  cost={mst_cost:.3f}  "
          f"flow(H→D)={mst_flow}  {'✓ two-path OK' if mst_flow >= 2 else '✗ two-path MISSING'}")

    print("\n[GA Running...]\n")
    ga     = RoadNetworkGA(NUM_NODES, all_edges, HOSPITAL_ID, DEPOT_ID, seed=42)
    result = ga.run(verbose=True)

    print("\n" + "-" * 60)
    print("[GA Result]")
    print(f"  Active edges : {len(result['active_edges'])}")
    print(f"  Total cost   : {result['total_cost']:.3f}")
    print(f"  Max-flow H→D : {result['flow']}  "
          f"{'✓ two-path guaranteed' if result['flow'] >= 2 else '✗ two-path NOT met'}")
    print(f"  Connected    : {'✓ Yes' if result['connected'] else '✗ No'}")
    print(f"  Fitness      : {result['fitness']:.3f}")
    print("\n  Edge list:")
    for u, v in sorted(result["active_edges"]):
        cost = next(c for c, a, b in all_edges if (a, b) == (u, v) or (a, b) == (v, u))
        print(f"    ({u}) ──[{cost:.3f}]── ({v})")
    print("=" * 60)