"""
city_graph.py
CityMind — Shared City Graph
"""

from enum import Enum
import math


class NodeType(Enum):
    RESIDENTIAL   = "Residential"
    HOSPITAL      = "Hospital"
    SCHOOL        = "School"
    INDUSTRIAL    = "Industrial"
    POWER_PLANT   = "Power Plant"
    AMBULANCE_DEPOT = "Ambulance Depot"


class CrimeRisk(Enum):
    HIGH   = "High"
    MEDIUM = "Medium"
    LOW    = "Low"


CRIME_MULTIPLIERS = {
    CrimeRisk.HIGH:   1.5,
    CrimeRisk.MEDIUM: 1.2,
    CrimeRisk.LOW:    1.0,
}

COST_STANDARD    = 1.0
COST_RESIDENTIAL = 0.8  



class Node:
    def __init__(self, row: int, col: int):
        self.node_id            = (row, col)
        self.location_type      = None          
        self.population_density = 0.0
        self.risk_index         = 0.0
        self.accessibility      = True
        self.crime_risk_level   = CrimeRisk.LOW
        self.police_officers    = 0             

    @property
    def crime_multiplier(self) -> float:
        return CRIME_MULTIPLIERS[self.crime_risk_level]

    def __repr__(self):
        lt = self.location_type.value if self.location_type else "Unassigned"
        return f"Node{self.node_id}[{lt}]"



class Edge:

    def __init__(self, node_a: tuple, node_b: tuple, base_cost: float = COST_STANDARD):
        self.node_a       = node_a
        self.node_b       = node_b
        self.base_cost    = base_cost
        self.is_blocked   = False
        self._effective_cost = base_cost   

    @property
    def effective_cost(self) -> float:
        return self._effective_cost

    def update_effective_cost(self, node_a_obj: Node, node_b_obj: Node):
        multiplier = max(node_a_obj.crime_multiplier, node_b_obj.crime_multiplier)
        self._effective_cost = self.base_cost * multiplier

    def other(self, node_id: tuple) -> tuple:
        return self.node_b if node_id == self.node_a else self.node_a

    def __repr__(self):
        status = "BLOCKED" if self.is_blocked else f"cost={self._effective_cost:.2f}"
        return f"Edge({self.node_a}↔{self.node_b} [{status}])"



class CityGraph:

    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.nodes: dict[tuple, Node] = {}     
        self.edges: dict[tuple, Edge] = {}     
        self.adjacency: dict[tuple, list] = {} 

        self._build_grid()


    def _build_grid(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                node = Node(r, c)
                self.nodes[(r, c)] = node
                self.adjacency[(r, c)] = []

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if c + 1 < self.grid_size:  
                    self._add_edge((r, c), (r, c + 1))
                if r + 1 < self.grid_size:  
                    self._add_edge((r, c), (r + 1, c))

    def _add_edge(self, a: tuple, b: tuple):
        key = frozenset({a, b})
        edge = Edge(a, b, base_cost=COST_STANDARD)
        self.edges[key] = edge
        self.adjacency[a].append(b)
        self.adjacency[b].append(a)


    def get_node(self, row: int, col: int) -> Node:
        return self.nodes[(row, col)]

    def get_edge(self, a: tuple, b: tuple) -> Edge | None:
        return self.edges.get(frozenset({a, b}))

    def neighbors(self, node_id: tuple, passable_only: bool = True) -> list[tuple]:
        result = []
        for nb in self.adjacency[node_id]:
            if passable_only:
                edge = self.get_edge(node_id, nb)
                if edge and edge.is_blocked:
                    continue
                if not self.nodes[nb].accessibility:
                    continue
            result.append(nb)
        return result

    def travel_cost(self, a: tuple, b: tuple) -> float | None:
        edge = self.get_edge(a, b)
        if edge is None or edge.is_blocked:
            return None
        return edge.effective_cost


    def block_road(self, a: tuple, b: tuple):
        edge = self.get_edge(a, b)
        if edge:
            edge.is_blocked = True
            print(f"[GRAPH] Road blocked: {a} ↔ {b}")

    def unblock_road(self, a: tuple, b: tuple):
        edge = self.get_edge(a, b)
        if edge:
            edge.is_blocked = False
            print(f"[GRAPH] Road unblocked: {a} ↔ {b}")


    def update_crime_risk(self, node_id: tuple, risk: CrimeRisk):
        node = self.nodes[node_id]
        node.crime_risk_level = risk

        for nb_id in self.adjacency[node_id]:
            edge = self.get_edge(node_id, nb_id)
            if edge:
                edge.update_effective_cost(node, self.nodes[nb_id])

    def apply_all_crime_risks(self, risk_map: dict[tuple, CrimeRisk]):
        for node_id, risk in risk_map.items():
            self.nodes[node_id].crime_risk_level = risk

        for key, edge in self.edges.items():
            a, b = edge.node_a, edge.node_b
            edge.update_effective_cost(self.nodes[a], self.nodes[b])

        print(f"[GRAPH] Crime risk applied to {len(risk_map)} nodes. All edge costs updated.")


    def set_node_type(self, node_id: tuple, location_type: NodeType,
                      population_density: float = 0.0):
        node = self.nodes[node_id]
        node.location_type = location_type
        node.population_density = population_density

        if location_type == NodeType.RESIDENTIAL:
            for nb_id in self.adjacency[node_id]:
                edge = self.get_edge(node_id, nb_id)
                if edge:
                    edge.base_cost = COST_RESIDENTIAL
                    edge.update_effective_cost(node, self.nodes[nb_id])


    def get_nodes_by_type(self, location_type: NodeType) -> list[Node]:
        return [n for n in self.nodes.values() if n.location_type == location_type]

    def euclidean_distance(self, a: tuple, b: tuple) -> float:
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def manhattan_distance(self, a: tuple, b: tuple) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def hop_distance(self, start: tuple, end: tuple) -> int | None:
        from collections import deque
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            current, depth = queue.popleft()
            if current == end:
                return depth
            for nb in self.adjacency[current]:  
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, depth + 1))
        return None


    def stats(self) -> dict:
        total_edges   = len(self.edges)
        blocked       = sum(1 for e in self.edges.values() if e.is_blocked)
        assigned      = sum(1 for n in self.nodes.values() if n.location_type is not None)
        return {
            "grid_size":      self.grid_size,
            "total_nodes":    len(self.nodes),
            "assigned_nodes": assigned,
            "total_edges":    total_edges,
            "blocked_edges":  blocked,
            "active_edges":   total_edges - blocked,
        }

    def __repr__(self):
        s = self.stats()
        return (f"CityGraph({s['grid_size']}x{s['grid_size']}) | "
                f"Nodes: {s['assigned_nodes']}/{s['total_nodes']} assigned | "
                f"Edges: {s['active_edges']} active, {s['blocked_edges']} blocked")


if __name__ == "__main__":
    g = CityGraph(grid_size=5)

    g.set_node_type((0, 0), NodeType.HOSPITAL, population_density=50)
    g.set_node_type((0, 1), NodeType.RESIDENTIAL, population_density=200)
    g.set_node_type((2, 2), NodeType.INDUSTRIAL, population_density=80)
    g.set_node_type((4, 4), NodeType.AMBULANCE_DEPOT, population_density=10)

    print(g)
    print()

    risk_map = {
        (2, 2): CrimeRisk.HIGH,
        (0, 1): CrimeRisk.MEDIUM,
        (0, 0): CrimeRisk.LOW,
    }
    g.apply_all_crime_risks(risk_map)

    edge = g.get_edge((0, 0), (0, 1))
    print(f"Edge (0,0)↔(0,1) effective cost: {edge.effective_cost:.2f}  (expect 0.96 = 0.8 × 1.2)")

    edge2 = g.get_edge((2, 2), (2, 3))
    print(f"Edge (2,2)↔(2,3) effective cost: {edge2.effective_cost:.2f}  (expect 1.50 = 1.0 × 1.5)")

    g.block_road((0, 0), (0, 1))
    print(f"Travel cost (0,0)→(0,1) after block: {g.travel_cost((0,0),(0,1))}")  
    print(f"Neighbors of (0,0) passable only: {g.neighbors((0,0))}")

    hops = g.hop_distance((0, 0), (4, 4))
    print(f"Hop distance (0,0) to (4,4): {hops}")  

    print()
    print(g.stats())