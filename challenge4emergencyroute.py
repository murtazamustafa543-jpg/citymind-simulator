"""
CityMind — Challenge 4: Emergency Routing (A* + Dynamic Replanning)
Author    : Hadiya Sajjad (24i-0552)

"""

import math
import heapq
from collections import defaultdict

from city_graph import CityGraph, NodeType


def euclidean_heuristic(node: tuple, goal: tuple) -> float:
    
    return math.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)


def astar(
    start      : tuple,       
    goal       : tuple,       
    city_graph : CityGraph,
) -> list:
    
    open_set = []
    heapq.heappush(open_set, (0.0, 0.0, start))

    came_from = {}
    g_score   = defaultdict(lambda: math.inf)
    g_score[start] = 0.0

    closed = set()

    while open_set:
        f, g, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if current in closed:
            continue
        closed.add(current)

        for neighbour in city_graph.neighbors(current, passable_only=True):
            if neighbour in closed:
                continue
            cost = city_graph.travel_cost(current, neighbour)
            if cost is None:
                continue          

            tentative_g = g + cost
            if tentative_g < g_score[neighbour]:
                g_score[neighbour]   = tentative_g
                came_from[neighbour] = current
                h = euclidean_heuristic(neighbour, goal)
                heapq.heappush(open_set, (tentative_g + h, tentative_g, neighbour))

    return []   


class EventLog:
    def __init__(self):
        self.entries = []

    def log(self, step: int, level: str, message: str):
        color  = {"INFO": "blue", "WARNING": "orange", "CRITICAL": "red"}.get(level, "blue")
        symbol = {"INFO": "ℹ",   "WARNING": "⚠",      "CRITICAL": "✖"}.get(level, "ℹ")
        entry  = {"step": step, "level": level, "color": color, "message": message}
        self.entries.append(entry)
        print(f"  [Step {step:>2}] {symbol} [{level:<8}] {message}")

    def dump(self):
        print("\n" + "=" * 60)
        print("EVENT LOG SUMMARY")
        print("=" * 60)
        for e in self.entries:
            print(f"  Step {e['step']:>2} | {e['level']:<8} | {e['message']}")



class EmergencyRouter:
   

    NO_PATH_STRIKE_LIMIT = 1   

    def __init__(
        self,
        city_graph    : CityGraph,
        start_node    : tuple,           
        civilian_nodes: list,            
        log           : EventLog,
    ):
        self.graph           = city_graph   
        self.team_position   = start_node
        self.log             = log
        self.civilians_saved = []
        self.current_path    = []
        self.step            = 0

        self.deferred: list  = []
        self._no_path_strikes = 0         

        self.civilian_queue = self._order_civilians(start_node, civilian_nodes)
        log.log(0, "INFO",
                f"Team start: {start_node} | "
                f"Civilians (nearest-first): {self.civilian_queue}")


    def _dist(self, a: tuple, b: tuple) -> float:
        return euclidean_heuristic(a, b)

    def _order_civilians(self, from_node: tuple, civilians: list) -> list:
        remaining = civilians[:]
        ordered   = []
        current   = from_node
        while remaining:
            nearest = min(remaining, key=lambda c: self._dist(current, c))
            ordered.append(nearest)
            remaining.remove(nearest)
            current = nearest
        return ordered

    def _defer_current(self):
        skipped = self.civilian_queue.pop(0)
        self.deferred.append(skipped)
        self._no_path_strikes = 0
        self.current_path     = []
        self.log.log(self.step, "WARNING",
                     f"Civilian {skipped} DEFERRED — no path after "
                     f"{self.NO_PATH_STRIKE_LIMIT} attempts. "
                     f"Will retry later. Deferred: {self.deferred}")

    def _maybe_restore_deferred(self):
        if self.deferred:
            self.log.log(self.step, "INFO",
                         f"Retrying {len(self.deferred)} deferred civilian(s): "
                         f"{self.deferred}")
            self.civilian_queue = self._order_civilians(
                self.team_position, self.deferred
            )
            self.deferred = []
            self.plan_to_next()


    def plan_to_next(self) -> bool:
        if not self.civilian_queue:
            return False

        target = self.civilian_queue[0]
        path   = astar(self.team_position, target, self.graph)

        if not path:
            self._no_path_strikes += 1
            self.log.log(self.step, "CRITICAL",
                         f"NO PATH to civilian {target} "
                         f"(strike {self._no_path_strikes}/{self.NO_PATH_STRIKE_LIMIT})")
            self.current_path = []

            if self._no_path_strikes >= self.NO_PATH_STRIKE_LIMIT:
                self._defer_current()
                if self.civilian_queue:
                    self.plan_to_next()
                else:
                    self._maybe_restore_deferred()
            return False

        self._no_path_strikes = 0
        self.current_path = path[1:]  

        cost = 0.0
        for i in range(len(path) - 1):
            c = self.graph.travel_cost(path[i], path[i + 1])
            if c is not None:
                cost += c

        self.log.log(self.step, "INFO",
                     f"A* planned: {self.team_position} → {target} | "
                     f"path={path} | cost={cost:.3f}")
        return True

    def replan(self):
       
        self._no_path_strikes = 0
        target = self.civilian_queue[0] if self.civilian_queue else "—"
        self.log.log(self.step, "WARNING",
                     f"REPLANNING from {self.team_position} → civilian {target}")
        self.plan_to_next()


    def step_team(self) -> str:
        if not self.civilian_queue and not self.deferred:
            return "done"

        if not self.civilian_queue:
            self._maybe_restore_deferred()
            if not self.civilian_queue:
                return "done"   

        if not self.current_path:
            if not self.plan_to_next():
                if not self.current_path:
                    return "waiting"   

        next_node = self.current_path[0]

        cost = self.graph.travel_cost(self.team_position, next_node)
        if cost is None:
            self.log.log(self.step, "WARNING",
                         f"Next edge {self.team_position}──{next_node} blocked mid-move")
            self.replan()
            return "waiting"

        self.team_position = next_node
        self.current_path  = self.current_path[1:]
        self.log.log(self.step, "INFO",
                     f"Team moved to {self.team_position}")

        if self.team_position == self.civilian_queue[0]:
            rescued = self.civilian_queue.pop(0)
            self.civilians_saved.append(rescued)
            self._no_path_strikes = 0
            self.log.log(self.step, "INFO",
                         f"✓ Civilian RESCUED at {rescued} | "
                         f"remaining={self.civilian_queue} | "
                         f"deferred={self.deferred}")
            if self.civilian_queue:
                self.civilian_queue = self._order_civilians(
                    self.team_position, self.civilian_queue
                )
                self.plan_to_next()
            elif self.deferred:
                self._maybe_restore_deferred()
            return "rescued"

        return "moving"



def run_simulation(
    city_graph    : CityGraph,
    start_node    : tuple,             
    civilian_nodes: list,              
    flood_events  : list,              
    max_steps     : int = 20,
) -> dict:
   
    log    = EventLog()
    router = EmergencyRouter(city_graph, start_node, civilian_nodes, log)

    router.plan_to_next()

    flood_map = defaultdict(list)
    for step, a, b in flood_events:
        flood_map[step].append((a, b))

    print("\n[Simulation Running...]\n")

    for step in range(1, max_steps + 1):
        router.step = step

        replanned = False
        for a, b in flood_map.get(step, []):
            edge = city_graph.get_edge(a, b)
            if edge and not edge.is_blocked:
                city_graph.block_road(a, b)
                log.log(step, "CRITICAL", f"Road BLOCKED: {a} ↔ {b} [flooded]")
                if not replanned:
                    router.replan()
                    replanned = True

        status = router.step_team()

        if status == "done":
            log.log(step, "INFO", "✓ ALL civilians rescued — mission complete")
            break

    return {
        "civilians_saved"  : router.civilians_saved,
        "civilians_missed" : router.civilian_queue,
        "team_final_node"  : router.team_position,
        "total_steps"      : step,
        "log"              : log.entries,
    }


if __name__ == "__main__":
    from city_graph import CityGraph, NodeType
    from challenge1csp import CityLayoutCSP

    print("=" * 60)
    print("CityMind — Challenge 4: Emergency Routing (A*)")
    print("=" * 60)

    GRID = 6
    graph = CityGraph(grid_size=GRID)
    csp   = CityLayoutCSP(graph, seed=42)
    csp.solve()

    print(f"\n{graph}\n")

    hospitals   = graph.get_nodes_by_type(NodeType.HOSPITAL)
    residentials = graph.get_nodes_by_type(NodeType.RESIDENTIAL)

    if not hospitals:
        raise RuntimeError("No hospital found — re-run with a different seed.")

    start_node     = hospitals[0].node_id
    civilian_nodes = [n.node_id for n in residentials[:4]]   

    all_edge_keys = list(graph.edges.keys())
    flood_edge_1  = tuple(sorted(all_edge_keys[5]))    
    flood_edge_2  = tuple(sorted(all_edge_keys[15]))

    flood_events = [
        (3, flood_edge_1[0], flood_edge_1[1]),
        (7, flood_edge_2[0], flood_edge_2[1]),
    ]

    print(f"Team start     : {start_node}")
    print(f"Civilian nodes : {civilian_nodes}")
    print(f"Flood events   : step 3 → {flood_edge_1}  |  step 7 → {flood_edge_2}")

    result = run_simulation(
        city_graph     = graph,
        start_node     = start_node,
        civilian_nodes = civilian_nodes,
        flood_events   = flood_events,
        max_steps      = 20,
    )

    print("\n" + "=" * 60)
    print("EVENT LOG SUMMARY")
    print("=" * 60)
    for e in result["log"]:
        print(f"  Step {e['step']:>2} | {e['level']:<8} | {e['message']}")

    print(f"\n{'─'*60}")
    print(f"[Mission Summary]")
    print(f"  Civilians rescued : {result['civilians_saved']}")
    print(f"  Civilians missed  : {result['civilians_missed']}")
    print(f"  Team final node   : {result['team_final_node']}")
    print(f"  Steps taken       : {result['total_steps']}")
    print("=" * 60)