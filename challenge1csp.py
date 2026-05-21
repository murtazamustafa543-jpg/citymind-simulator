"""
challenge1csp.py
CityMind — Challenge 1: City Layout Planning
Algorithm : CSP with Backtracking + Forward Checking + MRV + Smart Seeding
Author    : Murtaza Mustafa (24i-0876)

"""

import random
import time
from collections import defaultdict, deque

try:
    from city_graph import CityGraph, NodeType, CrimeRisk
    _STANDALONE = False
except ImportError:
    _STANDALONE = True
    from enum import Enum

    class NodeType(Enum):
        RESIDENTIAL     = "Residential"
        HOSPITAL        = "Hospital"
        SCHOOL          = "School"
        INDUSTRIAL      = "Industrial"
        POWER_PLANT     = "Power Plant"
        AMBULANCE_DEPOT = "Ambulance Depot"

    class CrimeRisk(Enum):
        LOW    = "Low"
        MEDIUM = "Medium"
        HIGH   = "High"

    class _FakeNode:
        def __init__(self, nid, nt, density):
            self.node_id            = nid
            self.location_type      = nt
            self.population_density = density
            self.crime_risk_level   = CrimeRisk.LOW
        def __repr__(self):
            return f"Node{self.node_id}[{self.location_type.value}]"

    class CityGraph:
        def __init__(self, grid_size=10):
            self.grid_size = grid_size
            self._nodes    = {}
        def set_node_type(self, node_id, node_type, population_density=0):
            self._nodes[node_id] = _FakeNode(node_id, node_type, population_density)
        def get_node(self, r, c):
            return self._nodes.get((r, c))
        def __repr__(self):
            return f"CityGraph({self.grid_size}x{self.grid_size})"




CONSTRAINTS = {
    "C1_EXCLUSION_RADIUS": 1,
    "C2_HOSPITAL_HOPS": 3,
    "C3_INDUSTRIAL_HOPS": 2,
    "HOSPITAL_SEED_INSET": 1,
}



NODE_COUNTS = {
    NodeType.HOSPITAL:        (6, 8),
    NodeType.SCHOOL:          (3, 6),
    NodeType.INDUSTRIAL:      (6, 8),
    NodeType.POWER_PLANT:     (2, 3),
    NodeType.AMBULANCE_DEPOT: (2, 3),
}

SPECIAL_TYPES = [
    NodeType.HOSPITAL,
    NodeType.SCHOOL,
    NodeType.INDUSTRIAL,
    NodeType.POWER_PLANT,
    NodeType.AMBULANCE_DEPOT,
]



class CityLayoutCSP:

    SPECIAL_ORDER = [
        NodeType.HOSPITAL,
        NodeType.INDUSTRIAL,
        NodeType.POWER_PLANT,
        NodeType.SCHOOL,
        NodeType.AMBULANCE_DEPOT,
    ]

    def __init__(self, graph: CityGraph, seed: int = None):
        self.graph = graph
        self.size  = graph.grid_size
        if seed is not None:
            random.seed(seed)

        self._c1_radius    = CONSTRAINTS["C1_EXCLUSION_RADIUS"]
        self._c2_hops      = CONSTRAINTS["C2_HOSPITAL_HOPS"]
        self._c3_hops      = CONSTRAINTS["C3_INDUSTRIAL_HOPS"]
        self._seed_inset   = CONSTRAINTS["HOSPITAL_SEED_INSET"]

        self.target_counts = self._sample_counts()
        self.n_specials    = sum(self.target_counts[nt] for nt in SPECIAL_TYPES)

       
        self.adjacent: dict = {}
        self.within_c1: dict = {}   
        self.within_c2: dict = {}   
        self.within_c3: dict = {}   
        self._precompute()

        self.hosp_reach:  dict = defaultdict(int)  
        self.indus_reach: dict = defaultdict(int)  

        self.assignment: dict = {}
        self.hospitals_placed:   int = 0
        self.industrials_placed: int = 0

        self.failing_constraint: str = None
        self.violations:         int = 0


    def _bfs_hops(self, start: tuple, max_hops: int) -> frozenset:
        visited = {start}
        queue   = deque([(start, 0)])
        result  = []
        while queue:
            cur, d = queue.popleft()
            if d > 0:
                result.append(cur)
            if d < max_hops:
                for nb in self.adjacent.get(cur, []):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, d + 1))
        return frozenset(result)

    def _precompute(self):
        size = self.size
        for r in range(size):
            for c in range(size):
                cell = (r, c)
                adj  = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < size and 0 <= nc < size:
                        adj.append((nr, nc))
                self.adjacent[cell] = adj

        for r in range(size):
            for c in range(size):
                cell = (r, c)
                self.within_c1[cell] = self._bfs_hops(cell, self._c1_radius)
                self.within_c2[cell] = self._bfs_hops(cell, self._c2_hops)
                self.within_c3[cell] = self._bfs_hops(cell, self._c3_hops)

        print(f"[CSP] Pre-computation done: {size}x{size} = {size*size} cells.")
        print(f"[CSP] Active constraints:")
        print(f"      C1 exclusion radius : {self._c1_radius} hop(s)")
        print(f"      C2 hospital reach   : {self._c2_hops} hop(s)  [soft]")
        print(f"      C3 industrial reach : {self._c3_hops} hop(s)  [hard]")


    def _sample_counts(self) -> dict:
        counts        = {}
        total_special = 0
        for nt, (lo, hi) in NODE_COUNTS.items():
            counts[nt]    = random.randint(lo, hi)
            total_special += counts[nt]

        total_cells = self.size * self.size
        residential = total_cells - total_special

        if residential < 5:
            print("[CSP] Warning: too many specials — scaling to minimums.")
            for nt in NODE_COUNTS:
                counts[nt] = NODE_COUNTS[nt][0]
            total_special = sum(counts.values())
            residential   = total_cells - total_special

        counts[NodeType.RESIDENTIAL] = residential

        print("\n[CSP] Target node counts:")
        for nt, cnt in counts.items():
            print(f"      {nt.value:<20} -> {cnt}")
        print(f"      Total cells: {total_cells}\n")
        return counts

    def _c1_ok(self, cell: tuple, nt: NodeType) -> bool:
        forbidden_neighbours = self.within_c1[cell]  

        if nt == NodeType.INDUSTRIAL:
            for nb in forbidden_neighbours:
                if self.assignment.get(nb) in (NodeType.SCHOOL, NodeType.HOSPITAL):
                    return False
        elif nt in (NodeType.SCHOOL, NodeType.HOSPITAL):
            for nb in forbidden_neighbours:
                if self.assignment.get(nb) == NodeType.INDUSTRIAL:
                    return False
        return True

    def _c2_covered(self, cell: tuple) -> bool:
        return self.hosp_reach[cell] > 0

    def _c3_ok_strict(self, cell: tuple) -> bool:
        return self.indus_reach[cell] > 0

    def _consistent(self, cell: tuple, nt: NodeType) -> tuple:
        if not self._c1_ok(cell, nt):
            return False, (
                f"C1: Industrial within {self._c1_radius} hop(s) of School/Hospital"
            )
        return True, None


    def _apply(self, cell: tuple, nt: NodeType):
        self.assignment[cell] = nt
        if nt == NodeType.HOSPITAL:
            self.hospitals_placed += 1
            for nb in self.within_c2[cell]:
                self.hosp_reach[nb] += 1
        elif nt == NodeType.INDUSTRIAL:
            self.industrials_placed += 1
            for nb in self.within_c3[cell]:
                self.indus_reach[nb] += 1

    def _undo(self, cell: tuple, nt: NodeType):
        del self.assignment[cell]
        if nt == NodeType.HOSPITAL:
            self.hospitals_placed -= 1
            for nb in self.within_c2[cell]:
                self.hosp_reach[nb] -= 1
        elif nt == NodeType.INDUSTRIAL:
            self.industrials_placed -= 1
            for nb in self.within_c3[cell]:
                self.indus_reach[nb] -= 1


    def _pick_variable(self, unassigned: list, remaining: dict,
                        available: list) -> tuple:
        best_cell   = None
        best_mrv    = float('inf')
        best_degree = -1

        for cell in unassigned:
            valid_count = sum(
                1 for nt in available
                if remaining.get(nt, 0) > 0 and self._consistent(cell, nt)[0]
            )
            degree = sum(1 for nb in self.adjacent[cell] if nb in self.assignment)

            if valid_count < best_mrv or (
                    valid_count == best_mrv and degree > best_degree):
                best_mrv    = valid_count
                best_degree = degree
                best_cell   = cell

        return best_cell

    def _ordered_types(self, remaining: dict) -> list:
        return [nt for nt in self.SPECIAL_ORDER if remaining.get(nt, 0) > 0]


    def _backtrack(self, unassigned: list, remaining: dict, placed: int) -> bool:
        if placed == self.n_specials:
            return True

        available = self._ordered_types(remaining)
        if not available:
            return True

        cell = self._pick_variable(unassigned, remaining, available)
        if cell is None:
            self.failing_constraint = (
                f"No feasible cell for: {[nt.value for nt in available]}"
            )
            return False

        new_unassigned = [c for c in unassigned if c != cell]

        for nt in available:
            if remaining.get(nt, 0) == 0:
                continue
            ok, reason = self._consistent(cell, nt)
            if not ok:
                self.failing_constraint = reason
                continue

            self._apply(cell, nt)
            remaining[nt] -= 1

            if self._backtrack(new_unassigned, remaining, placed + 1):
                return True

            self._undo(cell, nt)
            remaining[nt] += 1

        remaining_specials = sum(remaining.get(nt, 0) for nt in available)
        if len(new_unassigned) >= remaining_specials:
            if self._backtrack(new_unassigned, remaining, placed):
                return True

        return False


    def _fill_residential(self) -> int:
        c2_violations = 0
        for r in range(self.size):
            for c in range(self.size):
                cell = (r, c)
                if cell not in self.assignment:
                    if not self._c2_covered(cell):
                        c2_violations += 1
                    self.assignment[cell] = NodeType.RESIDENTIAL

        if c2_violations:
            print(f"[CSP] C2 note: {c2_violations} Residential cell(s) are beyond "
                  f"{self._c2_hops}-hop reach of any Hospital.")
            print(f"[CSP] This is expected with {self.hospitals_placed} hospitals "
                  f"on a {self.size}x{self.size} grid (radius-{self._c2_hops} coverage math).")
            print(f"[CSP] Treated as soft constraint — layout is still valid.")
        else:
            print(f"[CSP] C2: All Residential cells have a Hospital within "
                  f"{self._c2_hops} hops.")

        return c2_violations


    def _validate(self) -> bool:
        print("\n[CSP] Final constraint validation...")
        all_ok = True

        for cell, nt in self.assignment.items():
            if nt == NodeType.INDUSTRIAL:
                for nb in self.within_c1[cell]:
                    nbt = self.assignment.get(nb)
                    if nbt in (NodeType.SCHOOL, NodeType.HOSPITAL):
                        print(f"  [FAIL C1] Industrial {cell} within "
                              f"{self._c1_radius} hop(s) of {nbt.value} at {nb}")
                        all_ok = False
            if nt == NodeType.POWER_PLANT and not self._c3_ok_strict(cell):
                print(f"  [FAIL C3] Power Plant {cell} has no Industrial within "
                      f"{self._c3_hops} hops")
                all_ok = False

        if all_ok:
            print("  [OK] C1 and C3 fully satisfied.")
        return all_ok


    def _diagnose_failing_constraint(self) -> str:
        counts = {
            "C1 — Industrial adjacent to School/Hospital":          0,
            f"C3 — Power Plant not within {self._c3_hops} hops of Industrial": 0,
            f"C2 — Residential beyond {self._c2_hops} hops of Hospital  [soft]": 0,
        }

        for cell, nt in self.assignment.items():
            if nt == NodeType.INDUSTRIAL:
                for nb in self.within_c1[cell]:
                    if self.assignment.get(nb) in (NodeType.SCHOOL, NodeType.HOSPITAL):
                        counts["C1 — Industrial adjacent to School/Hospital"] += 1
                        break  

            elif nt == NodeType.POWER_PLANT:
                if not self._c3_ok_strict(cell):
                    counts[
                        f"C3 — Power Plant not within {self._c3_hops} hops of Industrial"
                    ] += 1

            elif nt == NodeType.RESIDENTIAL:
                if not self._c2_covered(cell):
                    counts[
                        f"C2 — Residential beyond {self._c2_hops} hops of Hospital  [soft]"
                    ] += 1

        self.violation_counts = counts

        worst_name  = max(counts, key=counts.get)
        worst_count = counts[worst_name]

        if worst_count > 0:
            self.failing_constraint = worst_name

        return self.failing_constraint


    def _minimum_violation_fallback(self):
        diagnosed = self._diagnose_failing_constraint()

        print("\n[CSP] ══════════════════════════════════════════════════")
        print("[CSP] CONSTRAINT FAILURE REPORT")
        print("[CSP] ══════════════════════════════════════════════════")
        print(f"[CSP] Primary failing constraint : {diagnosed}")
        print(f"[CSP] Violation breakdown (partial assignment):")
        for name, cnt in self.violation_counts.items():
            marker = " ◄ PRIMARY" if name == diagnosed else ""
            print(f"[CSP]   {cnt:>3}  {name}{marker}")
        print("[CSP] ══════════════════════════════════════════════════")
        print("[CSP] Switching to minimum-violation fallback ...\n")

        self.assignment         = {}
        self.hosp_reach         = defaultdict(int)
        self.indus_reach        = defaultdict(int)
        self.hospitals_placed   = 0
        self.industrials_placed = 0

        remaining = dict(self.target_counts)
        all_cells = [(r, c) for r in range(self.size) for c in range(self.size)]

        self._fallback_place_spread(NodeType.HOSPITAL, remaining, all_cells)
        self._fallback_place_safe(NodeType.SCHOOL, remaining, all_cells)
        self._fallback_place_safe(NodeType.INDUSTRIAL, remaining, all_cells)
        self._fallback_place_near(NodeType.POWER_PLANT, NodeType.INDUSTRIAL,
                                   remaining, all_cells, radius=self._c3_hops)
        self._fallback_place_safe(NodeType.AMBULANCE_DEPOT, remaining, all_cells)

        for cell in all_cells:
            if cell not in self.assignment:
                self._apply(cell, NodeType.RESIDENTIAL)
                remaining[NodeType.RESIDENTIAL] = max(
                    0, remaining.get(NodeType.RESIDENTIAL, 0) - 1)

        self.violations = self._count_violations()
        print(f"[CSP] Fallback complete: {self.violations} violation(s).")

    def _fallback_place_spread(self, nt: NodeType, remaining: dict, all_cells: list):
        import math
        n    = remaining.get(nt, 0)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        step_r = self.size / rows
        step_c = self.size / cols

        placed = 0
        for i in range(rows):
            for j in range(cols):
                if placed >= n:
                    break
                r = min(max(int((i + 0.5) * step_r), 0), self.size - 1)
                c = min(max(int((j + 0.5) * step_c), 0), self.size - 1)
                cell = (r, c)
                if cell not in self.assignment:
                    self._apply(cell, nt)
                    remaining[nt] -= 1
                    placed += 1
                else:
                    for cr in range(self.size):
                        for cc in range(self.size):
                            nb = (cr, cc)
                            if nb not in self.assignment:
                                self._apply(nb, nt)
                                remaining[nt] -= 1
                                placed += 1
                                break
                        if placed > i * cols + j:
                            break

    def _fallback_place_safe(self, nt: NodeType, remaining: dict, all_cells: list):
        n = remaining.get(nt, 0)
        placed = 0
        for cell in all_cells:
            if placed >= n:
                break
            if cell in self.assignment:
                continue
            if self._c1_ok(cell, nt):
                self._apply(cell, nt)
                remaining[nt] -= 1
                placed += 1

        if placed < n:
            print(f"[CSP] Fallback warning: could only place {placed}/{n} {nt.value} "
                  f"without C1 violation.")

    def _fallback_place_near(self, nt: NodeType, anchor: NodeType,
                              remaining: dict, all_cells: list, radius: int):
        n = remaining.get(nt, 0)
        anchors = {cell for cell, t in self.assignment.items() if t == anchor}
        reach_dict = self.within_c3 if radius == self._c3_hops else self.within_c2

        def score(cell):
            if cell in self.assignment:
                return 999
            nearby = reach_dict[cell]
            return 0 if anchors.intersection(nearby) else 1

        candidates = sorted(all_cells, key=score)
        placed = 0
        for cell in candidates:
            if placed >= n:
                break
            if cell in self.assignment:
                continue
            if self._c1_ok(cell, nt):
                self._apply(cell, nt)
                remaining[nt] -= 1
                placed += 1


    def _count_violations(self) -> int:
        count = 0
        for cell, nt in self.assignment.items():
            if nt == NodeType.INDUSTRIAL:
                for nb in self.within_c1[cell]:
                    if self.assignment.get(nb) in (NodeType.SCHOOL, NodeType.HOSPITAL):
                        count += 1
            if nt == NodeType.RESIDENTIAL and not self._c2_covered(cell):
                count += 1
            if nt == NodeType.POWER_PLANT and not self._c3_ok_strict(cell):
                count += 1
        return count


    def _preseed_hospitals(self, n_hospitals: int) -> list:
        
        import math
        cols = math.ceil(math.sqrt(n_hospitals))
        rows = math.ceil(n_hospitals / cols)

        inset = self._seed_inset
        r_positions = [max(inset, round((i + 0.5) * (self.size / rows)))
                       for i in range(rows)]
        c_positions = [max(inset, round((j + 0.5) * (self.size / cols)))
                       for j in range(cols)]
        r_positions = [min(r, self.size - inset - 1) for r in r_positions]
        c_positions = [min(c, self.size - inset - 1) for c in c_positions]

        seeds = []
        for r in r_positions:
            for c in c_positions:
                if len(seeds) >= n_hospitals:
                    break
                cell = (r, c)
                if cell not in self.assignment:
                    self._apply(cell, NodeType.HOSPITAL)
                    seeds.append(cell)

        print(f"[CSP] Pre-seeded {len(seeds)} Hospital(s) at: {seeds}")
        return seeds


    def solve(self) -> bool:
        print("=" * 60)
        print("  Challenge 1 — City Layout Planning  (CSP Solver)")
        print("=" * 60)
        print(f"[CSP] Grid: {self.size}x{self.size} = {self.size**2} cells")
        print(f"[CSP] Specials to place: {self.n_specials}\n")

        self.failing_constraint = None

        remaining = {nt: self.target_counts[nt] for nt in SPECIAL_TYPES}

        n_hosp = self.target_counts[NodeType.HOSPITAL]
        seeded = self._preseed_hospitals(n_hosp)
        remaining[NodeType.HOSPITAL] -= len(seeded)
        placed_count = len(seeded)

        all_cells = [(r, c) for r in range(self.size) for c in range(self.size)
                     if (r, c) not in self.assignment]
        random.shuffle(all_cells)

        t0      = time.perf_counter()
        success = self._backtrack(all_cells, remaining, placed_count)
        t1      = time.perf_counter()
        print(f"[CSP] Phase 1 (backtracking specials): {t1-t0:.4f}s — "
              f"{'SUCCESS' if success else 'FAILED'}")

        if not success:
            self._minimum_violation_fallback()
            self._apply_to_graph()
            self._print_grid()
            return False

        t2            = time.perf_counter()
        c2_violations = self._fill_residential()
        t3            = time.perf_counter()
        print(f"[CSP] Phase 2 (residential fill):      {t3-t2:.4f}s")

        all_ok = self._validate()
        if not all_ok:
            self.violations = self._count_violations()
            self._diagnose_failing_constraint()
            print(f"[CSP] Primary failing constraint : {self.failing_constraint}")
            for name, cnt in self.violation_counts.items():
                if cnt:
                    print(f"[CSP]   {cnt:>3}  {name}")

        self._apply_to_graph()
        self._print_grid(c2_violations)

        print(f"[CSP] Total solve time: {t3-t0:.4f}s")
        return all_ok


    def _apply_to_graph(self):
        density_ranges = {
            NodeType.RESIDENTIAL:     (100, 500),
            NodeType.HOSPITAL:        (30,  80),
            NodeType.SCHOOL:          (20,  60),
            NodeType.INDUSTRIAL:      (50,  150),
            NodeType.POWER_PLANT:     (10,  30),
            NodeType.AMBULANCE_DEPOT: (5,   20),
        }
        for cell, nt in self.assignment.items():
            lo, hi  = density_ranges[nt]
            density = random.randint(lo, hi)
            self.graph.set_node_type(cell, nt, population_density=density)
        print(f"\n[CSP] {len(self.assignment)} nodes written to shared graph.")


    def _print_grid(self, c2_violations: int = 0):
        symbols = {
            NodeType.HOSPITAL:        " H ",
            NodeType.SCHOOL:          " S ",
            NodeType.INDUSTRIAL:      " I ",
            NodeType.POWER_PLANT:     " P ",
            NodeType.AMBULANCE_DEPOT: " A ",
            NodeType.RESIDENTIAL:     " . ",
        }
        print("\nCity Grid  (H=Hospital S=School I=Industrial P=PowerPlant "
              "A=AmbulanceDepot .=Residential)")
        print("    " + "".join(f"{c:^3}" for c in range(self.size)))
        print("   +" + "---" * self.size + "+")
        for r in range(self.size):
            row = f"{r:2} |"
            for c in range(self.size):
                nt   = self.assignment.get((r, c))
                row += symbols.get(nt, " ? ")
            print(row + "|")
        print("   +" + "---" * self.size + "+\n")

        from collections import Counter
        counts = Counter(self.assignment.values())
        print("Node type summary:")
        for nt in SPECIAL_TYPES + [NodeType.RESIDENTIAL]:
            print(f"  {nt.value:<20} : {counts.get(nt, 0)}")
        if c2_violations:
            print(f"\n  [SOFT] C2 violations: {c2_violations} corner/edge residential "
                  f"cells beyond {self._c2_hops}-hop hospital reach "
                  f"(expected with {self.hospitals_placed} hospitals)")
        print()



if __name__ == "__main__":
    graph   = CityGraph(grid_size=10)
    csp     = CityLayoutCSP(graph, seed=42)
    success = csp.solve()

    print("=" * 60)
    if success:
        print("  Result: VALID layout (C1 + C3 satisfied)")
    else:
        print("  Result: minimum-violation layout — see details above")
    print("=" * 60)

    if not _STANDALONE:
        print("\nSample nodes from shared graph:")
        for r in range(0, 10, 3):
            for c in range(0, 10, 3):
                n = graph.get_node(r, c)
                if n:
                    print(f"  {n} | density={n.population_density}")

    print("\n" + "=" * 60)
    print("  LIVE MODIFICATION DEMO")
    print("  Changing C2_HOSPITAL_HOPS from 3 → 4 and re-solving...")
    print("=" * 60)
    CONSTRAINTS["C2_HOSPITAL_HOPS"] = 4
    graph2   = CityGraph(grid_sizes=10)
    csp2     = CityLayoutCSP(graph2, seed=42)
    success2 = csp2.solve()
    print(f"\n  Result with 4-hop hospital reach: "
          f"{'VALID' if success2 else 'minimum-violation'}")