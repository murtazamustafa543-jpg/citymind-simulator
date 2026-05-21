"""
CityMind — Dashboard UI (Pygame) — Light Theme
"""

import pygame
import pygame.freetype
import sys
import math
import random
import threading
import time
from collections import defaultdict

from city_graph               import CityGraph, NodeType, CrimeRisk
from challenge1csp            import CityLayoutCSP
from challenge2road           import RoadNetworkGA, apply_road_network
from challenge3ambulance      import AmbulanceSA, rerun_placement
from challenge4emergencyroute import EmergencyRouter, EventLog
from challenge5crime          import CrimeRiskPipeline



GRID_SIZE         = 10
CELL              = 72
SIDEBAR_W         = 240
LOG_W             = 340
TOP_H             = 72
BOT_H             = 140   
GAP               = 2

GRID_PX           = GRID_SIZE * (CELL + GAP) - GAP
GRID_X            = SIDEBAR_W + 12
GRID_Y            = TOP_H + 10

WIN_W             = SIDEBAR_W + 12 + GRID_PX + 12 + LOG_W + 8
WIN_H             = TOP_H + 10 + GRID_PX + 10 + BOT_H

SEED              = 42
NUM_CIVILIANS     = 5
CRIME_SHIFT_EVERY = 5
FLOOD_PROB        = 0.2  
MAX_STEPS         = 20

SPEED_LEVELS      = [2000, 1200, 600, 300, 100]
DEFAULT_SPEED     = 2



def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

BG          = rgb("F0F2F5")
PANEL_BG    = rgb("FFFFFF")
PANEL2      = rgb("F7F8FA")
BORDER      = rgb("D0D7DE")

TXT_PRI     = rgb("1C2128")
TXT_SEC     = rgb("656D76")
TXT_DIM     = rgb("9AA6B2")

BLUE        = rgb("0969DA")
GREEN       = rgb("1A7F37")
AMBER       = rgb("9A6700")
RED_COL     = rgb("CF222E")
PURPLE      = rgb("8250DF")
ORANGE      = rgb("BC4C00")
TEAL        = rgb("0A7060")

NODE_COL = {
    NodeType.HOSPITAL:        rgb("D4EDDA"),
    NodeType.SCHOOL:          rgb("CCE5FF"),
    NodeType.RESIDENTIAL:     rgb("F0F2F5"),
    NodeType.INDUSTRIAL:      rgb("FDE5CC"),
    NodeType.POWER_PLANT:     rgb("FFF3CD"),
    NodeType.AMBULANCE_DEPOT: rgb("EAD5FB"),
    None:                     rgb("E8ECEF"),
}

NODE_BORDER = {
    NodeType.HOSPITAL:        rgb("28A745"),
    NodeType.SCHOOL:          rgb("0D6EFD"),
    NodeType.RESIDENTIAL:     rgb("CED4DA"),
    NodeType.INDUSTRIAL:      rgb("FD7E14"),
    NodeType.POWER_PLANT:     rgb("FFC107"),
    NodeType.AMBULANCE_DEPOT: rgb("6F42C1"),
    None:                     rgb("DEE2E6"),
}

NODE_TXT = {
    NodeType.HOSPITAL:        rgb("155724"),
    NodeType.SCHOOL:          rgb("084298"),
    NodeType.RESIDENTIAL:     rgb("9AA6B2"),
    NodeType.INDUSTRIAL:      rgb("7B3300"),
    NodeType.POWER_PLANT:     rgb("664D03"),
    NodeType.AMBULANCE_DEPOT: rgb("4A0E8F"),
    None:                     rgb("9AA6B2"),
}

NODE_LBL = {
    NodeType.HOSPITAL:        "H",
    NodeType.SCHOOL:          "S",
    NodeType.RESIDENTIAL:     "·",
    NodeType.INDUSTRIAL:      "I",
    NodeType.POWER_PLANT:     "P",
    NodeType.AMBULANCE_DEPOT: "A",
    None:                     "?",
}



class SimState:
    def __init__(self):
        self.ready       = False
        self.graph       = None
        self.router      = None
        self.sa          = None
        self.log         = None
        self.amb_pos     = []
        self.civilians   = []
        self.step        = 0
        self.done        = False
        self.status_msg  = "Initialising…"
        self.c2_flow     = 0
        self.c2_cost     = 0.0
        self._coverage   = {}
        self.police_deployment = {}

        self.history: list = []
        self.flood_count: int = 0
        self.timeline: list  = []

    def build(self, on_done):
        threading.Thread(target=self._run, args=(on_done,), daemon=True).start()

    def _run(self, on_done):
        try:
            self._build_inner()
            self.ready = True
            on_done()
        except Exception as e:
            self.status_msg = f"ERROR: {e}"
            raise

    def _build_inner(self):
        random.seed(SEED)
        size = GRID_SIZE

        self.graph = CityGraph(grid_size=size)
        self.log   = EventLog()

        self.status_msg = "C1 — City Layout Planning…"
        csp = CityLayoutCSP(self.graph, seed=SEED)
        csp.solve()

        all_hospitals = self.graph.get_nodes_by_type(NodeType.HOSPITAL)
        all_depots    = self.graph.get_nodes_by_type(NodeType.AMBULANCE_DEPOT)
        if all_hospitals:
            self.primary_hospital = max(
                all_hospitals, key=lambda n: n.population_density
            )
        else:
            self.primary_hospital = list(self.graph.nodes.values())[0]

        if all_depots:
            self.primary_depot = max(
                all_depots, key=lambda n: n.population_density
            )
        else:
            self.primary_depot = list(self.graph.nodes.values())[-1]

        print(f"[C2] Primary Hospital : {self.primary_hospital.node_id} "
              f"(density={self.primary_hospital.population_density})")
        print(f"[C2] Primary Depot    : {self.primary_depot.node_id} "
              f"(density={self.primary_depot.population_density})")

        self.status_msg = "C2 — Road Network Optimization…"
        edges = []
        for edge in self.graph.edges.values():
            if not edge.is_blocked:
                u = edge.node_a[0] * size + edge.node_a[1]
                v = edge.node_b[0] * size + edge.node_b[1]
                edges.append((edge.effective_cost, u, v))
        h_id = self.primary_hospital.node_id[0] * size + self.primary_hospital.node_id[1]
        d_id = self.primary_depot.node_id[0]    * size + self.primary_depot.node_id[1]
        ga   = RoadNetworkGA(size*size, edges, h_id, d_id, seed=SEED)
        res  = ga.run(verbose=False)
        apply_road_network(self.graph, res)
        self.c2_flow = res["flow"]
        self.c2_cost = res["total_cost"]
        if self.c2_flow < 2:
            self.log.log(0, "WARNING",
                         f"[C2] Two-path guarantee NOT met — "
                         f"flow={self.c2_flow}. Single road failure may cut Hospital↔Depot route.")

        self.status_msg = "C5 — Crime Risk Prediction & Police Deployment…"
        c5_result = CrimeRiskPipeline(self.graph, seed=SEED).run()
        self.police_deployment = c5_result["deployment"]

        self.status_msg = "C3 — Ambulance Placement…"
        self.sa      = AmbulanceSA(city_graph=self.graph, seed=SEED)
        amb_res      = self.sa.run(verbose=False)
        self.amb_pos = amb_res["positions"]
        self._coverage = amb_res["coverage"]

        self.status_msg = "C4 — Emergency Router Setup…"
        start      = self.primary_hospital.node_id
        res_nodes  = [n.node_id for n in self.graph.get_nodes_by_type(NodeType.RESIDENTIAL)]
        random.shuffle(res_nodes)
        self.civilians = [n for n in res_nodes if n != start][:NUM_CIVILIANS]

        self.router = EmergencyRouter(
            city_graph     = self.graph,
            start_node     = start,
            civilian_nodes = self.civilians[:],
            log            = self.log,
        )
        self.router.plan_to_next()
        self.step       = 0
        self.done       = False
        self.status_msg = "Ready"

    def _snapshot(self):
        return {
            "step"          : self.step,
            "team_pos"      : self.router.team_position,
            "civilian_queue": list(self.router.civilian_queue),
            "deferred"      : list(self.router.deferred),
            "civilians_saved": list(self.router.civilians_saved),
            "amb_pos"       : list(self.amb_pos),
            "blocked"       : [
                (e.node_a, e.node_b)
                for e in self.graph.edges.values() if e.is_blocked
            ],
            "log_len"       : len(self.log.entries),
            "flood_count"   : self.flood_count,
            "timeline"      : list(self.timeline),
        }

    def advance(self):
        if self.done or not self.ready:
            return
        self.history.append(self._snapshot())

        self.step += 1
        self.router.step = self.step
        crime_shifted = False

        if random.random() < FLOOD_PROB:
            passable = [(e.node_a, e.node_b)
                        for e in self.graph.edges.values() if not e.is_blocked]
            if passable:
                a, b = random.choice(passable)
                self.graph.block_road(a, b)
                self.flood_count += 1
                self.timeline.append(("flood", self.step))
                self.log.log(self.step, "CRITICAL", f"Road FLOODED: {a} ↔ {b}")
                self.router.replan()

        if self.step % CRIME_SHIFT_EVERY == 0:
            candidates = [(nid, n) for nid, n in self.graph.nodes.items()
                          if n.crime_risk_level != CrimeRisk.HIGH]
            if candidates:
                nid, node = random.choice(candidates)
                new_risk  = (CrimeRisk.MEDIUM if node.crime_risk_level == CrimeRisk.LOW
                             else CrimeRisk.HIGH)
                self.graph.update_crime_risk(nid, new_risk)
                self.log.log(self.step, "INFO",
                             f"[C5] Crime shift: {nid} → {new_risk.value}")
                crime_shifted = True

        if crime_shifted:
            amb_res        = rerun_placement(self.sa, verbose=False)
            self.amb_pos   = amb_res["positions"]
            self._coverage = amb_res["coverage"]
            self.log.log(self.step, "INFO", f"[C3] Ambulances → {self.amb_pos}")

        status = self.router.step_team()
        if status == "rescued":
            self.timeline.append(("rescue", self.step))
        elif status == "moving":
            self.timeline.append(("move", self.step))

        if status == "done" or self.step >= MAX_STEPS:
            self.done = True
            if self.router.civilian_queue:
                self.log.log(self.step, "WARNING",
                            f"Sim ended — {len(self.router.civilian_queue)} missed")

    def coverage_dist(self, node_id):
        best = math.inf
        for dists in self._coverage.values():
            d = dists.get(node_id, math.inf)
            if d < best:
                best = d
        return best

    def worst_response(self):
        if not self.amb_pos:
            return 0.0
        max_d = 0.0
        for nid in self.graph.nodes:
            d = self.coverage_dist(nid)
            if d != math.inf and d > max_d:
                max_d = d
        return max_d

    def coverage_pct(self, threshold: float = 5.0) -> float:
        if not self.amb_pos:
            return 0.0
        populated = [
            nid for nid, n in self.graph.nodes.items()
            if n.population_density > 0
        ]
        if not populated:
            return 0.0
        covered = sum(
            1 for nid in populated
            if self.coverage_dist(nid) <= threshold
        )
        return 100.0 * covered / len(populated)


def cell_rect(r, c):
    x = GRID_X + c * (CELL + GAP)
    y = GRID_Y + r * (CELL + GAP)
    return pygame.Rect(x, y, CELL, CELL)

def cell_center(r, c):
    rect = cell_rect(r, c)
    return rect.centerx, rect.centery

def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def blend_surf(surface, rect, color, alpha):
    s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    s.fill((*color, alpha))
    surface.blit(s, rect.topleft)

def shadow_rect(surf, rect, radius=8, blur=6, alpha=30):
    for i in range(blur, 0, -1):
        r2 = rect.inflate(i*2, i*2)
        a  = int(alpha * (1 - i/blur))
        s  = pygame.Surface((r2.width, r2.height), pygame.SRCALPHA)
        pygame.draw.rect(s, (0, 0, 0, a), s.get_rect(), border_radius=radius+i)
        surf.blit(s, (r2.x, r2.y))


class Button:
    def __init__(self, x, y, w, h, label,
                 bg=PANEL_BG, fg=TXT_PRI, border=BORDER,
                 hover_bg=None, active_bg=None,
                 toggle=False, key=None, icon=None):
        self.rect      = pygame.Rect(x, y, w, h)
        self.label     = label
        self.icon      = icon
        self.bg        = bg
        self.fg        = fg
        self.border    = border
        self.hover_bg  = hover_bg  or lerp_color(bg, TXT_PRI, 0.06)
        self.active_bg = active_bg or BLUE
        self.toggle    = toggle
        self.active    = False
        self.key       = key
        self._hover    = False

    def draw(self, surf, ft):
        bg         = self.bg
        fg         = self.fg
        border_col = self.border

        if self.toggle and self.active:
            bg         = lerp_color(self.active_bg, PANEL_BG, 0.85)
            fg         = self.active_bg
            border_col = self.active_bg
        elif self._hover:
            bg = self.hover_bg

        shadow_rect(surf, self.rect, radius=7, blur=3, alpha=18)
        pygame.draw.rect(surf, bg, self.rect, border_radius=7)
        pygame.draw.rect(surf, border_col, self.rect,
                         2 if (self.toggle and self.active) else 1, border_radius=7)

        text = (self.icon + "  " + self.label) if (self.icon and self.label) else (self.icon or self.label)
        tw = ft.get_rect(text).width
        th = ft.get_rect(text).height
        tx = self.rect.centerx - tw // 2
        ty = self.rect.centery - th // 2
        ft.render_to(surf, (tx, ty), text, fg)

        if self.toggle and self.active:
            bar = pygame.Rect(self.rect.x + 6, self.rect.bottom - 5,
                              self.rect.width - 12, 3)
            pygame.draw.rect(surf, self.active_bg, bar, border_radius=2)

            dot_x = self.rect.right - 10
            dot_y = self.rect.centery
            pygame.draw.circle(surf, self.active_bg, (dot_x, dot_y), 4)

    def handle(self, event):
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.toggle:
                    self.active = not self.active
                return True
        return False

class Slider:
    def __init__(self, x, y, w, min_val, max_val, start_val):
        self.rect = pygame.Rect(x, y, w, 6)

        self.min_val = min_val
        self.max_val = max_val
        self.value   = start_val

        self.knob_r = 10
        self.dragging = False

    def knob_x(self):
        t = (self.value - self.min_val) / (self.max_val - self.min_val)
        return int(self.rect.x + t * self.rect.width)

    def draw(self, surf):
        pygame.draw.rect(
            surf,
            BORDER,
            self.rect,
            border_radius=3
        )

        fill_w = self.knob_x() - self.rect.x
        pygame.draw.rect(
            surf,
            BLUE,
            (self.rect.x, self.rect.y, fill_w, self.rect.height),
            border_radius=3
        )

        pygame.draw.circle(
            surf,
            PANEL_BG,
            (self.knob_x(), self.rect.centery),
            self.knob_r
        )

        pygame.draw.circle(
            surf,
            BLUE,
            (self.knob_x(), self.rect.centery),
            self.knob_r,
            2
        )

    def handle(self, event):
        knob_rect = pygame.Rect(
            self.knob_x() - self.knob_r,
            self.rect.centery - self.knob_r,
            self.knob_r * 2,
            self.knob_r * 2
        )

        if event.type == pygame.MOUSEBUTTONDOWN:
            if knob_rect.collidepoint(event.pos):
                self.dragging = True

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mx = max(self.rect.x, min(event.pos[0], self.rect.right))

            t = (mx - self.rect.x) / self.rect.width

            self.value = round(
                self.min_val +
                t * (self.max_val - self.min_val)
            )

            return True

        return False


class CityMindApp:

    def __init__(self):
        pygame.init()
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
        pygame.freetype.init()

        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("CityMind — Urban Intelligence Dashboard")

        try:
            self.ft_title  = pygame.freetype.SysFont("Segoe UI", 26, bold=True)
            self.ft_body   = pygame.freetype.SysFont("Segoe UI", 18)
            self.ft_small  = pygame.freetype.SysFont("Segoe UI", 15)
            self.ft_node   = pygame.freetype.SysFont("Segoe UI", 20, bold=True)
            self.ft_step   = pygame.freetype.SysFont("Segoe UI", 30, bold=True)
            self.ft_label  = pygame.freetype.SysFont("Segoe UI", 15)
        except Exception:
            self.ft_title  = pygame.freetype.SysFont(None, 20, bold=True)
            self.ft_body   = pygame.freetype.SysFont(None, 16)
            self.ft_small  = pygame.freetype.SysFont(None, 15)
            self.ft_node   = pygame.freetype.SysFont(None, 17, bold=True)
            self.ft_step   = pygame.freetype.SysFont(None, 26, bold=True)
            self.ft_label  = pygame.freetype.SysFont(None, 14)

        self.sim        = SimState()
        self.playing    = False
        self.speed_idx  = DEFAULT_SPEED
        self._last_t    = 0
        self.log_scroll = 0

        ox = GRID_X
        oy = 14
        bh = 36
        self.ov_road  = Button(ox,      oy, 148, bh, "Road Network",
                               toggle=True, active_bg=BLUE,    icon="⬡")
        self.ov_cov   = Button(ox+156,  oy, 158, bh, "Amb. Coverage",
                               toggle=True, active_bg=GREEN,   icon="◎")
        self.ov_crime = Button(ox+322,  oy, 158, bh, "Crime Heatmap",
                               toggle=True, active_bg=RED_COL, icon="▲")

        STATS_ROW_H  = 55
        TIMELINE_H   = 28
        bby = WIN_H - BOT_H + STATS_ROW_H + TIMELINE_H + (BOT_H - STATS_ROW_H - TIMELINE_H - 40) // 2
        bx  = GRID_X

        self.btn_play   = Button(bx,      bby,  90, 40, "Play",
                                 bg=rgb("E6F4EA"), fg=GREEN,
                                 hover_bg=rgb("D1EBD8"), icon="▶")
        self.btn_pause  = Button(bx+98,   bby,  90, 40, "Pause",   icon="⏸")
        self.btn_step   = Button(bx+196,  bby, 110, 40, "Next Step",
                                 bg=rgb("E8F0FE"), fg=BLUE,
                                 hover_bg=rgb("D2E3FC"), icon="▶|")
        self.btn_rewind = Button(bx+314,  bby, 110, 40, "Rewind",
                                 bg=rgb("FFF3E0"), fg=AMBER,
                                 hover_bg=rgb("FFE0B2"), icon="◀|")
        self.btn_reset  = Button(bx+432,  bby,  90, 40, "Reset",
                                 bg=rgb("FEECEC"), fg=RED_COL,
                                 hover_bg=rgb("FDD8D8"), icon="↺")
        self.speed_slider = Slider(
            bx + 540,
            bby + 17,
            140,
            0,
            len(SPEED_LEVELS) - 1,
            DEFAULT_SPEED
        )

        self.btn_close  = Button(WIN_W - 44, 12, 32, 32, "",
                                 bg=PANEL_BG, fg=RED_COL,
                                 hover_bg=rgb("FEECEC"), icon="✕")

        self.overlays  = [self.ov_road, self.ov_cov, self.ov_crime]
        self.ctrl_btns = [self.btn_play, self.btn_pause, self.btn_step,
                          self.btn_rewind, self.btn_reset]
        self.all_btns  = self.overlays + self.ctrl_btns + [self.btn_close]

        self.sim.build(on_done=lambda: None)
        self.clock = pygame.time.Clock()

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if k == pygame.K_SPACE:
                    self._toggle_play()
                if k == pygame.K_s and self.sim.ready and not self.sim.done:
                    self._do_step()
                if k == pygame.K_z and self.sim.ready:
                    self._do_rewind()
                if k == pygame.K_r:
                    self._reset()
                if k == pygame.K_1: self.ov_road.active  = not self.ov_road.active
                if k == pygame.K_2: self.ov_cov.active   = not self.ov_cov.active
                if k == pygame.K_3: self.ov_crime.active = not self.ov_crime.active
                if k == pygame.K_UP:
                    self.log_scroll = max(0, self.log_scroll - 1)
                if k == pygame.K_DOWN:
                    self.log_scroll += 1
            if self.speed_slider.handle(ev):
                self.speed_idx = self.speed_slider.value

            if ev.type == pygame.MOUSEWHEEL:
                self.log_scroll = max(0, self.log_scroll - ev.y)

            for btn in self.all_btns:
                if btn.handle(ev):
                    self._on_btn(btn)

    def _toggle_play(self):
        if self.sim.ready and not self.sim.done:
            self.playing = not self.playing

    def _do_step(self):
        self.sim.advance()
        if self.sim.log:
            self.log_scroll = max(0, len(self.sim.log.entries) - 20)

    def _on_btn(self, btn):
        if btn == self.btn_close:
            pygame.quit(); sys.exit()
        elif btn == self.btn_play:
            if self.sim.ready and not self.sim.done:
                self.playing = True
        elif btn == self.btn_pause:
            self.playing = False
        elif btn == self.btn_step:
            if self.sim.ready and not self.sim.done:
                self.playing = False
                self._do_step()
        elif btn == self.btn_rewind:
            self.playing = False
            self._do_rewind()
        elif btn == self.btn_reset:
            self._reset()
        

    def _do_rewind(self):
        if not self.sim.history:
            return
        snap = self.sim.history.pop()
        self.sim.step                   = snap["step"]
        self.sim.done                   = False
        self.sim.router.team_position   = snap["team_pos"]
        self.sim.router.civilian_queue  = snap["civilian_queue"]
        self.sim.router.deferred        = snap["deferred"]
        self.sim.router.civilians_saved = snap["civilians_saved"]
        self.sim.amb_pos                = snap["amb_pos"]
        for e in self.sim.graph.edges.values():
            e.is_blocked = False
        for a, b in snap["blocked"]:
            edge = self.sim.graph.get_edge(a, b)
            if edge:
                edge.is_blocked = True
        self.sim.log.entries       = self.sim.log.entries[:snap["log_len"]]
        self.sim.flood_count = snap.get("flood_count", 0)
        self.sim.timeline    = list(snap.get("timeline", []))
        self.log_scroll = max(0, len(self.sim.log.entries) - 20)

    def _reset(self):
        self.playing     = False
        self.log_scroll  = 0
        self.sim         = SimState()
        self.sim.build(on_done=lambda: None)

    def _tick(self):
        if not self.playing or not self.sim.ready or self.sim.done:
            return
        now = time.time() * 1000
        if now - self._last_t >= SPEED_LEVELS[self.speed_idx]:
            self._do_step()
            self._last_t = now

    def draw(self):
        self.screen.fill(BG)
        self._draw_top_bar()
        self._draw_bottom_bar()
        self._draw_sidebar()
        self._draw_log_panel()
        self.speed_slider.draw(self.screen)

        if not self.sim.ready:
            self._draw_loading()
        else:
            self._draw_grid()
            if self.ov_road.active:  self._draw_roads_overlay()
            if self.ov_cov.active:   self._draw_coverage_overlay()
            if self.ov_crime.active: self._draw_crime_overlay()
            self._draw_agents()
            self._draw_timeline()
            self._draw_completion_banner()

        for btn in self.overlays:
            btn.draw(self.screen, self.ft_small)
        for btn in self.ctrl_btns:
            btn.draw(self.screen, self.ft_body)
        self.btn_close.draw(self.screen, self.ft_body)

        pygame.display.flip()


    def _draw_top_bar(self):
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, WIN_W, TOP_H))
        pygame.draw.line(self.screen, BORDER, (0, TOP_H), (WIN_W, TOP_H), 1)

        self.ft_step.render_to(self.screen, (16, 12), "CityMind", BLUE)
        self.ft_small.render_to(self.screen, (16, 46), "Urban Intelligence System", TXT_DIM)

        self.ft_label.render_to(self.screen, (GRID_X, TOP_H - 14),
                                "OVERLAYS  [1] [2] [3]", TXT_DIM)

        if self.sim.ready:
            txt = f"Step  {self.sim.step:>2}  /  {MAX_STEPS}"
            col = RED_COL if self.sim.done else (AMBER if self.playing else TXT_PRI)
            tw  = self.ft_step.get_rect(txt).width
            cx  = WIN_W - tw - 180
            self.ft_step.render_to(self.screen, (cx, 10), txt, col)

            state = "● DONE" if self.sim.done else ("▶ LIVE" if self.playing else "⏸ PAUSED")
            scol  = RED_COL if self.sim.done else (GREEN if self.playing else TXT_SEC)
            sw    = self.ft_small.get_rect(state).width
            self.ft_small.render_to(self.screen, (WIN_W - sw - 180, 44), state, scol)

            bar_x = cx
            bar_w = tw
            bar_y = TOP_H - 8
            bar_h = 4
            pygame.draw.rect(self.screen, BORDER,
                             (bar_x, bar_y, bar_w, bar_h), border_radius=2)
            filled  = int(bar_w * self.sim.step / MAX_STEPS)
            bar_col = RED_COL if self.sim.done else (GREEN if self.playing else BLUE)
            if filled > 0:
                pygame.draw.rect(self.screen, bar_col,
                                 (bar_x, bar_y, filled, bar_h), border_radius=2)


    def _draw_bottom_bar(self):
        by = WIN_H - BOT_H
        pygame.draw.rect(self.screen, PANEL_BG, (0, by, WIN_W, BOT_H))
        pygame.draw.line(self.screen, BORDER, (0, by), (WIN_W, by), 1)
        div_y = by + 55 + 28
        pygame.draw.line(self.screen, BORDER, (GRID_X, div_y), (WIN_W - LOG_W - 20, div_y), 1)

        if not self.sim.ready:
            return

        sx  = GRID_X
        sy  = by + 8       
        bh2 = 10           

        saved = len(self.sim.router.civilians_saved)
        total = NUM_CIVILIANS
        col1  = GREEN if saved == total else ORANGE

        sx  = GRID_X        

        sx2 = sx + 200      

        sx3 = sx + 380      

        sx5 = sx + 530      

        self.ft_label.render_to(self.screen, (sx, sy), "CIVILIANS RESCUED", TXT_DIM)
        bar_y = sy + 19
        bar_w = 130
        pygame.draw.rect(self.screen, BORDER, (sx, bar_y, bar_w, bh2), border_radius=5)
        if total > 0 and saved > 0:
            fill = int(bar_w * saved / total)
            pygame.draw.rect(self.screen, col1, (sx, bar_y, fill, bh2), border_radius=5)
        val_x = sx + bar_w + 6
        self.ft_small.render_to(self.screen, (val_x, bar_y - 1), f"{saved}/{total}", col1)

        sx2   = sx + 230
        pct   = self.sim.coverage_pct(threshold=5.0)
        col2  = GREEN if pct >= 80 else (AMBER if pct >= 50 else RED_COL)

        self.ft_label.render_to(self.screen, (sx2, sy), "AMB. COVERAGE ≤5", TXT_DIM)
        bar2_y = sy + 19
        bar2_w = 120
        pygame.draw.rect(self.screen, BORDER, (sx2, bar2_y, bar2_w, bh2), border_radius=5)
        fill2 = int(bar2_w * pct / 100)
        if fill2 > 0:
            pygame.draw.rect(self.screen, col2, (sx2, bar2_y, fill2, bh2), border_radius=5)
        self.ft_small.render_to(self.screen, (sx2 + bar2_w + 6, bar2_y - 1), f"{pct:.0f}%", col2)

        sx3 = sx + 420                          
        wd  = self.sim.worst_response()
        wc  = GREEN if wd < 5 else (AMBER if wd < 10 else RED_COL)

        self.ft_label.render_to(self.screen, (sx3, sy), "WORST RESPONSE", TXT_DIM)
        self.ft_small.render_to(self.screen, (sx3, sy + 18), f"{wd:.2f}", wc)

        sx5 = sx + 560                          

        bk  = sum(1 for e in self.sim.graph.edges.values() if e.is_blocked)
        bkc = RED_COL if bk > 0 else TXT_PRI

        self.ft_label.render_to(self.screen, (sx5, sy), "BLOCKED ROADS", TXT_DIM)
        self.ft_small.render_to(self.screen, (sx5, sy + 18), str(bk), bkc)

    def _draw_sidebar(self):
        sr = pygame.Rect(4, TOP_H+4, SIDEBAR_W-8, WIN_H-TOP_H-BOT_H-8)
        shadow_rect(self.screen, sr, radius=8, blur=4, alpha=14)
        pygame.draw.rect(self.screen, PANEL_BG, sr, border_radius=8)
        pygame.draw.rect(self.screen, BORDER,   sr, 1, border_radius=8)

        x, y = sr.x+12, sr.y+12

        self.ft_label.render_to(self.screen, (x, y), "LOCATION TYPES", TXT_DIM)
        y += 18

        legend = [
            ("H  Hospital",       NodeType.HOSPITAL),
            ("S  School",         NodeType.SCHOOL),
            ("·  Residential",    NodeType.RESIDENTIAL),
            ("I  Industrial",     NodeType.INDUSTRIAL),
            ("P  Power Plant",    NodeType.POWER_PLANT),
            ("A  Amb. Depot",     NodeType.AMBULANCE_DEPOT),
        ]
        for lbl, nt in legend:
            col  = NODE_COL[nt]
            bcol = NODE_BORDER[nt]
            pygame.draw.rect(self.screen, col,  (x, y+1, 13, 13), border_radius=3)
            pygame.draw.rect(self.screen, bcol, (x, y+1, 13, 13), 1, border_radius=3)
            self.ft_small.render_to(self.screen, (x+18, y), lbl, TXT_PRI)
            y += 18

        y += 8
        self.ft_label.render_to(self.screen, (x, y), "MARKERS", TXT_DIM)
        y += 18

        markers = [
            ("⊕  Ambulance",    GREEN),
            ("M  Med. Team",    AMBER),
            ("C  Civilian",     ORANGE),
            ("✕  Blocked road", RED_COL),
            ("●  Police (C5)",  BLUE),
        ]
        for lbl, col in markers:
            self.ft_small.render_to(self.screen, (x, y), lbl, col)
            y += 18

        if not self.sim.ready:
            return

        y += 8
        pygame.draw.line(self.screen, BORDER, (x, y), (sr.right-12, y))
        y += 8

        self.ft_label.render_to(self.screen, (x, y), "NETWORK", TXT_DIM)
        y += 16
        self.ft_small.render_to(self.screen, (x, y), f"Cost    {self.sim.c2_cost:.1f}", TXT_PRI)
        y += 16
        fc = GREEN if self.sim.c2_flow >= 2 else RED_COL
        self.ft_small.render_to(self.screen, (x, y), f"Flow    {self.sim.c2_flow} paths", fc)
        y += 16
        bk = sum(1 for e in self.sim.graph.edges.values() if e.is_blocked)
        bc = RED_COL if bk else TXT_PRI
        self.ft_small.render_to(self.screen, (x, y), f"Blocked {bk}", bc)
        y += 20

        self.ft_label.render_to(self.screen, (x, y), "CIVILIANS", TXT_DIM)
        y += 16
        saved = len(self.sim.router.civilians_saved)
        total = NUM_CIVILIANS
        cc    = GREEN if saved == total else ORANGE
        self.ft_small.render_to(self.screen, (x, y), f"{saved} / {total} rescued", cc)
        y += 16
        deferred = len(self.sim.router.deferred)
        if deferred:
            self.ft_small.render_to(self.screen, (x, y), f"{deferred} deferred", AMBER)
            y += 16

        self.ft_label.render_to(self.screen, (x, y), "POLICE (C5)", TXT_DIM)
        y += 16
        total_officers = sum(self.sim.police_deployment.values())
        nodes_covered  = len(self.sim.police_deployment)
        self.ft_small.render_to(self.screen, (x, y),
                                f"{total_officers} officers / {nodes_covered} nodes", BLUE)
        y += 20

        pygame.draw.line(self.screen, BORDER, (x, y), (sr.right-12, y))
        y += 8
        self.ft_label.render_to(self.screen, (x, y), "CSP CONSTRAINTS", TXT_DIM)
        y += 16
        from challenge1csp import CONSTRAINTS
        self.ft_small.render_to(self.screen, (x, y),
            f"C1 excl radius : {CONSTRAINTS['C1_EXCLUSION_RADIUS']} hop(s)", TXT_PRI)
        y += 18
        self.ft_small.render_to(self.screen, (x, y),
            f"C2 hosp reach  : {CONSTRAINTS['C2_HOSPITAL_HOPS']} hop(s) [soft]", TXT_SEC)
        y += 18
        self.ft_small.render_to(self.screen, (x, y),
            f"C3 indus reach : {CONSTRAINTS['C3_INDUSTRIAL_HOPS']} hop(s)", TXT_PRI)
        y += 20

        pygame.draw.line(self.screen, BORDER, (x, y), (sr.right-12, y))
        y += 8
        self.ft_label.render_to(self.screen, (x, y), "SIM EVENTS", TXT_DIM)
        y += 16
        fc2 = RED_COL if self.sim.flood_count > 0 else TXT_PRI
        self.ft_small.render_to(self.screen, (x, y),
            f"Roads flooded  : {self.sim.flood_count}", fc2)
        y += 14
        rescued = len(self.sim.router.civilians_saved)
        rc = GREEN if rescued == NUM_CIVILIANS else ORANGE
        self.ft_small.render_to(self.screen, (x, y),
            f"Rescued        : {rescued} / {NUM_CIVILIANS}", rc)


    def _draw_timeline(self):
        if not self.sim.ready:
            return
        bx = GRID_X
        bx = GRID_X
        by = WIN_H - BOT_H + 55 + 4
        tw = GRID_PX

        pygame.draw.rect(self.screen, BORDER, (bx, by, tw, 8), border_radius=4)

        event_map = {}

        priority = {
            "move": 1,
            "rescue": 2,
            "flood": 3
        }

        for etype, step in self.sim.timeline:
            if step not in event_map:
                event_map[step] = etype
            else:
                current = event_map[step]

                if priority[etype] > priority[current]:
                    event_map[step] = etype

        step_w = tw / MAX_STEPS
        for s in range(1, MAX_STEPS + 1):
            etype = event_map.get(s)
            col   = (GREEN    if etype == "rescue"
                     else RED_COL if etype == "flood"
                     else BLUE    if etype == "move"
                     else BORDER)
            x = int(bx + (s - 1) * step_w)
            w = max(1, int(step_w) - 1)
            pygame.draw.rect(self.screen, col, (x, by, w, 8), border_radius=2)

        if self.sim.step > 0:
            cx = int(bx + (self.sim.step - 1) * step_w)
            pygame.draw.rect(self.screen, TXT_PRI, (cx, by - 2, max(2, int(step_w)), 12), border_radius=3)

        self.ft_label.render_to(self.screen, (bx, by + 10), "MISSION TIMELINE", TXT_DIM)
        self.ft_label.render_to(self.screen, (bx + 125, by + 10), "■ rescue", GREEN)
        self.ft_label.render_to(self.screen, (bx + 205, by + 10), "■ flood", RED_COL)
        self.ft_label.render_to(self.screen, (bx + 265, by + 10), "■ move", BLUE)


    def _draw_completion_banner(self):
        if not self.sim.done or not self.sim.ready:
            return
        saved   = len(self.sim.router.civilians_saved)
        total   = NUM_CIVILIANS
        floods  = self.sim.flood_count
        all_ok  = saved == total

        bw, bh = 340, 120
        bx = GRID_X + (GRID_PX - bw) // 2
        by = GRID_Y  + (GRID_PX - bh) // 2

        surf = pygame.Surface((bw, bh), pygame.SRCALPHA)
        surf.fill((255, 255, 255, 220))
        self.screen.blit(surf, (bx, by))
        pygame.draw.rect(self.screen, GREEN if all_ok else AMBER, (bx, by, bw, bh), 2, border_radius=10)

        title   = "✓ MISSION COMPLETE" if all_ok else "⚠ MISSION ENDED"
        title_c = GREEN if all_ok else AMBER
        tw      = self.ft_step.get_rect(title).width
        self.ft_step.render_to(self.screen, (bx + (bw - tw) // 2, by + 14), title, title_c)

        lines = [
            f"Civilians rescued : {saved} / {total}",
            f"Roads flooded     : {floods}",
            f"Steps taken       : {self.sim.step} / {MAX_STEPS}",
            f"Network cost      : {self.sim.c2_cost:.1f}  |  Flow: {self.sim.c2_flow} paths",
        ]
        for i, ln in enumerate(lines):
            self.ft_small.render_to(self.screen, (bx + 20, by + 52 + i * 16), ln, TXT_PRI)


    def _draw_loading(self):
        cx = GRID_X + GRID_PX // 2
        cy = GRID_Y + GRID_PX // 2
        tw = self.ft_step.get_rect("Building city…").width
        self.ft_step.render_to(self.screen, (cx - tw//2, cy-20), "Building city…", BLUE)
        sw = self.ft_body.get_rect(self.sim.status_msg).width
        self.ft_body.render_to(self.screen, (cx - sw//2, cy+14), self.sim.status_msg, TXT_SEC)


    def _draw_grid(self):
        gr = pygame.Rect(GRID_X-6, GRID_Y-6, GRID_PX+12, GRID_PX+12)
        shadow_rect(self.screen, gr, radius=8, blur=5, alpha=16)
        pygame.draw.rect(self.screen, PANEL_BG, gr, border_radius=8)
        pygame.draw.rect(self.screen, BORDER,   gr, 1, border_radius=8)

        for (r, c), node in self.sim.graph.nodes.items():
            rect  = cell_rect(r, c)
            col   = NODE_COL.get(node.location_type, NODE_COL[None])
            bcol  = NODE_BORDER.get(node.location_type, BORDER)
            tcol  = NODE_TXT.get(node.location_type, TXT_SEC)

            pygame.draw.rect(self.screen, col,  rect, border_radius=4)
            pygame.draw.rect(self.screen, bcol, rect, 1, border_radius=4)

            lbl  = NODE_LBL.get(node.location_type, "?")
            tw   = self.ft_node.get_rect(lbl).width
            th   = self.ft_node.get_rect(lbl).height
            self.ft_node.render_to(self.screen,
                                   (rect.centerx - tw//2, rect.centery - th//2),
                                   lbl, tcol)


    def _draw_roads_overlay(self):
        drawn = set()
        for key, edge in self.sim.graph.edges.items():
            pair = (edge.node_a, edge.node_b)
            if pair in drawn: continue
            drawn.add(pair); drawn.add((edge.node_b, edge.node_a))

            ax, ay = cell_center(*edge.node_a)
            bx, by = cell_center(*edge.node_b)

            if edge.is_blocked:
                pygame.draw.line(self.screen, RED_COL, (ax,ay), (bx,by), 2)
                mx, my = (ax+bx)//2, (ay+by)//2
                tw = self.ft_node.get_rect("✕").width
                self.ft_node.render_to(self.screen, (mx-tw//2, my-7), "✕", RED_COL)
            else:
                t   = min(edge.effective_cost / 2.0, 1.0)
                col = lerp_color(GREEN, TXT_DIM, t)
                pygame.draw.line(self.screen, col, (ax,ay), (bx,by), 1)


    def _draw_coverage_overlay(self):
        if not self.sim.amb_pos:
            return
        max_d = max(
            (self.sim.coverage_dist(nid) for nid in self.sim.graph.nodes
             if self.sim.coverage_dist(nid) != math.inf),
            default=1.0
        )
        if max_d == 0:
            return
        for (r, c) in self.sim.graph.nodes:
            d = self.sim.coverage_dist((r, c))
            if d == math.inf: continue
            t    = min(d / max_d, 1.0)
            tint = lerp_color(GREEN, RED_COL, t)
            blend_surf(self.screen, cell_rect(r, c), tint, 80)


    def _draw_crime_overlay(self):
        tints  = {CrimeRisk.LOW: None, CrimeRisk.MEDIUM: (180,120,0), CrimeRisk.HIGH: (200,30,30)}
        alphas = {CrimeRisk.LOW: 0,    CrimeRisk.MEDIUM: 80,          CrimeRisk.HIGH: 120}
        for (r, c), node in self.sim.graph.nodes.items():
            risk  = node.crime_risk_level
            tint  = tints[risk]
            alpha = alphas[risk]
            if tint and alpha:
                blend_surf(self.screen, cell_rect(r, c), tint, alpha)
                if risk == CrimeRisk.HIGH:
                    rect = cell_rect(r, c)
                    tw   = self.ft_label.get_rect("▲").width
                    self.ft_label.render_to(self.screen,
                                            (rect.centerx-tw//2, rect.bottom-14),
                                            "▲", RED_COL)

        for node_id, count in self.sim.police_deployment.items():
            r, c  = node_id
            rect  = cell_rect(r, c)
            bx    = rect.right - 18
            by    = rect.top   + 10
            pygame.draw.circle(self.screen, BLUE,     (bx, by), 8)
            pygame.draw.circle(self.screen, PANEL_BG, (bx, by), 8, 1)
            badge = str(count)
            tw    = self.ft_label.get_rect(badge).width
            self.ft_label.render_to(self.screen, (bx - tw//2, by - 6), badge, PANEL_BG)


    def _draw_agents(self):
        router = self.sim.router

        for civ in router.civilian_queue:
            cx, cy = cell_center(*civ)
            pygame.draw.circle(self.screen, PANEL_BG, (cx, cy), 11)
            pygame.draw.circle(self.screen, ORANGE,   (cx, cy), 11, 2)
            tw = self.ft_node.get_rect("C").width
            self.ft_node.render_to(self.screen, (cx-tw//2, cy-7), "C", ORANGE)

        for pos in self.sim.amb_pos:
            ax, ay = cell_center(*pos)
            pygame.draw.circle(self.screen, PANEL_BG, (ax, ay), 12)
            pygame.draw.circle(self.screen, GREEN,    (ax, ay), 12, 2)
            tw = self.ft_node.get_rect("+").width
            self.ft_node.render_to(self.screen, (ax-tw//2, ay-7), "+", GREEN)

        if router.current_path:
            prev = router.team_position
            for nxt in router.current_path:
                px, py = cell_center(*prev)
                nx, ny = cell_center(*nxt)
                total  = math.hypot(nx-px, ny-py)
                steps  = max(int(total // 7), 1)
                for i in range(steps):
                    if i % 2 == 0:
                        x1 = int(px + (nx-px)*i/steps)
                        y1 = int(py + (ny-py)*i/steps)
                        x2 = int(px + (nx-px)*(i+1)/steps)
                        y2 = int(py + (ny-py)*(i+1)/steps)
                        pygame.draw.line(self.screen, BLUE, (x1,y1), (x2,y2), 2)
                prev = nxt

        tx, ty = cell_center(*router.team_position)
        pygame.draw.circle(self.screen, AMBER,    (tx, ty), 13)
        pygame.draw.circle(self.screen, PANEL_BG, (tx, ty), 13, 2)
        tw = self.ft_node.get_rect("M").width
        self.ft_node.render_to(self.screen, (tx-tw//2, ty-7), "M", PANEL_BG)


    def _draw_log_panel(self):
        lx = GRID_X + GRID_PX + 12
        ly = TOP_H + 4
        lw = LOG_W - 8
        lh = WIN_H - TOP_H - BOT_H - 8

        lr = pygame.Rect(lx, ly, lw, lh)
        shadow_rect(self.screen, lr, radius=8, blur=4, alpha=14)
        pygame.draw.rect(self.screen, PANEL_BG, lr, border_radius=8)
        pygame.draw.rect(self.screen, BORDER,   lr, 1, border_radius=8)

        self.ft_body.render_to(self.screen, (lx+12, ly+8), "EVENT LOG", TXT_DIM)
        pygame.draw.line(self.screen, BORDER, (lx+8, ly+28), (lx+lw-8, ly+28))

        if not self.sim.log:
            return

        entries  = self.sim.log.entries
        line_h   = 52
        visible  = (lh - 38) // line_h
        start    = max(0, min(self.log_scroll, max(0, len(entries)-visible)))
        self.log_scroll = start

        LOG_ICON = {
            "INFO":     ("ℹ", BLUE),
            "WARNING":  ("⚠", AMBER),
            "CRITICAL": ("✖", RED_COL),
        }
        LOG_BG = {
            "INFO":     None,
            "WARNING":  rgb("FFFAE5"),
            "CRITICAL": rgb("FEECEC"),
        }

        clip = pygame.Rect(lx+4, ly+32, lw-8, lh-38)
        self.screen.set_clip(clip)

        for i, entry in enumerate(entries[start: start+visible]):
            ey  = ly + 34 + i * line_h
            lvl = entry["level"]
            icon, icol = LOG_ICON.get(lvl, ("ℹ", TXT_SEC))
            bg         = LOG_BG.get(lvl)

            if bg:
                row_r = pygame.Rect(lx+6, ey, lw-12, line_h-2)
                pygame.draw.rect(self.screen, bg, row_r, border_radius=4)

            badge = f"S{entry['step']}"
            self.ft_body.render_to(self.screen, (lx+10, ey+4),  badge, TXT_DIM)
            self.ft_body.render_to(self.screen, (lx+10, ey+22), icon,  icol)

            msg   = entry["message"]
            line1 = msg[:30]
            line2 = msg[30:60] if len(msg) > 30 else ""
            msg_x = lx + 48

            self.ft_body.render_to(self.screen, (msg_x, ey+4), line1, TXT_PRI)

            if line2:
                self.ft_body.render_to(self.screen, (msg_x, ey+24), line2, TXT_SEC)
        self.screen.set_clip(None)

        if len(entries) > visible:
            track_h = lh - 38
            bar_h   = max(24, int(track_h * visible / max(len(entries), 1)))
            bar_y   = ly+32 + int(track_h * start / max(len(entries)-visible, 1))
            pygame.draw.rect(self.screen, BORDER,
                             (lx+lw-6, bar_y, 4, bar_h), border_radius=2)


    def run(self):
        while True:
            self.handle_events()
            self._tick()
            self.draw()
            self.clock.tick(30)


if __name__ == "__main__":
    CityMindApp().run()