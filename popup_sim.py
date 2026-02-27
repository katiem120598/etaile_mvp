import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict

# ----------------------------
# Utility helpers
# ----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def softmax(weights: List[float], temperature: float = 1.0) -> List[float]:
    # numerically stable softmax
    if not weights:
        return []
    t = max(1e-6, temperature)
    m = max(weights)
    exps = [math.exp((w - m) / t) for w in weights]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [e / s for e in exps]

def sample_poisson(lam: float) -> int:
    # Knuth algorithm: fine for MVP and moderate lam
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Module:
    id: str
    type: str
    x: int
    y: int
    w: int
    h: int
    rot: int = 0

    def cells(self) -> List[Tuple[int,int]]:
        # rectangle occupancy in grid cells: [x, x+w-1] x [y, y+h-1]
        return [(cx, cy) for cx in range(self.x, self.x + self.w)
                        for cy in range(self.y, self.y + self.h)]

    def center(self) -> Tuple[int,int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

@dataclass
class Visitor:
    id: int
    archetype: str
    pos: Tuple[int,int]
    state: str  # "moving", "dwelling", "queued", "exiting", "done"
    target_module_id: Optional[str]
    remaining_dwell_steps: int
    entered_step: int
    exited_step: Optional[int]
    converted: bool
    shared: float  # expected shares accumulator

# ----------------------------
# Core simulator
# ----------------------------

class PopupSimulator:
    """
    MVP pop-up simulator:
      - grid-based geometry
      - passersby -> entrants
      - agent chooses modules via utility = attractor * archetype_mult * distance_decay * queue_penalty
      - capacity + queue modeled per module
      - simple pathing via BFS next-step (costly but fine for MVP footprints)
    """

    def __init__(self, layout: dict, config: dict, seed: int = 0):
        random.seed(seed)

        self.layout = layout
        self.config = config

        self.units = layout["meta"]["units"]
        self.cell_size = float(layout["meta"]["cell_size"])

        self.W = int(round(layout["space"]["width"] / self.cell_size))
        self.H = int(round(layout["space"]["height"] / self.cell_size))

        self.step_seconds = int(config["sim_params"]["step_seconds"])
        self.time_horizon_min = int(config["sim_params"]["time_horizon_min"])
        self.steps = int((self.time_horizon_min * 60) / self.step_seconds)

        self.min_aisle_cells = int(round(config["sim_params"]["min_aisle_width_ft"] / self.cell_size))

        # Parse entry/exit
        self.entry_cells = self._segment_opening_to_cells(layout["entry"])
        self.exit_cells = self._segment_opening_to_cells(layout["exit"])

        # Parse modules
        self.modules: Dict[str, Module] = {}
        for m in layout["modules"]:
            mod = Module(
                id=m["id"],
                type=m["type"],
                x=int(round(m["x"] / self.cell_size)),
                y=int(round(m["y"] / self.cell_size)),
                w=int(round(m["w"] / self.cell_size)),
                h=int(round(m["h"] / self.cell_size)),
                rot=int(m.get("rot", 0))
            )
            self.modules[mod.id] = mod

        # Build occupancy grid (0 = walkable, 1 = blocked)
        self.blocked = [[0 for _ in range(self.H)] for _ in range(self.W)]
        self._apply_perimeter_walls_blocking()
        self._apply_modules_blocking()

        # Basic validation
        self._validate_layout()

        # Queues / capacity state
        self.module_params = config["module_params"]
        self.module_in_service: Dict[str, List[int]] = {mid: [] for mid in self.modules}  # visitor ids currently using
        self.module_queue: Dict[str, deque] = {mid: deque() for mid in self.modules}      # visitor ids waiting

        # Visitors
        self.visitors: Dict[int, Visitor] = {}
        self.next_vid = 1

        # Metrics accumulators
        self.passersby_total = 0
        self.entrants_total = 0
        self.exits_total = 0
        self.conversions_total = 0
        self.total_shares = 0.0

        # For dwell time: track those who exit
        self.dwell_times_steps: List[int] = []

    # ----------------------------
    # Geometry / validation
    # ----------------------------

    def _segment_opening_to_cells(self, opening: dict) -> List[Tuple[int,int]]:
        """
        Opening described by wall + offset + width in feet.
        Wall is one of: "south","north","west","east"
        offset is along the wall from the left/bottom corner in feet.
        width is length of opening in feet.
        Returns list of grid cells on the boundary considered 'open'.
        """
        wall = opening["wall"]
        offset_ft = float(opening["offset"])
        width_ft = float(opening["width"])

        offset = int(round(offset_ft / self.cell_size))
        width = int(round(width_ft / self.cell_size))

        cells = []
        if wall == "south":
            y = 0
            for x in range(offset, min(self.W, offset + width)):
                cells.append((x, y))
        elif wall == "north":
            y = self.H - 1
            for x in range(offset, min(self.W, offset + width)):
                cells.append((x, y))
        elif wall == "west":
            x = 0
            for y in range(offset, min(self.H, offset + width)):
                cells.append((x, y))
        elif wall == "east":
            x = self.W - 1
            for y in range(offset, min(self.H, offset + width)):
                cells.append((x, y))
        else:
            raise ValueError(f"Invalid wall: {wall}")
        if not cells:
            raise ValueError("Opening produced zero cells; check offset/width/cell_size.")
        return cells

    def _apply_perimeter_walls_blocking(self):
        # block perimeter; then carve entry/exit openings back open
        for x in range(self.W):
            self.blocked[x][0] = 1
            self.blocked[x][self.H - 1] = 1
        for y in range(self.H):
            self.blocked[0][y] = 1
            self.blocked[self.W - 1][y] = 1

        # carve openings
        for (x, y) in self.entry_cells + self.exit_cells:
            self.blocked[x][y] = 0

    def _apply_modules_blocking(self):
        for mod in self.modules.values():
            for (x, y) in mod.cells():
                if 0 <= x < self.W and 0 <= y < self.H:
                    self.blocked[x][y] = 1

    def _validate_layout(self):
        # In-bounds & overlaps
        occupied = set()
        for mod in self.modules.values():
            if mod.x < 0 or mod.y < 0 or mod.x + mod.w > self.W or mod.y + mod.h > self.H:
                raise ValueError(f"Module {mod.id} out of bounds.")
            for cell in mod.cells():
                if cell in occupied:
                    raise ValueError(f"Module overlap detected at cell {cell}.")
                occupied.add(cell)

        # Entry/exit should be on perimeter (they are by construction)
        # Ensure entry and exit are not identical openings (optional)
        if set(self.entry_cells) == set(self.exit_cells):
            raise ValueError("Entry and exit openings are identical; choose different segments.")

    # ----------------------------
    # Core behavior models
    # ----------------------------

    def _visible_attraction_near_entry(self) -> float:
        """
        MVP proxy: sum of base attractors for modules within a radius of entry.
        Doesn't do true line-of-sight yet.
        """
        r = int(self.config["sim_params"]["entry_visibility_radius_cells"])
        entry_anchor = self.entry_cells[len(self.entry_cells)//2]
        s = 0.0
        for mod in self.modules.values():
            d = manhattan(entry_anchor, mod.center())
            if d <= r:
                s += float(self.module_params[mod.type]["attractor"])
        return s

    def _entry_congestion_score(self) -> float:
        """
        MVP proxy: count of agents currently in the entry neighborhood (radius r).
        """
        r = int(self.config["sim_params"]["entry_congestion_radius_cells"])
        entry_anchor = self.entry_cells[len(self.entry_cells)//2]
        count = 0
        for v in self.visitors.values():
            if v.state not in ("done",):
                if manhattan(v.pos, entry_anchor) <= r:
                    count += 1
        return float(count)

    def _entry_probability(self) -> float:
        """
        P(enter) = logistic( a + b*visible_attraction - c*entry_congestion )
        plus a baseline base_entry_prob.
        """
        base_entry = float(self.config["sim_params"]["base_entry_prob"])

        a = float(self.config["sim_params"]["entry_logit_a"])
        b = float(self.config["sim_params"]["entry_logit_b"])
        c = float(self.config["sim_params"]["entry_logit_c"])

        vis = self._visible_attraction_near_entry()
        cong = self._entry_congestion_score()
        p = logistic(a + b * vis - c * cong)
        # blend with base so it doesn't go crazy
        p = clamp(0.5 * base_entry + 0.5 * p, 0.0, 1.0)
        return p

    def _choose_archetype(self) -> str:
        mix = self.config["visitor_mix"]
        keys = list(mix.keys())
        probs = [float(mix[k]) for k in keys]
        # normalize
        s = sum(probs)
        probs = [p/s for p in probs] if s > 0 else [1.0/len(keys)]*len(keys)
        r = random.random()
        cum = 0.0
        for k, p in zip(keys, probs):
            cum += p
            if r <= cum:
                return k
        return keys[-1]

    def _queue_length(self, module_id: str) -> int:
        return len(self.module_queue[module_id])

    def _queue_penalty(self, archetype: str, module_id: str) -> float:
        """
        Q = exp(-beta * max(0, q - q_free))
        beta comes from archetype (queue aversion)
        q_free from module params
        """
        q = self._queue_length(module_id)
        q_free = int(self.module_params[self.modules[module_id].type].get("queue_free", 2))
        beta = float(self.config["archetypes"][archetype]["queue_beta"])
        return math.exp(-beta * max(0, q - q_free))

    def _distance_decay(self, from_pos: Tuple[int,int], to_pos: Tuple[int,int]) -> float:
        d = manhattan(from_pos, to_pos)
        return 1.0 / (1.0 + float(d))

    def _utility_for_module(self, v: Visitor, module_id: str) -> float:
        mod = self.modules[module_id]
        base_attr = float(self.module_params[mod.type]["attractor"])
        mult = float(self.config["archetypes"][v.archetype]["attraction_mult"].get(mod.type, 1.0))
        dist = self._distance_decay(v.pos, mod.center())
        qpen = self._queue_penalty(v.archetype, module_id)
        # (Visibility factor omitted in MVP; can add later)
        return base_attr * mult * dist * qpen

    # ----------------------------
    # Movement / pathing
    # ----------------------------

    def _neighbors(self, x: int, y: int) -> List[Tuple[int,int]]:
        nbrs = []
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.W and 0 <= ny < self.H and self.blocked[nx][ny] == 0:
                nbrs.append((nx, ny))
        return nbrs

    def _next_step_toward(self, start: Tuple[int,int], goal: Tuple[int,int]) -> Tuple[int,int]:
        """
        BFS to find next step toward goal. MVP-friendly; fine for small grids.
        If goal is blocked (inside module), path to nearest walkable neighbor of goal.
        """
        if self.blocked[goal[0]][goal[1]] == 1:
            # find a nearby walkable cell around goal
            candidates = []
            for dx in range(-1,2):
                for dy in range(-1,2):
                    gx, gy = goal[0]+dx, goal[1]+dy
                    if 0 <= gx < self.W and 0 <= gy < self.H and self.blocked[gx][gy] == 0:
                        candidates.append((gx, gy))
            if candidates:
                goal = min(candidates, key=lambda c: manhattan(start, c))
            else:
                return start

        if start == goal:
            return start

        q = deque([start])
        prev = {start: None}
        found = False
        while q:
            cur = q.popleft()
            if cur == goal:
                found = True
                break
            for nb in self._neighbors(cur[0], cur[1]):
                if nb not in prev:
                    prev[nb] = cur
                    q.append(nb)

        if not found:
            return start

        # backtrack one step
        cur = goal
        while prev[cur] is not None and prev[cur] != start:
            cur = prev[cur]
        return cur if prev[cur] == start else goal

    # ----------------------------
    # Module service / dwell / conversion / sharing
    # ----------------------------

    def _draw_dwell_steps(self, archetype: str, module_type: str) -> int:
        """
        Dwell time draw: normal(mean, sd) clipped to >= 1 step.
        Multiply by archetype dwell_mult.
        """
        mean_s = float(self.module_params[module_type]["dwell_mean_s"])
        sd_s = float(self.module_params[module_type]["dwell_sd_s"])
        dwell_mult = float(self.config["archetypes"][archetype]["dwell_mult"])
        x = random.gauss(mean_s, sd_s)
        x = max(5.0, x)  # at least 5 seconds
        x *= dwell_mult
        steps = int(round(x / self.step_seconds))
        return max(1, steps)

    def _maybe_convert(self, v: Visitor, interacted_types: List[str]) -> bool:
        """
        Simple conversion model:
          p = base_conversion + sum(conversion_lift for each interacted module type)
        Clip to [0,1]. Sample once at exit.
        """
        base = float(self.config["archetypes"][v.archetype]["base_conversion"])
        lift = 0.0
        for t in interacted_types:
            lift += float(self.module_params[t].get("conversion_lift", 0.0))
        p = clamp(base + lift, 0.0, 1.0)
        return random.random() < p

    def _share_increment(self, v: Visitor, module_type: str) -> float:
        """
        Expected share contribution from an interaction:
          share_prob(module) * share_mult(archetype)
        """
        base = float(self.module_params[module_type].get("share_prob", 0.0))
        mult = float(self.config["archetypes"][v.archetype]["share_mult"])
        return base * mult

    # ----------------------------
    # Simulation loop
    # ----------------------------

    def run(self) -> dict:
        # Track per-visitor interacted module types for conversion calculation
        interacted: Dict[int, List[str]] = defaultdict(list)

        passersby_rate = float(self.config["sim_params"]["passersby_rate_per_min"])
        temperature = float(self.config["sim_params"].get("choice_temperature", 1.0))

        entry_anchor = self.entry_cells[len(self.entry_cells)//2]
        exit_anchor = self.exit_cells[len(self.exit_cells)//2]

        for step in range(self.steps):
            # 1) Passersby arrive externally
            n_pass = sample_poisson(passersby_rate * (self.step_seconds / 60.0))
            self.passersby_total += n_pass

            # 2) Each passerby decides to enter based on visibility + entry congestion
            p_enter = self._entry_probability()
            for _ in range(n_pass):
                if random.random() < p_enter:
                    # spawn at entry anchor (or a random entry cell)
                    spawn = random.choice(self.entry_cells)
                    archetype = self._choose_archetype()
                    vid = self.next_vid
                    self.next_vid += 1
                    self.visitors[vid] = Visitor(
                        id=vid,
                        archetype=archetype,
                        pos=spawn,
                        state="moving",
                        target_module_id=None,
                        remaining_dwell_steps=0,
                        entered_step=step,
                        exited_step=None,
                        converted=False,
                        shared=0.0
                    )
                    self.entrants_total += 1

            # 3) Release module service if dwell done; move queued into service if capacity allows
            for mid, service_list in self.module_in_service.items():
                # remove visitors who finished dwelling (their state will be set below in visitor step)
                # capacity refill handled when they transition out
                pass

            # 4) Step each visitor
            for vid, v in list(self.visitors.items()):
                if v.state == "done":
                    continue

                # If dwelling: decrement and potentially leave module
                if v.state == "dwelling":
                    v.remaining_dwell_steps -= 1
                    if v.remaining_dwell_steps <= 0:
                        # finish service at module
                        if v.target_module_id is not None and vid in self.module_in_service[v.target_module_id]:
                            self.module_in_service[v.target_module_id].remove(vid)

                            # move next in queue into service if any
                            cap = int(self.module_params[self.modules[v.target_module_id].type]["capacity"])
                            while len(self.module_in_service[v.target_module_id]) < cap and self.module_queue[v.target_module_id]:
                                next_vid = self.module_queue[v.target_module_id].popleft()
                                nv = self.visitors[next_vid]
                                nv.state = "dwelling"
                                nv.remaining_dwell_steps = self._draw_dwell_steps(nv.archetype, self.modules[v.target_module_id].type)
                                self.module_in_service[v.target_module_id].append(next_vid)

                        # pick next target (or exit)
                        v.state = "moving"
                        v.target_module_id = None
                    continue

                # If queued: do nothing; they’ll get popped into service when capacity frees
                if v.state == "queued":
                    continue

                # If exiting: move toward exit and finish
                if v.state == "exiting":
                    if v.pos in self.exit_cells:
                        v.state = "done"
                        v.exited_step = step
                        self.exits_total += 1
                        dt = step - v.entered_step
                        self.dwell_times_steps.append(dt)

                        # conversion at exit
                        v.converted = self._maybe_convert(v, interacted[vid])
                        if v.converted:
                            self.conversions_total += 1

                        # accumulate shares
                        self.total_shares += v.shared
                    else:
                        nxt = self._next_step_toward(v.pos, exit_anchor)
                        v.pos = nxt
                    continue

                # Otherwise: moving
                if v.state == "moving":
                    # If no target: choose one or decide to exit
                    if v.target_module_id is None:
                        # simple rule: after N interactions, exit; else pick next module
                        # MVP: probabilistic exit decision based on time spent
                        time_inside_steps = step - v.entered_step
                        max_steps = int(self.config["sim_params"]["max_visit_minutes"] * 60 / self.step_seconds)
                        # increasing chance to exit as time grows
                        p_exit = clamp(time_inside_steps / max(1, max_steps), 0.0, 1.0) * 0.4
                        if random.random() < p_exit:
                            v.state = "exiting"
                            continue

                        mids = list(self.modules.keys())
                        utils = [self._utility_for_module(v, mid) for mid in mids]
                        probs = softmax(utils, temperature=temperature)
                        chosen = random.choices(mids, weights=probs, k=1)[0]
                        v.target_module_id = chosen

                    # Move toward target module center
                    target_mod = self.modules[v.target_module_id]
                    goal = target_mod.center()
                    nxt = self._next_step_toward(v.pos, goal)
                    v.pos = nxt

                    # If reached adjacency (near module center), attempt to use it
                    if manhattan(v.pos, goal) <= 1:
                        module_type = target_mod.type
                        cap = int(self.module_params[module_type]["capacity"])

                        if len(self.module_in_service[v.target_module_id]) < cap:
                            # enter service immediately
                            v.state = "dwelling"
                            v.remaining_dwell_steps = self._draw_dwell_steps(v.archetype, module_type)
                            self.module_in_service[v.target_module_id].append(vid)

                            interacted[vid].append(module_type)
                            v.shared += self._share_increment(v, module_type)

                        else:
                            # queue (or skip if too long)
                            q = len(self.module_queue[v.target_module_id])
                            q_max = int(self.module_params[module_type].get("queue_max", 10))
                            if q >= q_max:
                                # skip and choose a new target next step
                                v.target_module_id = None
                            else:
                                v.state = "queued"
                                self.module_queue[v.target_module_id].append(vid)
                    continue

            # Optional: end-of-horizon cleanup (handled after loop)

        # End-of-sim: mark remaining visitors as not exited
        still_inside = sum(1 for v in self.visitors.values() if v.state != "done")

        # Compute metrics
        entrants = self.entrants_total
        exits = self.exits_total
        conversions = self.conversions_total

        conversion_rate = (conversions / entrants) if entrants > 0 else 0.0
        foot_traffic_inside = entrants  # definition: entries
        dwell_time_avg_minutes = 0.0
        if self.dwell_times_steps:
            avg_steps = sum(self.dwell_times_steps) / len(self.dwell_times_steps)
            dwell_time_avg_minutes = (avg_steps * self.step_seconds) / 60.0

        social_engagement = self.total_shares  # expected shares

        return {
            "inputs": {
                "passersby_total": self.passersby_total,
                "passersby_rate_per_min": float(self.config["sim_params"]["passersby_rate_per_min"]),
                "time_horizon_min": self.time_horizon_min,
            },
            "outputs": {
                "foot_traffic_inside": foot_traffic_inside,
                "conversion_rate": conversion_rate,
                "avg_dwell_time_min": dwell_time_avg_minutes,
                "social_engagement_expected_shares": social_engagement,
            },
            "debug": {
                "exits_total": exits,
                "still_inside_end": still_inside
            }
        }


# ----------------------------
# Convenience: load + run
# ----------------------------

def run_from_files(layout_path: str, config_path: str, seed: int = 0) -> dict:
    with open(layout_path, "r") as f:
        layout = json.load(f)
    with open(config_path, "r") as f:
        config = json.load(f)
    sim = PopupSimulator(layout, config, seed=seed)
    return sim.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run_from_files(args.layout, args.config, seed=args.seed)
    print(json.dumps(result, indent=2))