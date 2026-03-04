import json
import os
import copy
import csv
import argparse
from typing import Dict, List, Tuple
import random

# Import from your simulator file
from popup_sim import PopupSimulator


DEFAULT_LAYOUT_TEMPLATE = {
    "meta": {"units": "ft", "cell_size": 1.0},
    "space": {"width": 30, "height": 40},
    "entry": {"wall": "south", "offset": 10, "width": 4},
    "exit": {"wall": "north", "offset": 18, "width": 4},
    "modules": [
        {"id": "m1", "type": "photo_booth", "x": 22, "y": 8, "w": 6, "h": 6, "rot": 0},
        {"id": "m2", "type": "merch_display", "x": 10, "y": 25, "w": 10, "h": 4, "rot": 90},
        {"id": "m3", "type": "sample_station", "x": 6, "y": 12, "w": 4, "h": 4, "rot": 0},
        {"id": "m4", "type": "ar_vr", "x": 20, "y": 30, "w": 6, "h": 6, "rot": 0},
        {"id": "m5", "type": "digital_display", "x": 12, "y": 3, "w": 6, "h": 2, "rot": 0},
        {"id": "m6", "type": "physical_experience", "x": 2, "y": 28, "w": 10, "h": 10, "rot": 0}
    ]
}


# -----------------------------
# Helpers
# -----------------------------
def auto_fix_layout(layout: dict,
                    max_iters: int = 500,
                    max_nudge_cells: int = 3,
                    seed: int = 0,
                    config: dict = None) -> dict:
    """
    Attempt to fix overlaps by nudging randomly selected modules a small amount
    until validate_layout_fast passes.

    Strategy:
      - If overlap exists, randomly pick one of the overlapping modules and nudge it.
      - Repeat until valid or max_iters.
    """
    rng = random.Random(seed)
    cell_size = float(layout["meta"]["cell_size"])
    fixed = copy.deepcopy(layout)

    for _ in range(max_iters):
        try:
            if config is None:
                validate_layout_fast(fixed)
            else:
                validate_layout_with_sim(fixed, config, seed=seed)
            return fixed
        except ValueError as e:
            msg = str(e)
            if "Overlap between" in msg:
                parts = msg.split()
                a = parts[2]
                b = parts[4]
                move_id = a if rng.random() < 0.5 else b

                idx = None
                for i, m in enumerate(fixed["modules"]):
                    if m["id"] == move_id:
                        idx = i
                        break
                if idx is None:
                    continue

                W, H = layout_to_grid_dims(fixed)
                dx = rng.randint(-max_nudge_cells, max_nudge_cells)
                dy = rng.randint(-max_nudge_cells, max_nudge_cells)
                if dx == 0 and dy == 0:
                    dx = 1

                new_x = fixed["modules"][idx]["x"] + dx * cell_size
                new_y = fixed["modules"][idx]["y"] + dy * cell_size

                w_ft = fixed["modules"][idx]["w"]
                h_ft = fixed["modules"][idx]["h"]
                max_x_ft = (W * cell_size) - w_ft
                max_y_ft = (H * cell_size) - h_ft

                fixed["modules"][idx]["x"] = float(clamp_int(int(round(new_x)), 0, int(round(max_x_ft))))
                fixed["modules"][idx]["y"] = float(clamp_int(int(round(new_y)), 0, int(round(max_y_ft))))

            else:
                i = rng.randrange(len(fixed["modules"]))
                W, H = layout_to_grid_dims(fixed)
                dx = rng.randint(-max_nudge_cells, max_nudge_cells)
                dy = rng.randint(-max_nudge_cells, max_nudge_cells)
                new_x = fixed["modules"][i]["x"] + dx * cell_size
                new_y = fixed["modules"][i]["y"] + dy * cell_size
                w_ft = fixed["modules"][i]["w"]
                h_ft = fixed["modules"][i]["h"]
                max_x_ft = (W * cell_size) - w_ft
                max_y_ft = (H * cell_size) - h_ft
                fixed["modules"][i]["x"] = float(clamp_int(int(round(new_x)), 0, int(round(max_x_ft))))
                fixed["modules"][i]["y"] = float(clamp_int(int(round(new_y)), 0, int(round(max_y_ft))))

    raise RuntimeError("auto_fix_layout: couldn't fix layout within max_iters. Increase space or max_nudge_cells.")


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def layout_to_grid_dims(layout: dict) -> Tuple[int, int]:
    cell_size = float(layout["meta"]["cell_size"])
    W = int(round(layout["space"]["width"] / cell_size))
    H = int(round(layout["space"]["height"] / cell_size))
    return W, H


def modules_to_occupied_cells(layout: dict) -> Dict[Tuple[int, int], str]:
    """Returns a dict mapping occupied grid cell -> module_id."""
    cell_size = float(layout["meta"]["cell_size"])
    occ = {}
    for m in layout["modules"]:
        mid = m["id"]
        x = int(round(m["x"] / cell_size))
        y = int(round(m["y"] / cell_size))
        w = int(round(m["w"] / cell_size))
        h = int(round(m["h"] / cell_size))
        for cx in range(x, x + w):
            for cy in range(y, y + h):
                if (cx, cy) in occ:
                    raise ValueError(f"Overlap between {occ[(cx, cy)]} and {mid} at {(cx, cy)}")
                occ[(cx, cy)] = mid
    return occ


def validate_layout_fast(layout: dict) -> None:
    """Fast geometry checks: bounds and overlaps (perimeter walls handled in simulator)."""
    cell_size = float(layout["meta"]["cell_size"])
    W, H = layout_to_grid_dims(layout)

    for m in layout["modules"]:
        x = int(round(m["x"] / cell_size))
        y = int(round(m["y"] / cell_size))
        w = int(round(m["w"] / cell_size))
        h = int(round(m["h"] / cell_size))
        if x < 0 or y < 0 or x + w > W or y + h > H:
            raise ValueError(f"Module {m['id']} out of bounds after move.")

    _ = modules_to_occupied_cells(layout)


def validate_layout_with_sim(layout: dict, config: dict, seed: int) -> None:
    """Validate against fast geometry checks + simulator constraints."""
    validate_layout_fast(layout)
    _ = PopupSimulator(layout, config, seed=seed)


def repair_layout_to_valid(layout: dict,
                           config: dict,
                           seed: int,
                           max_attempts: int = 6) -> dict:
    """
    Ensure a layout passes simulator validation.
    If it fails, repeatedly auto-fix and perturb until it passes.
    """
    candidate = copy.deepcopy(layout)

    for attempt in range(max_attempts):
        try:
            validate_layout_with_sim(candidate, config, seed=seed + attempt)
            return candidate
        except Exception as e:
            print(f"Layout validation failed (attempt {attempt + 1}/{max_attempts}): {e}")

        candidate = auto_fix_layout(
            candidate,
            max_iters=700,
            max_nudge_cells=4,
            seed=seed + attempt * 17,
            config=config
        )

        if attempt < max_attempts - 1:
            candidate = perturb_layout(
                candidate,
                max_move_cells=4,
                tries=300,
                allow_swap=True,
                swap_prob=0.25,
                seed=seed + attempt * 31
            )

    raise RuntimeError("Unable to repair layout to satisfy simulator validation constraints.")


def score_outputs(outputs: dict, weights: dict) -> float:
    conv = float(outputs["conversion_rate"])
    entry = float(outputs["entry_rate"])
    dwell = float(outputs["avg_dwell_time_min"])
    social = float(outputs["social_per_entrant"])

    target_dwell = float(weights.get("target_dwell_min", 10.0))
    target_social = float(weights.get("target_social", 50.0))

    dwell_n = max(0.0, min(1.0, dwell / max(1e-6, target_dwell)))
    social_n = max(0.0, min(1.0, social / max(1e-6, target_social)))

    w_conv = float(weights.get("conversion_rate", 0.4))
    w_entry = float(weights.get("entry_rate", 0.2))
    w_dwell = float(weights.get("dwell_time", 0.2))
    w_social = float(weights.get("social_engagement", 0.2))

    return w_conv * conv + w_entry * entry + w_dwell * dwell_n + w_social * social_n


# -----------------------------
# Layout generation
# -----------------------------
def randomize_opening(opening: dict, layout: dict, rng: random.Random) -> dict:
    """Small randomization around an opening while staying on the same wall."""
    out = copy.deepcopy(opening)
    wall = out["wall"]
    cell_size = float(layout["meta"]["cell_size"])
    W, H = layout_to_grid_dims(layout)
    wall_len_cells = W if wall in ("south", "north") else H

    width_cells = int(round(float(out["width"]) / cell_size))
    width_cells = clamp_int(width_cells + rng.randint(-1, 1), 2, max(2, wall_len_cells - 2))

    offset_cells = int(round(float(out["offset"]) / cell_size))
    max_offset = max(0, wall_len_cells - width_cells)
    offset_cells = clamp_int(offset_cells + rng.randint(-4, 4), 0, max_offset)

    out["width"] = float(width_cells * cell_size)
    out["offset"] = float(offset_cells * cell_size)
    return out


def generate_random_layout(template_layout: dict,
                           config: dict,
                           seed: int,
                           tries: int = 200) -> dict:
    """
    Generate a randomized layout dictionary from a template.
    Randomizes module type, rotation, and position while preserving module sizes.
    """
    rng = random.Random(seed)
    module_types = list(config["module_params"].keys())
    if not module_types:
        raise ValueError("config.module_params is empty; cannot assign module types.")

    for _ in range(tries):
        candidate = copy.deepcopy(template_layout)
        cell_size = float(candidate["meta"]["cell_size"])
        W, H = layout_to_grid_dims(candidate)

        candidate["entry"] = randomize_opening(candidate["entry"], candidate, rng)
        candidate["exit"] = randomize_opening(candidate["exit"], candidate, rng)

        occupied = set()
        placed = []

        for base_mod in candidate["modules"]:
            mod = copy.deepcopy(base_mod)
            mod["type"] = rng.choice(module_types)
            mod["rot"] = rng.choice([0, 90, 180, 270])

            w_cells = int(round(float(mod["w"]) / cell_size))
            h_cells = int(round(float(mod["h"]) / cell_size))

            if w_cells <= 0 or h_cells <= 0:
                raise ValueError(f"Invalid module size for {mod['id']}")
            if w_cells >= W - 1 or h_cells >= H - 1:
                raise ValueError(f"Module {mod['id']} too large for current space")

            ok = False
            for _ in range(250):
                x = rng.randint(1, (W - w_cells - 1))
                y = rng.randint(1, (H - h_cells - 1))
                cells = [(cx, cy) for cx in range(x, x + w_cells) for cy in range(y, y + h_cells)]
                if any(c in occupied for c in cells):
                    continue

                mod["x"] = float(x * cell_size)
                mod["y"] = float(y * cell_size)
                for c in cells:
                    occupied.add(c)
                placed.append(mod)
                ok = True
                break

            if not ok:
                break

        if len(placed) != len(candidate["modules"]):
            continue

        candidate["modules"] = placed

        try:
            validate_layout_with_sim(candidate, config, seed=seed)
            return candidate
        except Exception:
            continue

    raise RuntimeError("Failed to generate a valid random layout from template.")


# -----------------------------
# Perturbations
# -----------------------------
def perturb_layout(layout: dict,
                   max_move_cells: int,
                   tries: int = 200,
                   allow_swap: bool = True,
                   swap_prob: float = 0.15,
                   seed: int = 0) -> dict:
    """
    Create a NEW layout by randomly moving one module (or swapping two).
    Ensures valid geometry (bounds + no overlap).
    max_move_cells is in grid cells.
    """
    rng = random.Random(seed)
    cell_size = float(layout["meta"]["cell_size"])

    base = copy.deepcopy(layout)
    W, H = layout_to_grid_dims(base)

    for _ in range(tries):
        candidate = copy.deepcopy(base)
        mods = candidate["modules"]

        if allow_swap and rng.random() < swap_prob and len(mods) >= 2:
            a, b = rng.sample(range(len(mods)), 2)
            mods[a]["x"], mods[b]["x"] = mods[b]["x"], mods[a]["x"]
            mods[a]["y"], mods[b]["y"] = mods[b]["y"], mods[a]["y"]
        else:
            i = rng.randrange(len(mods))
            dx = rng.randint(-max_move_cells, max_move_cells)
            dy = rng.randint(-max_move_cells, max_move_cells)

            new_x = mods[i]["x"] + dx * cell_size
            new_y = mods[i]["y"] + dy * cell_size

            w_ft = mods[i]["w"]
            h_ft = mods[i]["h"]
            max_x_ft = (W * cell_size) - w_ft
            max_y_ft = (H * cell_size) - h_ft

            mods[i]["x"] = float(clamp_int(int(round(new_x)), 0, int(round(max_x_ft))))
            mods[i]["y"] = float(clamp_int(int(round(new_y)), 0, int(round(max_y_ft))))

        try:
            validate_layout_fast(candidate)
            return candidate
        except Exception:
            continue

    raise RuntimeError("Failed to generate a valid perturbed layout; try smaller max_move_cells or more space.")


# -----------------------------
# Batch runner
# -----------------------------
def run_one_layout(layout_dict: dict, config_dict: dict, seed: int) -> dict:
    sim = PopupSimulator(layout_dict, config_dict, seed=seed)
    result = sim.run()
    passersby_total = float(result["inputs"]["passersby_total"])
    entrants_total = float(result["outputs"]["foot_traffic_inside"])
    entry_rate = (entrants_total / passersby_total) if passersby_total > 0 else 0.0

    result["outputs"]["entry_rate"] = entry_rate
    return result


def average_metrics(results: List[dict]) -> dict:
    out_keys = ["conversion_rate", "entry_rate", "avg_dwell_time_min", "social_per_entrant"]
    avg = {k: 0.0 for k in out_keys}
    for r in results:
        for k in out_keys:
            avg[k] += float(r["outputs"][k])
    n = len(results)
    for k in out_keys:
        avg[k] /= max(1, n)
    return avg


def load_layout_template(args_layout_template_json: str) -> dict:
    if not args_layout_template_json:
        return copy.deepcopy(DEFAULT_LAYOUT_TEMPLATE)

    if os.path.exists(args_layout_template_json):
        with open(args_layout_template_json, "r") as f:
            return json.load(f)

    return json.loads(args_layout_template_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="", help="Optional base layout.json path (if omitted, random layouts are generated)")
    ap.add_argument("--layout_template_json", default="", help="Optional template JSON file path or raw JSON string for random mode")
    ap.add_argument("--config", required=True, help="Path to config.json (required)")
    ap.add_argument("--out_csv", default="batch_results.csv", help="CSV output path")
    ap.add_argument("--save_variants_dir", default="", help="Optional folder to save generated layout JSONs")
    ap.add_argument("--n_layouts", type=int, default=30, help="Number of layouts to evaluate")
    ap.add_argument("--runs_per_layout", type=int, default=5, help="Monte Carlo runs per layout (different seeds)")
    ap.add_argument("--max_move_cells", type=int, default=3, help="Max grid-cell move in x/y for perturbation (file-layout mode)")
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--swap_prob", type=float, default=0.15)
    ap.add_argument("--no_swap", action="store_true")
    ap.add_argument("--weights_json", default="", help="Optional scoring weights JSON string or file path")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    weights = {
        "conversion_rate": 0.4,
        "entry_rate": 0.2,
        "dwell_time": 0.2,
        "social_engagement": 0.2,
        "target_dwell_min": 10.0,
        "target_social": 50.0
    }
    if args.weights_json:
        if os.path.exists(args.weights_json):
            with open(args.weights_json, "r") as wf:
                weights.update(json.load(wf))
        else:
            weights.update(json.loads(args.weights_json))

    if args.save_variants_dir:
        os.makedirs(args.save_variants_dir, exist_ok=True)

    rows = []

    def eval_layout(layout_dict: dict, tag: str, layout_seed: int):
        run_results = []
        for r in range(args.runs_per_layout):
            seed = args.base_seed + layout_seed * 1000 + r
            run_results.append(run_one_layout(layout_dict, config, seed=seed))
        avg = average_metrics(run_results)
        sc = score_outputs(avg, weights)

        rows.append({"tag": tag, "score": sc, **avg})

    if args.layout:
        with open(args.layout, "r") as f:
            base_layout = json.load(f)

        try:
            validate_layout_with_sim(base_layout, config, seed=args.base_seed)
        except Exception as e:
            print(f"Base layout invalid ({e}). Attempting repair...")
            base_layout = repair_layout_to_valid(base_layout, config, seed=args.base_seed)
            with open("layout_FIXED.json", "w") as f:
                json.dump(base_layout, f, indent=2)
            print("Wrote layout_FIXED.json (repaired baseline layout).")

        eval_layout(base_layout, tag="baseline", layout_seed=0)

        for i in range(1, args.n_layouts + 1):
            var = perturb_layout(
                base_layout,
                max_move_cells=args.max_move_cells,
                allow_swap=(not args.no_swap),
                swap_prob=args.swap_prob,
                seed=args.base_seed + i
            )

            var = repair_layout_to_valid(var, config, seed=args.base_seed + i)

            if args.save_variants_dir:
                out_path = os.path.join(args.save_variants_dir, f"layout_variant_{i:03d}.json")
                with open(out_path, "w") as f:
                    json.dump(var, f, indent=2)

            eval_layout(var, tag=f"variant_{i:03d}", layout_seed=i)

            if i % 5 == 0:
                print(f"Evaluated {i}/{args.n_layouts} variants...")

    else:
        layout_template = load_layout_template(args.layout_template_json)

        baseline = generate_random_layout(layout_template, config, seed=args.base_seed)
        baseline = repair_layout_to_valid(baseline, config, seed=args.base_seed)
        eval_layout(baseline, tag="baseline_random", layout_seed=0)

        if args.save_variants_dir:
            baseline_path = os.path.join(args.save_variants_dir, "layout_variant_000.json")
            with open(baseline_path, "w") as f:
                json.dump(baseline, f, indent=2)

        for i in range(1, args.n_layouts + 1):
            var = generate_random_layout(layout_template, config, seed=args.base_seed + i)
            var = repair_layout_to_valid(var, config, seed=args.base_seed + i)

            if args.save_variants_dir:
                out_path = os.path.join(args.save_variants_dir, f"layout_variant_{i:03d}.json")
                with open(out_path, "w") as f:
                    json.dump(var, f, indent=2)

            eval_layout(var, tag=f"variant_{i:03d}", layout_seed=i)

            if i % 5 == 0:
                print(f"Generated + evaluated {i}/{args.n_layouts} random variants...")

    rows.sort(key=lambda r: r["score"], reverse=True)

    fieldnames = ["tag", "score", "conversion_rate", "entry_rate", "avg_dwell_time_min", "social_per_entrant"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\nWrote results to: {args.out_csv}")
    print("Top 5:")
    for r in rows[:5]:
        print(
            f"  {r['tag']}: score={r['score']:.4f}  conv={r['conversion_rate']:.3f}  "
            f"entry={r['entry_rate']:.3f}  dwell={r['avg_dwell_time_min']:.2f}  social={r['social_per_entrant']:.2f}"
        )


if __name__ == "__main__":
    main()
