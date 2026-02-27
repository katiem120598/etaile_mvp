import json
import os
import copy
import csv
import argparse
from typing import Dict, List, Tuple
import random

# Import from your simulator file
from popup_sim import PopupSimulator


# -----------------------------
# Helpers
# -----------------------------
def auto_fix_layout(layout: dict,
                    max_iters: int = 500,
                    max_nudge_cells: int = 3,
                    seed: int = 0) -> dict:
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
            validate_layout_fast(fixed)
            return fixed
        except ValueError as e:
            msg = str(e)
            # Example: "Overlap between m2 and m6 at (10, 28)"
            if "Overlap between" in msg:
                parts = msg.split()
                a = parts[2]  # m2
                b = parts[4]  # m6
                # choose one to move (randomly)
                move_id = a if rng.random() < 0.5 else b

                # find that module in list
                idx = None
                for i, m in enumerate(fixed["modules"]):
                    if m["id"] == move_id:
                        idx = i
                        break
                if idx is None:
                    continue

                W, H = layout_to_grid_dims(fixed)

                # nudge in grid cells
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
                # Other errors (bounds etc.) — just random nudge on a random module
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

def modules_to_occupied_cells(layout: dict) -> Dict[Tuple[int,int], str]:
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
                    # overlap
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

    # Overlaps
    _ = modules_to_occupied_cells(layout)

def score_outputs(outputs: dict, weights: dict) -> float:
    """
    Weighted ROE score.
    IMPORTANT: normalize metrics to comparable scales.
    MVP normalization:
      - conversion_rate already 0..1
      - entry_rate already 0..1
      - dwell normalized by / target_dwell (configurable) and clipped to 0..1
      - social normalized by / target_social and clipped to 0..1
    """
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
    modules = base["modules"]
    W, H = layout_to_grid_dims(base)

    for _ in range(tries):
        candidate = copy.deepcopy(base)
        mods = candidate["modules"]

        if allow_swap and rng.random() < swap_prob and len(mods) >= 2:
            a, b = rng.sample(range(len(mods)), 2)
            # swap positions only
            mods[a]["x"], mods[b]["x"] = mods[b]["x"], mods[a]["x"]
            mods[a]["y"], mods[b]["y"] = mods[b]["y"], mods[a]["y"]
        else:
            i = rng.randrange(len(mods))
            dx = rng.randint(-max_move_cells, max_move_cells)
            dy = rng.randint(-max_move_cells, max_move_cells)

            # Convert to feet for JSON by multiplying cell_size
            new_x = mods[i]["x"] + dx * cell_size
            new_y = mods[i]["y"] + dy * cell_size

            # Clamp to bounds in FEET (keeping module size in mind)
            w_ft = mods[i]["w"]
            h_ft = mods[i]["h"]

            # Convert W,H back to feet bounds
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
    # Add entry_rate explicitly
    passersby_total = float(result["inputs"]["passersby_total"])
    entrants_total = float(result["outputs"]["foot_traffic_inside"])
    entry_rate = (entrants_total / passersby_total) if passersby_total > 0 else 0.0

    # Attach entry_rate into outputs for scoring + CSV
    result["outputs"]["entry_rate"] = entry_rate
    return result

def average_metrics(results: List[dict]) -> dict:
    # Average over multiple seeds
    out_keys = ["conversion_rate", "entry_rate", "avg_dwell_time_min", "social_per_entrant"]
    avg = {k: 0.0 for k in out_keys}
    for r in results:
        for k in out_keys:
            avg[k] += float(r["outputs"][k])
    n = len(results)
    for k in out_keys:
        avg[k] /= max(1, n)
    return avg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True, help="Path to base layout.json")
    ap.add_argument("--config", required=True, help="Path to config.json")
    ap.add_argument("--out_csv", default="batch_results.csv", help="CSV output path")
    ap.add_argument("--save_variants_dir", default="", help="Optional folder to save generated layout JSONs")
    ap.add_argument("--n_layouts", type=int, default=30, help="Number of perturbed layouts to evaluate")
    ap.add_argument("--runs_per_layout", type=int, default=5, help="Monte Carlo runs per layout (different seeds)")
    ap.add_argument("--max_move_cells", type=int, default=3, help="Max grid-cell move in x/y for perturbation")
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--swap_prob", type=float, default=0.15)
    ap.add_argument("--no_swap", action="store_true")
    ap.add_argument("--weights_json", default="", help="Optional scoring weights JSON string or file path")
    args = ap.parse_args()

    with open(args.layout, "r") as f:
        base_layout = json.load(f)
    with open(args.config, "r") as f:
        config = json.load(f)

    # Scoring weights (optional)
    weights = {
        "conversion_rate": 0.4,
        "entry_rate": 0.2,
        "dwell_time": 0.2,
        "social_engagement": 0.2,
        "target_dwell_min": 10.0,
        "target_social": 50.0
    }
    if args.weights_json:
        # allow either a file path or a raw JSON string
        if os.path.exists(args.weights_json):
            with open(args.weights_json, "r") as wf:
                weights.update(json.load(wf))
        else:
            weights.update(json.loads(args.weights_json))

    # Optional: save variants
    if args.save_variants_dir:
        os.makedirs(args.save_variants_dir, exist_ok=True)

    # Evaluate baseline too (layout 0)
    rows = []

    def eval_layout(layout_dict: dict, tag: str, layout_seed: int):
        # run multiple seeds
        run_results = []
        for r in range(args.runs_per_layout):
            seed = args.base_seed + layout_seed * 1000 + r
            run_results.append(run_one_layout(layout_dict, config, seed=seed))
        avg = average_metrics(run_results)
        sc = score_outputs(avg, weights)

        row = {
            "tag": tag,
            "score": sc,
            **avg
        }
        rows.append(row)

    # Baseline
    # Auto-fix baseline if needed
    try:
        validate_layout_fast(base_layout)
    except Exception as e:
        print(f"Baseline layout invalid ({e}). Attempting auto-fix...")
        base_layout = auto_fix_layout(base_layout, seed=args.base_seed)
        print("Baseline auto-fix successful.")

        with open("layout_FIXED.json", "w") as f:
            json.dump(base_layout, f, indent=2)
        print("Wrote layout_FIXED.json (use this going forward).")

    eval_layout(base_layout, tag="baseline", layout_seed=0)

    # Perturbed layouts
    for i in range(1, args.n_layouts + 1):
        var = perturb_layout(
            base_layout,
            max_move_cells=args.max_move_cells,
            allow_swap=(not args.no_swap),
            swap_prob=args.swap_prob,
            seed=args.base_seed + i
        )

        if args.save_variants_dir:
            out_path = os.path.join(args.save_variants_dir, f"layout_variant_{i:03d}.json")
            with open(out_path, "w") as f:
                json.dump(var, f, indent=2)

        eval_layout(var, tag=f"variant_{i:03d}", layout_seed=i)

        if i % 5 == 0:
            print(f"Evaluated {i}/{args.n_layouts} variants...")

    # Sort by score descending
    rows.sort(key=lambda r: r["score"], reverse=True)

    # Write CSV
    fieldnames = ["tag", "score", "conversion_rate", "entry_rate", "avg_dwell_time_min", "social_per_entrant"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\nWrote results to: {args.out_csv}")
    print("Top 5:")
    for r in rows[:5]:
        print(f"  {r['tag']}: score={r['score']:.4f}  conv={r['conversion_rate']:.3f}  entry={r['entry_rate']:.3f}  dwell={r['avg_dwell_time_min']:.2f}  social={r['social_per_entrant']:.2f}")


if __name__ == "__main__":
    main()