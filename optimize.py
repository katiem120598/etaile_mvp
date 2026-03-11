import argparse
import copy
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import random
import math
from collections import defaultdict

from batch_run import (
    perturb_layout,
    repair_layout_to_valid,
    validate_layout_with_sim,
    generate_random_layout,
    load_layout_template,
    average_metrics,
    run_one_layout,
    score_outputs,
    DEFAULT_LAYOUT_TEMPLATE,
)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _fmt_delta(x: float, digits: int = 3) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{digits}f}"


def _layout_opening_signature(layout: dict, key: str) -> Tuple[str, int, int]:
    """Return (wall, offset_cells, width_cells)."""
    cell_size = float(layout["meta"]["cell_size"])
    o = layout[key]
    wall = str(o["wall"])
    offset_cells = int(round(float(o["offset"]) / cell_size))
    width_cells = int(round(float(o["width"]) / cell_size))
    return wall, offset_cells, width_cells


def _module_signature(layout: dict) -> Dict[str, Tuple[str, int, int, int, int, int]]:
    """Map id -> (type, x_cells, y_cells, w_cells, h_cells, rot)."""
    cell_size = float(layout["meta"]["cell_size"])
    out = {}
    for m in layout["modules"]:
        out[str(m["id"])] = (
            str(m["type"]),
            int(round(float(m["x"]) / cell_size)),
            int(round(float(m["y"]) / cell_size)),
            int(round(float(m["w"]) / cell_size)),
            int(round(float(m["h"]) / cell_size)),
            int(m.get("rot", 0)),
        )
    return out


def describe_layout_changes(before: dict, after: dict) -> List[str]:
    """
    Human-readable diff between two layouts.
    Focused on "suggested change" narratives (moves/swaps/openings/type changes).
    """
    changes: List[str] = []

    # Openings
    for key in ("entry", "exit"):
        b = _layout_opening_signature(before, key)
        a = _layout_opening_signature(after, key)
        if b != a:
            bw, bo, bwd = b
            aw, ao, awd = a
            parts = []
            if bw != aw:
                parts.append(f"wall {bw}->{aw}")
            if bo != ao:
                parts.append(f"offset {bo}->{ao} cells")
            if bwd != awd:
                parts.append(f"width {bwd}->{awd} cells")
            changes.append(f"{key}: " + ", ".join(parts))

    # Modules
    bmods = _module_signature(before)
    amods = _module_signature(after)
    common = sorted(set(bmods.keys()) & set(amods.keys()))
    moved: List[Tuple[str, int, int]] = []
    type_changed: List[Tuple[str, str, str]] = []
    rot_changed: List[Tuple[str, int, int]] = []

    for mid in common:
        bt, bx, by, bw, bh, brot = bmods[mid]
        at, ax, ay, aw, ah, arot = amods[mid]
        if bt != at:
            type_changed.append((mid, bt, at))
        if brot != arot:
            rot_changed.append((mid, brot, arot))
        if (bx, by) != (ax, ay):
            moved.append((mid, ax - bx, ay - by))

    # Detect a simple swap: two ids whose deltas are exact opposites
    swap_pairs = set()
    for i in range(len(moved)):
        mi, dx_i, dy_i = moved[i]
        for j in range(i + 1, len(moved)):
            mj, dx_j, dy_j = moved[j]
            if (dx_i == -dx_j) and (dy_i == -dy_j) and (dx_i != 0 or dy_i != 0):
                swap_pairs.add(tuple(sorted((mi, mj))))

    if swap_pairs:
        for a, b in sorted(swap_pairs):
            changes.append(f"swap positions: {a} ↔ {b}")
        swapped_ids = {x for pair in swap_pairs for x in pair}
        moved = [m for m in moved if m[0] not in swapped_ids]

    for mid, dx, dy in sorted(moved, key=lambda t: (abs(t[1]) + abs(t[2])), reverse=True)[:10]:
        changes.append(f"move {mid}: dx={dx} cells, dy={dy} cells")

    for mid, bt, at in sorted(type_changed):
        changes.append(f"change type {mid}: {bt}->{at}")

    for mid, br, ar in sorted(rot_changed):
        changes.append(f"rotate {mid}: {br}deg->{ar}deg")

    if not changes:
        changes.append("no structural changes detected")
    return changes


@dataclass
class EvalResult:
    layout: dict
    avg: dict
    score: float


def eval_layout(layout: dict, config: dict, weights: dict, base_seed: int, runs_per_layout: int, layout_seed: int) -> EvalResult:
    run_results = []
    for r in range(runs_per_layout):
        seed = base_seed + layout_seed * 1000 + r
        run_results.append(run_one_layout(layout, config, seed=seed))
    avg = average_metrics(run_results)
    sc = score_outputs(avg, weights)
    return EvalResult(layout=layout, avg=avg, score=sc)


def _mutate_openings(layout: dict, rng: random.Random, max_offset_nudge_cells: int = 3, max_width_nudge_cells: int = 1) -> dict:
    out = copy.deepcopy(layout)
    cell_size = float(out["meta"]["cell_size"])
    W = int(round(out["space"]["width"] / cell_size))
    H = int(round(out["space"]["height"] / cell_size))

    for key in ("entry", "exit"):
        wall = out[key]["wall"]
        wall_len = W if wall in ("south", "north") else H
        width_cells = int(round(float(out[key]["width"]) / cell_size))
        offset_cells = int(round(float(out[key]["offset"]) / cell_size))

        width_cells = max(2, min(wall_len - 2, width_cells + rng.randint(-max_width_nudge_cells, max_width_nudge_cells)))
        max_offset = max(0, wall_len - width_cells)
        offset_cells = max(0, min(max_offset, offset_cells + rng.randint(-max_offset_nudge_cells, max_offset_nudge_cells)))

        out[key]["width"] = float(width_cells * cell_size)
        out[key]["offset"] = float(offset_cells * cell_size)

    return out


def propose_neighbors(
    current: dict,
    config: dict,
    rng: random.Random,
    n: int,
    max_move_cells: int,
    swap_prob: float,
    allow_swap: bool,
    mutate_openings_prob: float,
    random_restart_prob: float,
    layout_template: dict,
    seed_base: int,
) -> List[dict]:
    neighbors: List[dict] = []
    for i in range(n):
        if rng.random() < random_restart_prob:
            cand = generate_random_layout(layout_template, config, seed=seed_base + i * 17)
        else:
            cand = perturb_layout(
                current,
                max_move_cells=max_move_cells,
                tries=300,
                allow_swap=allow_swap,
                swap_prob=swap_prob,
                seed=seed_base + i * 31,
            )
            if rng.random() < mutate_openings_prob:
                cand = _mutate_openings(cand, rng)

        cand = repair_layout_to_valid(cand, config, seed=seed_base + i * 43)
        neighbors.append(cand)
    return neighbors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.json (required)")
    ap.add_argument("--layout", default="", help="Base layout.json path (optional; if omitted uses template/random start)")
    ap.add_argument("--layout_template_json", default="", help="Template JSON file path or raw JSON string (random start and restarts)")
    ap.add_argument("--weights_json", default="", help="Optional scoring weights JSON string or file path")

    ap.add_argument("--iters", type=int, default=40, help="Optimization iterations")
    ap.add_argument("--neighbors", type=int, default=25, help="Neighbors evaluated per iteration")
    ap.add_argument("--runs_per_layout", type=int, default=5, help="Monte Carlo runs per layout")
    ap.add_argument("--base_seed", type=int, default=42)

    ap.add_argument("--max_move_cells", type=int, default=3)
    ap.add_argument("--swap_prob", type=float, default=0.15)
    ap.add_argument("--no_swap", action="store_true")
    ap.add_argument("--mutate_openings_prob", type=float, default=0.10)
    ap.add_argument("--random_restart_prob", type=float, default=0.05)

    ap.add_argument("--save_best", default="best_layout.json", help="Where to write best layout JSON")
    ap.add_argument("--log_csv", default="opt_log.csv", help="Iteration log CSV")
    ap.add_argument("--top_k_changes", type=int, default=3, help="How many top change suggestions to print")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    weights = {
        "conversion_rate": 0.4,
        "entry_rate": 0.2,
        "dwell_time": 0.2,
        "social_engagement": 0.2,
        "target_dwell_min": 10.0,
        "target_social": 50.0,
    }
    if args.weights_json:
        if os.path.exists(args.weights_json):
            with open(args.weights_json, "r") as wf:
                weights.update(json.load(wf))
        else:
            weights.update(json.loads(args.weights_json))

    layout_template = load_layout_template(args.layout_template_json) if args.layout_template_json else copy.deepcopy(DEFAULT_LAYOUT_TEMPLATE)

    if args.layout:
        with open(args.layout, "r") as f:
            base_layout = json.load(f)
    else:
        base_layout = generate_random_layout(layout_template, config, seed=args.base_seed)

    # Ensure validity (repair if needed)
    try:
        validate_layout_with_sim(base_layout, config, seed=args.base_seed)
    except Exception:
        base_layout = repair_layout_to_valid(base_layout, config, seed=args.base_seed)

    rng = random.Random(args.base_seed)

    baseline_eval = eval_layout(
        base_layout, config, weights, base_seed=args.base_seed, runs_per_layout=args.runs_per_layout, layout_seed=0
    )
    best = baseline_eval
    current = baseline_eval

    # store best improvements (relative to baseline) for "top 3 suggested changes"
    improvements: List[Tuple[float, EvalResult]] = []

    fieldnames = [
        "iter",
        "accepted",
        "score",
        "score_delta_vs_baseline",
        "score_delta_vs_prev",
        "conversion_rate",
        "entry_rate",
        "avg_dwell_time_min",
        "social_per_entrant",
    ]
    with open(args.log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        def log_row(it: int, accepted: int, res: EvalResult, prev_score: float):
            writer.writerow(
                {
                    "iter": it,
                    "accepted": accepted,
                    "score": res.score,
                    "score_delta_vs_baseline": res.score - baseline_eval.score,
                    "score_delta_vs_prev": res.score - prev_score,
                    "conversion_rate": _as_float(res.avg.get("conversion_rate")),
                    "entry_rate": _as_float(res.avg.get("entry_rate")),
                    "avg_dwell_time_min": _as_float(res.avg.get("avg_dwell_time_min")),
                    "social_per_entrant": _as_float(res.avg.get("social_per_entrant")),
                }
            )

        log_row(0, 1, baseline_eval, prev_score=baseline_eval.score)

        for it in range(1, args.iters + 1):
            neighbors = propose_neighbors(
                current.layout,
                config,
                rng=rng,
                n=args.neighbors,
                max_move_cells=args.max_move_cells,
                swap_prob=args.swap_prob,
                allow_swap=(not args.no_swap),
                mutate_openings_prob=args.mutate_openings_prob,
                random_restart_prob=args.random_restart_prob,
                layout_template=layout_template,
                seed_base=args.base_seed + it * 10000,
            )

            scored: List[EvalResult] = []
            for j, lay in enumerate(neighbors):
                scored.append(
                    eval_layout(
                        lay,
                        config,
                        weights,
                        base_seed=args.base_seed,
                        runs_per_layout=args.runs_per_layout,
                        layout_seed=it * 1000 + j,
                    )
                )

            scored.sort(key=lambda r: r.score, reverse=True)
            candidate = scored[0]

            # hill-climb accept only if strictly better than current
            accepted = 0
            prev_score = current.score
            if candidate.score > current.score + 1e-12:
                current = candidate
                accepted = 1

                # track improvements vs baseline
                improvements.append((candidate.score - baseline_eval.score, candidate))

                if candidate.score > best.score + 1e-12:
                    best = candidate

            log_row(it, accepted, current, prev_score=prev_score)

    with open(args.save_best, "w") as f:
        json.dump(best.layout, f, indent=2)

    # Print summary + top change suggestions
    def metrics_line(res: EvalResult) -> str:
        return (
            f"score={res.score:.4f}  conv={_as_float(res.avg.get('conversion_rate')):.3f}  "
            f"entry={_as_float(res.avg.get('entry_rate')):.3f}  "
            f"dwell={_as_float(res.avg.get('avg_dwell_time_min')):.2f}  "
            f"social={_as_float(res.avg.get('social_per_entrant')):.3f}"
        )

    print("\nBaseline:", metrics_line(baseline_eval))
    print("Best:    ", metrics_line(best))
    print(f"\nWrote best layout to: {args.save_best}")
    print(f"Wrote optimization log to: {args.log_csv}")

    if improvements:
        improvements.sort(key=lambda t: t[0], reverse=True)
        seen = set()
        top: List[EvalResult] = []
        for delta, res in improvements:
            # de-dupe by a coarse signature: openings + module positions/types
            sig = (
                _layout_opening_signature(res.layout, "entry"),
                _layout_opening_signature(res.layout, "exit"),
                tuple(sorted(_module_signature(res.layout).items())),
            )
            if sig in seen:
                continue
            seen.add(sig)
            top.append(res)
            if len(top) >= max(1, args.top_k_changes):
                break

        print(f"\nTop {len(top)} suggested change sets (vs baseline):")
        for idx, res in enumerate(top, start=1):
            delta_score = res.score - baseline_eval.score
            delta_conv = _as_float(res.avg.get("conversion_rate")) - _as_float(baseline_eval.avg.get("conversion_rate"))
            delta_entry = _as_float(res.avg.get("entry_rate")) - _as_float(baseline_eval.avg.get("entry_rate"))
            delta_dwell = _as_float(res.avg.get("avg_dwell_time_min")) - _as_float(baseline_eval.avg.get("avg_dwell_time_min"))
            delta_social = _as_float(res.avg.get("social_per_entrant")) - _as_float(baseline_eval.avg.get("social_per_entrant"))

            print(
                f"\n{idx}) dScore={_fmt_delta(delta_score, 4)}  "
                f"dConv={_fmt_delta(delta_conv)}  dEntry={_fmt_delta(delta_entry)}  "
                f"dDwell={_fmt_delta(delta_dwell, 2)}  dSocial={_fmt_delta(delta_social)}"
            )
            for line in describe_layout_changes(base_layout, res.layout)[:12]:
                print(f"   - {line}")
    else:
        print("\nNo improvements over baseline were accepted with current settings.")


if __name__ == "__main__":
    main()

