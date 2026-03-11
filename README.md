### Running Simulation
python popup_sim.py --layout layout.json --config config.json --seed 42

### Running Batch Simulations
python batch_run.py --layout layout.json --config config.json --save_variants_dir variants --n_layouts 50 --runs_per_layout 3

OR W/ 50 DIFFERENT LAYOUTS RATHER THAN PERTURBING
python batch_run.py --config config.json --save_variants_dir variants --n_layouts 50 --runs_per_layout 3

CAN ALSO ADD ARG FOR RANDOM SEED IF WANT
--base_seed 42

### Week 1: Optimization (Hill Climb)
Runs a simple AI-assisted layout optimization loop (propose neighbors → simulate → score → accept improvements),
logs metric deltas each iteration, writes the best layout, and prints the top suggested change sets.

python optimize.py --layout layout.json --config config.json --iters 40 --neighbors 25 --runs_per_layout 3 --save_best best_layout.json --log_csv opt_log.csv

#### Optimization outputs (where to find results)
- **Best layout JSON**: `best_layout.json` (or whatever you pass via `--save_best`)
  - This is a normal layout file in the same schema as `layout.json`.
  - You can simulate it directly with:
    - `python popup_sim.py --layout best_layout.json --config config.json --seed 42`
- **Iteration log**: `opt_log.csv` (or whatever you pass via `--log_csv`)
  - One row per iteration (including iteration 0 = baseline), containing:
    - `score`: weighted ROE score for the **accepted** layout at that iteration
    - `score_delta_vs_baseline`: how much score has improved vs iteration 0
    - `score_delta_vs_prev`: change vs previous accepted iteration (0 when not accepted)
    - The averaged metrics: `conversion_rate`, `entry_rate`, `avg_dwell_time_min`, `social_per_entrant`
- **Console summary** (printed at the end)
  - `Baseline:` metrics for the starting layout
  - `Best:` metrics for the best layout found during the search
  - `Top K suggested change sets (vs baseline):` the best improvement candidates, shown as:
    - `dScore/dConv/dEntry/dDwell/dSocial`: metric deltas vs the baseline
    - Bullets like `move m6: dx=0 cells, dy=2 cells`, `swap positions: m2 <-> m6`, `entry: offset ...`

#### How to interpret the optimization results
- **Score is a weighted blend** of the 4 metrics (same as `batch_run.py` scoring):
  - `conversion_rate` and `entry_rate` contribute directly.
  - `avg_dwell_time_min` and `social_per_entrant` are normalized vs targets (`target_dwell_min`, `target_social`)
    before weighting.
  - Override weights/targets with `--weights_json` (file path or raw JSON string) to reflect different goals.
- **Suggested changes are relative to the baseline layout**, not relative to the previous iteration.
  - If you want “step-by-step” change history, use `opt_log.csv` + the saved `best_layout.json` to compare
    the baseline against the final output (or rerun with fewer iterations for a more incremental path).

### Modifications with Web Input
places where code might need to be modified once the UI is integrated are denoted with comment ###CHANGE

## ASSUMPTIONS AND INPUTS
### config.json
MOST IMPORTANT DOCUMENT. Where all of the assumptions and constraints are housed. Should only change when data or assumptions change, NOT per run.

### popup_sim.py
SECOND MOST IMPORTANT DOCUMENT. Where the simulations are actually run. This includes assumptions about stochastic modeling and visitor behavior.

#### Simulation Assumption Sheet

##### Core Random Processes (Distributions)
- **Passersby arrivals**: Poisson per simulation step.
    - Formula: `N_t ~ Poisson(lambda * delta_t)`
    - Config knobs: `sim_params.passersby_rate_per_min`, `sim_params.step_seconds`

- **Entry decision**: Bernoulli draw per passerby.
    - Formula: `p_enter = clamp(0.5*base_entry_prob + 0.5*logistic(a + b*visibility - c*congestion), 0, 1)`
    - Config knobs: `sim_params.base_entry_prob`, `entry_logit_a`, `entry_logit_b`, `entry_logit_c`,
        `entry_visibility_radius_cells`, `entry_congestion_radius_cells`

- **Archetype assignment**: Categorical draw from `visitor_mix`.
    - Config knobs: `visitor_mix.transactional`, `visitor_mix.experiential`, `visitor_mix.social`

- **Module selection**: Softmax choice over module utilities.
    - Formula: `P(module_i) ∝ exp(u_i / temperature)`
    - Config knobs: `sim_params.choice_temperature` and utility inputs below

- **Dwell duration**: Gaussian (Normal) draw then clipped/scaled.
    - Formula: `X ~ Normal(dwell_mean_s, dwell_sd_s)`, then `X = max(5, X) * dwell_mult`
    - Config knobs: `module_params.<type>.dwell_mean_s`, `dwell_sd_s`, `archetypes.<type>.dwell_mult`

- **Conversion at exit**: Bernoulli draw once at exit.
    - Formula: `p_conv = clamp(base_conversion + sum(conversion_lift of interacted modules), 0, 1)`
    - Config knobs: `archetypes.<type>.base_conversion`, `module_params.<type>.conversion_lift`

- **Early exit while moving**: Bernoulli with increasing hazard over time.
    - Formula: `p_exit = 0.15 * clamp(time_inside / max_visit_time, 0, 1)`
    - Config knobs: `sim_params.max_visit_minutes`

##### Deterministic / Rule-Based Assumptions
- **Utility model**: `utility = attractor * attraction_mult * distance_decay * queue_penalty`
- **Distance decay**: `1 / (1 + ManhattanDistance)`
- **Queue penalty**: `exp(-queue_beta * max(0, q - queue_free))`
- **Pathing**: grid BFS shortest-path step toward module approach cell or exit
- **Capacity/queue mechanics**: module capacity enforced; overflow joins queue up to `queue_max`

##### Geometry Validation (Now Enforced)
- **Bounds + overlap check**: modules must be fully in-bounds and non-overlapping.
- **Entry/exit distinction**: entry and exit openings cannot be identical.
- **Entry→exit connectivity**: at least one walkable path must exist from entry opening to exit opening.
- **Minimum aisle path**: there must be at least one entry→exit path where every traversed cell satisfies
    `sim_params.min_aisle_width_ft` (converted to cells).
- **Validation errors are explicit**: `popup_sim.py` raises clear messages (blocked opening, no path, narrow choke points).

##### Important MVP Simplifications
- "Visibility" is a radius-based proximity proxy, not true line-of-sight or occlusion-based visibility.
- Social engagement is modeled as expected shares (`share_prob * share_mult`) accumulation, not explicit share events.

### batch_run_ii.py
`batch_run_ii.py` now validates every candidate layout against `popup_sim.py` geometry constraints before simulation.
If a layout fails validation, it prints the specific reason and attempts repair (auto-fix + perturbation) before running.

### batch_run.py
batch_run.py works by taking in a default layout.json file and config.json file housed in this folder and "perturbing" the layout to create different variations. However, if no layout argument is entered, it will create unique variants within the batch rather than perturbing. The config dict isn't changing. 

    weights = {
        "conversion_rate": 0.4,
        "entry_rate": 0.2,
        "dwell_time": 0.2,
        "social_engagement": 0.2,
        "target_dwell_min": 10.0,
        "target_social": 50.0
    }
This chunk of code sets default weights to how much each of these metrics matter in the scoring. These are automatically OVERRIDDEN if there is a weights_json (will need to be produced from user input) using the following code chunk:
    if args.weights_json:
        # allow either a file path or a raw JSON string
        if os.path.exists(args.weights_json):
            with open(args.weights_json, "r") as wf:
                weights.update(json.load(wf))
        else:
            weights.update(json.loads(args.weights_json))

