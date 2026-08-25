# Figures and tables

All reported figures and the allocation-diagnostics table are generated from
the published CSV and JSON inputs with one command. From the repository root,
run:

```bash
python paper/experiments/figures/generate_published.py
```

The generator calls the plotting scripts in this directory and writes the
artifacts to `paper/experiments/figures/published/`:

- `sou_comparison_various_eta_5panel.{png,pdf}`
- `aucc_eta_0p25_n40.png`
- `aucc_eta_0p5_0p75_n40.png`
- `aucc_eta_all_n80.png`
- `aucc_eta_all_n160.png`
- `shapley_estimation_trajectories_6panel.{png,pdf}`
- `runtime_breakdown_3panel.{png,pdf}`
- `pilot_fraction_sensitivity_4panel.{png,pdf}`
- `allocation_tv_diagnostics.tex`

The source mapping is:

- matched SOU comparison: `SOU_comparison_various_eta/results/published/`;
- SOU AUCC figures: `SOU_full_benchmark/results/published_n*/`;
- ACSIncome trajectory panels:
  `acs_income_state_valuation/results/published_{logistic,xgboost}/`;
- real-world trajectories and sensitivity:
  `real_world_benchmark/results/published/l2_summary.csv`;
- runtime breakdown:
  `real_world_benchmark/results/published/runtime_breakdown_3panel_summary.csv`;
- allocation diagnostics:
  `figures/data/allocation_tv_diagnostics.csv`.

Each experiment README documents the configuration used to produce these
published inputs.

## Allocation diagnostics

The allocation diagnostics combine ACSIncome with the Breast Cancer, NHANES I,
and Communities and Crime feature-attribution games. Their computation and
published inputs therefore live in this cross-experiment directory rather than
inside one benchmark.

`compute_allocation_diagnostics.py` uses the included ACSIncome coalition
utilities in `data/allocation_diagnostics_acs_utilities.csv` and loads the
three feature-attribution games from `real_world_benchmark/`. The reported
configuration is:

- three allocation updates;
- budget $200n$ and pilot fraction 0.20;
- ridge parameter 0.01;
- 10 repeated pilot/split draws;
- one ACSIncome utility instance; and
- instances 0 through 9 for each feature-attribution dataset.

Run the computation from the repository root with:

```bash
python paper/experiments/figures/compute_allocation_diagnostics.py
```

The published CSV contains only
$\mathrm{TV}(\hat q,\mathrm{LeverageSHAP})$ and
$\mathrm{TV}(\hat q,q^{\mathrm{init}})$, reported as mean with standard
error. `write_allocation_diagnostics_table.py` formats that CSV as
`published/allocation_tv_diagnostics.tex` when the main figure generator runs.
