# Experiments

This directory contains the experiments reported in the paper. Each experiment
has one README describing the configuration used to produce its published
result tables.

- [`SOU_comparison_various_eta/`](SOU_comparison_various_eta/README.md):
  matched SOU comparisons with RegressionMSR, OFA, and PolySHAP.
- [`SOU_full_benchmark/`](SOU_full_benchmark/README.md): SOU benchmark across
  player counts, probabilistic values, and estimators.
- [`acs_income_state_valuation/`](acs_income_state_valuation/README.md):
  ACSIncome state-valuation benchmark with logistic regression and XGBoost
  utilities.
- [`real_world_benchmark/`](real_world_benchmark/README.md): image and tabular
  Shapley/semivalue benchmarks and pilot-fraction sensitivity.
- [`figures/`](figures/README.md): the single entry point for generating all
  reported figures and the cross-experiment diagnostics table.

The exact CSV and JSON inputs used by the figure generator are stored in each
experiment's `results/published*` directories. To build every reported
artifact, run from the repository root:

```bash
python paper/experiments/figures/generate_published.py
```
