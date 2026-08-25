# Full SOU benchmark

This experiment compares EASE with the applicable Shapley and probabilistic-
value estimators on structured Gaussian SOU games.

## Reported configuration

- Players: $n\in\{40,80,160\}$.
- Game: `gameSOUStructuredGaussianBitset`, game seed 42.
- High-order terms: $n^2$, with maximum order 40.
- Values: Shapley, Beta-Shapley $(1,4)$, Beta-Shapley $(4,1)$, and weighted
  Banzhaf with parameters 0.25, 0.5, and 0.75.
- Game settings: $\eta\in\{0.25,0.5,0.75\}$ with
  $\alpha=\sqrt{\eta}$.
- Reference: analytic SOU values.
- Replicates: 10 estimator seeds, beginning at 2026 and increasing by 137.
- Total budget: 160,000 utility evaluations at each player count.
- Checkpoints: 50 equally spaced checkpoints.
- EASE-FO and EASE-SP: three pilot-design updates, fixed order-1 boundary
  handling, complement sampling for symmetric values, and two-fold
  cross-fitting.

The per-player budgets are 4,000 for $n=40$, 2,000 for $n=80$, and 1,000 for
$n=160$. The complete estimator specifications are recorded in the published
`config.json` files.

## Running the experiment

From this directory, run:

```bash
python run_benchmark_n40.py --phase all --max-high-order-size 40 --ease-fo-pilot-design-updates 3 --ease-sp-pilot-design-updates 3
python run_benchmark_n80.py --phase all --max-high-order-size 40 --ease-fo-pilot-design-updates 3 --ease-sp-pilot-design-updates 3
python run_benchmark_n160.py --phase all --max-high-order-size 40 --ease-fo-pilot-design-updates 3 --ease-sp-pilot-design-updates 3
```

The exact AUCC tables and configurations used for the reported figures are in
`results/published_n40/`, `results/published_n80/`, and
`results/published_n160/`. The central figure generator produces the four
`aucc_*.png` artifacts from `aucc_mean.csv` and `aucc_std.csv`.
