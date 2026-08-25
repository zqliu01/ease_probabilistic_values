# Matched SOU comparisons

This experiment compares EASE separately with RegressionMSR, OFA, and
PolySHAP on the structured Gaussian SOU game.

## Reported configuration

- Players: $n=40$.
- Game: `gameSOUStructuredGaussianBitset`, game seed 42.
- High-order terms: $n^2=1600$, with maximum order 40.
- Values: Shapley value at $\eta\in\{0.25,0.5,0.75\}$, implemented with
  $\alpha=\sqrt{\eta}$.
- Reference: analytic SOU Shapley values.
- Replicates: 10 estimator seeds, beginning at 2026 and increasing by 137.
- Budget: 5,000 average utility evaluations per player, recorded every 500.
- EASE allocation optimization: three pilot-design updates and two-fold
  cross-fitting.

The EASE surrogate matches the baseline being compared:

- RegressionMSR comparison: first-order interaction surrogate, no exact
  boundary handling, and no complement sampling.
- OFA comparison: size-player surrogate, fixed order-1 boundary handling,
  no complement sampling, and size-trace ridge regularization.
- PolySHAP comparison: second-order interaction surrogate, no exact boundary
  handling, and no complement sampling; PolySHAP uses maximum order 2.

## Running the experiment

From this directory, run the RegressionMSR experiment first because the other
two commands use its analytic reference values:

```bash
SOU_MAX_HIGH_ORDER_SIZE=40 python regmsr_unpaired_vs_easeshap.py --phase all
SOU_MAX_HIGH_ORDER_SIZE=40 python ofa_vs_easeshap.py --phase all
SOU_MAX_HIGH_ORDER_SIZE=40 python polyshap_vs_easeshap.py --phase all
```

The exact summary tables used for the paper figure are under
`results/published/`. The central figure generator reads those tables to create
`sou_comparison_various_eta_5panel.{png,pdf}`.
