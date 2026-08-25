# ACSIncome state valuation

This experiment treats the 50 U.S. states as players and estimates their
Shapley values for predicting income in Pennsylvania. It reports separate
panels for logistic-regression and XGBoost coalition utilities.

## Data preprocessing

`acs_data.py` downloads the 2018 one-year ACS Person files for all 50 states
and applies the Folktables ACSIncome definition. The deterministic split uses
seed 2026:

- Pennsylvania is shuffled once; the first 1,000 rows form the evaluation set
  and the next 500 rows form its training set, so the two sets are disjoint.
- Each other state contributes 500 training rows sampled with a deterministic
  state-specific seed.
- The unencoded state training files, Pennsylvania evaluation file, selected
  row indices, and label rates are recorded under `data/processed/`.

Prepare this split with:

```bash
python acs_data.py prepare --download --survey-year 2018 --target-state PA --train-size 500 --eval-size 1000 --seed 2026
```

The reported `full` encoder is fitted only on the concatenated training data.
It standardizes the numeric `AGEP` and `WKHP` features and one-hot encodes
`COW`, `SCHL`, `MAR`, `OCCP`, `POBP`, `RELP`, `SEX`, and `RAC1P`, ignoring
previously unseen categories. The fitted encoder produces 788 columns and is
then applied unchanged to every state and to the held-out Pennsylvania set.
The logistic-regression and XGBoost experiments use this same encoded split.

## Reported configuration

- Data: 2018 ACSIncome, target state Pennsylvania.
- Players: 50 states.
- Split: 500 training observations per state and 1,000 held-out Pennsylvania
  observations, sampled with data seed 2026.
- Encoder: full one-hot encoding, producing 788 features.
- Utility models: logistic regression (`liblinear`, 5,000 maximum iterations)
  and XGBoost (5 trees, depth 5, learning rate 0.1, `hist` tree method).
- Main budget: 200 average utility evaluations per player, or 10,000 total,
  with 20 checkpoints.
- Replicates: 20 estimator seeds beginning at 2026.
- Repeated coalition utilities: recomputed rather than memoized.
- EASE-FO: pilot fraction 0.20, three pilot-design updates, fixed order-1
  boundary handling, complement sampling, and two-fold cross-fitting.
- Reference: one OFA run at 4,000 average evaluations per player (200,000
  total), seed 102026, using 20 estimator processes.

The reported method set is EASE-FO, EASE-SP, OFA, OFA baseline, Sampling lift,
SHAP-IQ, GELS, Improved AME, kernelSHAP, LeverageSHAP, Permutation, Complement,
Group testing, and RegressionMSR.

## Running the experiment

After preprocessing, run the benchmark once with `--utility-model logistic`
and once with `--utility-model xgboost`:

```bash
python benchmark_shapley_estimators.py \
  --utility-model logistic \
  --encoder full \
  --utility-cache-mode recompute \
  --survey-year 2018 \
  --target-state PA \
  --train-size 500 \
  --eval-size 1000 \
  --data-seed 2026 \
  --model-seed 2026 \
  --estimator-seed-start 2026 \
  --num-seeds 20 \
  --nue-avg 200 \
  --num-checkpoints 20 \
  --reference-method OFA_fixed \
  --reference-nue-avg 4000 \
  --reference-num-seeds 1 \
  --reference-seed-start 102026 \
  --reference-estimator-processes 20 \
  --ease-fo-pilot-design-updates 3
```

The exact trajectory inputs used by the paper are in
`results/published_logistic/` and `results/published_xgboost/`. For each seed
and checkpoint, the figure generator converts `rmse_to_reference` to relative
squared $L_2$ error using

$$
\frac{n\,\mathrm{RMSE}(\widehat\phi,\phi)^2}{\lVert\phi\rVert_2^2},
$$

then plots the arithmetic mean over the 20 seeds.
