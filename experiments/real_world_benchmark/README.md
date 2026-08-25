# Real-world benchmark

This experiment evaluates Shapley and probabilistic-value estimators on image
and tabular feature-attribution games. It also supplies the pilot-fraction
sensitivity results.

## Data preprocessing

The ViT4by4 benchmark uses the 30 precomputed 16-player games distributed with
PolySHAP. Install them in the `shapiq==1.3.0` benchmark cache and verify that
all 30 games load with:

```bash
python prepare_polyshap_vit4by4.py
```

For CIFAR-10, `prepare_cifar10.py` uses the test split shuffled with random
state 40 and selects the first 30 images. Each image is divided into a $4\times
4$ grid of 16 players. Missing patches are replaced by gray value 128, and the
utility is the logit of the class predicted for the unmasked image by
`aaraki/vit-base-patch16-224-in21k-finetuned-cifar10`. The script enumerates
all $2^{16}$ coalitions and writes one compressed utility table per image.
Run it on a GPU with:

```bash
python prepare_cifar10.py --n-instances 30 --device cuda
```

The tabular datasets are loaded automatically: Breast Cancer from
scikit-learn, and NHANES I and Communities and Crime from SHAP. Missing feature
values are replaced by column means. For each dataset, the benchmark fits a
100-tree random-forest regressor with maximum depth 8, all features available
at each split, and random state 40. The baseline is the feature-wise data mean;
30 explained inputs are selected deterministically. These fitted trees and
baseline/explicand pairs define the coalition games and their exact tree-based
reference values.

## Reported configuration

- Datasets: ViT4by4 patches ($n=16$), CIFAR-10 ($n=16$), Breast Cancer
  ($n=30$), NHANES I ($n=79$), and Communities and Crime ($n=101$).
- Values: Shapley, Beta-Shapley $(1,4)$, Beta-Shapley $(4,1)$, and weighted
  Banzhaf with parameters 0.25, 0.5, and 0.75.
- Instances: 30 per dataset, with one estimator run per instance.
- Budget: $m=200n$ utility evaluations.
- Checkpoints: 20, spaced every $10n$ evaluations.
- EASE pilot fractions: 0.05, 0.10, 0.20, and 0.40.
- EASE-FO: three pilot-design updates, fixed order-1 boundary handling,
  complement sampling, and two-fold cross-fitting.
- Tabular games: `baseline_tree` mode with exact tree-based reference values.
- Seeds: base estimator seed 2026 and dataset random state 40.

The exact configuration is stored in `results/published/config.json`; the
aggregated results used by the paper are in
`results/published/l2_summary.csv`.

## Running the benchmark

The full task grid was divided into 200 shards. The command below runs shard 0;
repeat it with `--task-id` set to each integer through 199:

```bash
python run_benchmark.py \
  --budget-per-player 200 \
  --ease-switch-fractions 0.05,0.10,0.20,0.40 \
  --num-checkpoints 20 \
  --ease-fo-pilot-design-updates 3 \
  --config-name m200n_pilots0p05_0p10_0p20_0p40_updates3_ckpt20 \
  --exclude-method LeverageSHAP_paired_border \
  --exclude-method RegressionMSR_unbiased_no_replacement \
  --task-id 0 \
  --num-tasks 200 \
  --allow-failures
```

After every shard finishes, aggregate the result table with:

```bash
python aggregate_l2.py \
  --budget-per-player 200 \
  --ease-switch-fractions 0.05,0.10,0.20,0.40 \
  --num-checkpoints 20 \
  --ease-fo-pilot-design-updates 3 \
  --config-name m200n_pilots0p05_0p10_0p20_0p40_updates3_ckpt20
```

The reported six-panel trajectory figure uses the Shapley rows for CIFAR-10,
Breast Cancer, Communities and Crime, and NHANES I, with EASE-FO pilot
fraction 0.20. The sensitivity figure uses all four EASE-FO pilot fractions.

The runtime figure reads
`results/published/runtime_breakdown_3panel_summary.csv`; it reports the mean
utility-evaluation and estimator-overhead components for ACSIncome-XGBoost,
Communities and Crime, and NHANES I.
