# EaseSHAP

A Python package implementing EaseSHAP / EASE, an efficiency-aware method for
Monte Carlo estimation of Shapley values and more general probabilistic values.

## Features

- Fast estimation of various semivalues:
  - Shapley values
  - Weighted Banzhaf values
  - Beta-Shapley values
- Parallel processing support
- Exact computation for small problems
- Multiple sampling estimators

## Installation

### From source
```bash
pip install -e .
```

### With all optional dependencies
```bash
pip install -e ".[all]"
```

### For development
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np

from easeshap import runEstimator


class ThresholdGame:
    def __init__(self, weights, threshold):
        self.weights = np.asarray(weights, dtype=float)
        self.threshold = float(threshold)

    def evaluate(self, coalition):
        return float(self.weights @ np.asarray(coalition, dtype=bool) >= self.threshold)


estimator = runEstimator(
    estimator="exact_value",
    n_process=1,
    semivalue="shapley",
    semivalue_param=None,
    game_func=ThresholdGame,
    game_args={"weights": [1, 2, 3], "threshold": 3},
    num_player=3,
    nue_avg=1,
    nue_per_proc=24,
    nue_track_avg=1,
)

values, trajectory = estimator.run()
print(values)
```

## Package Structure

```
easeshap/
├── __init__.py          # Package initialization
├── ease.py              # Main EaseSHAP and EaseSHAP_group methods
├── base.py              # Shared estimator template
├── baselines.py         # Baseline estimators and comparison methods
├── runner.py            # Estimator orchestration
├── group_core.py        # Group-sum coefficients and base template
├── group_estimators.py  # Group-sum baselines and wrappers
├── registry.py          # Backend lookup compatibility layer
├── utils.py             # Utility functions
├── models.py            # Game models
└── datasets.py          # Dataset loading utilities
```

## License

This project is released under the [MIT License](LICENSE).
