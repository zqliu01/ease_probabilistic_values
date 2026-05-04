# EaseSHAP

**Profiled Augmented Contrast Estimation for SHAP**

A Python package for efficient estimation of semivalues (Shapley values, Banzhaf values, Beta-Shapley values) using profiled augmented contrast estimators.

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

### For development
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from easeshap import runEstimator, exact_value

# Define your game function and parameters
# ... (add example usage)

# Run estimator
estimator = runEstimator(
    estimator='exact_value',
    n_process=4,
    semivalue='shapley',
    game_func=your_game_function,
    num_player=10,
    # ... other parameters
)

values, trajectory = estimator.run()
```

## Package Structure

```
easeshap/
├── __init__.py          # Package initialization
├── estimators.py        # Core estimation classes
├── utils.py             # Utility functions
├── models.py            # Game models
└── datasets.py          # Dataset loading utilities
```
<!-- 
## Documentation

(Add link to documentation) -->

## License

MIT License
<!-- 
## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. -->
