# EaseSHAP

Code for **EaseSHAP / EASE**, an efficiency-aware method for Monte Carlo estimation of Shapley values and more general probabilistic values.

This repository accompanies the paper **“First-Order Efficiency for Probabilistic Value Estimation via A Statistical Viewpoint”**, which is currently in preparation. Citation information and fuller documentation will be added after the paper is public.

## Structure

- [`EaseSHAP/`](EaseSHAP/README.md): Python package implementation and usage.
- [`experiments/`](experiments/README.md): reported experiment configurations,
  published result inputs, and figure-generation scripts.

## Reproducing the published results

From the repository root, enter `paper/` and install the package together with
the additional dependency used by the reporting scripts:

```bash
cd paper
pip install -e ./EaseSHAP pandas
```

To regenerate every reported figure and the allocation-diagnostics table from
the included published inputs, run:

```bash
python experiments/figures/generate_published.py
```

See [`experiments/figures/README.md`](experiments/figures/README.md) for the
complete artifact list.

## License

This project is released under the [MIT License](LICENSE).
