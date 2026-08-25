from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="EaseSHAP",
    version="0.1.0",
    description="Efficiency-aware Monte Carlo estimation of Shapley and probabilistic values",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zqliu01/ease_probabilistic_values",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "torch": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "tree": [
            "xgboost>=1.7.0",
            "shap>=0.41.0",
        ],
        "all": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.44.0",
            "pillow>=10.0.0",
            "xgboost>=1.7.0",
            "shap>=0.41.0",
            "shapiq==1.3.0",
            "sparse-transform==0.2.1",
            "galois==0.4.6",
            "colour==0.1.5",
            "overrides>=7.7.0",
            "requests>=2.31.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "ipykernel>=6.0.0",
        ],
    },
)
