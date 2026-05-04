from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="EaseSHAP",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Profiled Augmented Contrast Estimation for SHAP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/easeshap",
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
            "xgboost>=1.7.0",
            "shap>=0.41.0",
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
