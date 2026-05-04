"""
EaseSHAP: Profiled Augmented Contrast Estimation for SHAP

A Python package for efficient estimation of semivalues (Shapley values,
Banzhaf values, Beta-Shapley values) using profiled augmented contrast
estimators.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main classes for easy access
from .estimators import (
    runEstimator,
    estimatorTemplate,
    exact_value,
    EaseSHAP,
)
from .group_estimators import (
    runGroupSumEstimator,
    groupEstimatorTemplate,
    EaseSHAP_group,
    FGSVShapley,
    IndividualRegressionMSRUnbiased,
    IndividualRegressionMSRUnpaired,
    IndividualOFAFixed,
    exact_group_sum_value,
)
from .utilityFuncs import gameSOU, gameSOUPaper, gameSOUStructuredGaussian
from .utils import *

__all__ = [
    'runEstimator',
    'estimatorTemplate', 
    'exact_value',
    'EaseSHAP',
    'runGroupSumEstimator',
    'groupEstimatorTemplate',
    'EaseSHAP_group',
    'FGSVShapley',
    'IndividualRegressionMSRUnbiased',
    'IndividualRegressionMSRUnpaired',
    'IndividualOFAFixed',
    'exact_group_sum_value',
    'gameSOU',
    'gameSOUPaper',
    'gameSOUStructuredGaussian',
]
