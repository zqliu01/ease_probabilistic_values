"""Efficiency-aware Monte Carlo estimation of probabilistic values."""

__version__ = "0.1.0"

# Import main classes for easy access
from .base import estimatorTemplate
from .baselines import exact_value
from .ease import EaseSHAP, EaseSHAP_group
from .group_core import exact_group_sum_value, groupEstimatorTemplate
from .group_estimators import (
    runGroupSumEstimator,
    FGSVShapley,
    IndividualRegressionMSRUnbiased,
    IndividualRegressionMSRUnpaired,
    IndividualOFAFixed,
)
from .runner import runEstimator
from .utilityFuncs import (
    gameSOU,
    gameSOUPaper,
    gameSOUStructuredGaussian,
    gameSOUStructuredGaussianBitset,
    generate_sou_game,
    sv_sou_true,
)
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
    'gameSOUStructuredGaussianBitset',
    'generate_sou_game',
    'sv_sou_true',
]
