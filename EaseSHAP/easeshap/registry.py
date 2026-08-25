"""Estimator registry compatibility layer.

Experiment scripts can use this module for backend lookup without depending on
where each estimator family is implemented internally.
"""

from . import baselines as _baselines
from .ease import EaseSHAP, EaseSHAP_group


def get_estimator_class(name):
    if name == "EaseSHAP":
        return EaseSHAP
    if name == "EaseSHAP_group":
        return EaseSHAP_group
    return getattr(_baselines, name)


def __getattr__(name):
    return get_estimator_class(name)


def __dir__():
    return sorted(set(globals()) | set(getattr(_baselines, "__all__", [])) | {"EaseSHAP", "EaseSHAP_group"})
