"""Semantic compressed encoder for the ACSIncome feature set.

The grouping rules use ACS PUMS variable meanings:

- OCCP is grouped into the broad occupation recode prefixes used in the
  2018 ACS PUMS data dictionary.
- POBP is grouped into Census-region birth states plus broad foreign-born
  world-area groups, with the target state kept separate.
- SCHL, RELP, RAC1P, and COW are grouped into coarse, interpretable categories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["AGEP", "WKHP"]
SEMANTIC_CATEGORICAL_FEATURES = [
    "COW_GROUP",
    "SCHL_GROUP",
    "MAR",
    "OCCP_GROUP",
    "POBP_GROUP",
    "RELP_GROUP",
    "SEX",
    "RAC1P_GROUP",
]

OCCP_GROUPS = [
    (10, 440, "MGR"),
    (500, 750, "BUS"),
    (800, 960, "FIN"),
    (1005, 1240, "CMM"),
    (1305, 1560, "ENG"),
    (1600, 1980, "SCI"),
    (2001, 2060, "CMS"),
    (2100, 2180, "LGL"),
    (2205, 2555, "EDU"),
    (2600, 2920, "ENT"),
    (3000, 3550, "MED"),
    (3601, 3655, "HLS"),
    (3700, 3960, "PRT"),
    (4000, 4160, "EAT"),
    (4200, 4255, "CLN"),
    (4330, 4655, "PRS"),
    (4700, 4965, "SAL"),
    (5000, 5940, "OFF"),
    (6005, 6130, "FFF"),
    (6200, 6765, "CON"),
    (6800, 6950, "EXT"),
    (7000, 7640, "RPR"),
    (7700, 8990, "PRD"),
    (9005, 9760, "TRN"),
    (9800, 9830, "MIL"),
    (9920, 9920, "NILF5"),
]

STATE_CODE_TO_ABBR = {
    1: "AL",
    2: "AK",
    4: "AZ",
    5: "AR",
    6: "CA",
    8: "CO",
    9: "CT",
    10: "DE",
    11: "DC",
    12: "FL",
    13: "GA",
    15: "HI",
    16: "ID",
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    21: "KY",
    22: "LA",
    23: "ME",
    24: "MD",
    25: "MA",
    26: "MI",
    27: "MN",
    28: "MS",
    29: "MO",
    30: "MT",
    31: "NE",
    32: "NV",
    33: "NH",
    34: "NJ",
    35: "NM",
    36: "NY",
    37: "NC",
    38: "ND",
    39: "OH",
    40: "OK",
    41: "OR",
    42: "PA",
    44: "RI",
    45: "SC",
    46: "SD",
    47: "TN",
    48: "TX",
    49: "UT",
    50: "VT",
    51: "VA",
    53: "WA",
    54: "WV",
    55: "WI",
    56: "WY",
}

BIRTH_REGIONS = {
    "Other Northeast birth state": {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY"},
    "Midwest birth state": {
        "IL",
        "IN",
        "MI",
        "OH",
        "WI",
        "IA",
        "KS",
        "MN",
        "MO",
        "NE",
        "ND",
        "SD",
    },
    "South birth state": {
        "DE",
        "DC",
        "FL",
        "GA",
        "MD",
        "NC",
        "SC",
        "VA",
        "WV",
        "AL",
        "KY",
        "MS",
        "TN",
        "AR",
        "LA",
        "OK",
        "TX",
    },
    "West birth state": {
        "AZ",
        "CO",
        "ID",
        "MT",
        "NV",
        "NM",
        "UT",
        "WY",
        "AK",
        "CA",
        "HI",
        "OR",
        "WA",
    },
}

TERRITORIES = {60, 66, 69, 72, 78}
CENTRAL_AMERICA = set(range(310, 317))
CARIBBEAN = {321, 323, 324, 327, 328, 329, 330, 332, 333, 338, 339, 340, 341, 343, 344}
SOUTH_AMERICA = {360, 361, 362, 363, 364, 365, 368, 369, 370, 372, 373, 374}


def _to_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def occp_group(values: pd.Series) -> pd.Series:
    labels = []
    for value in _to_numeric(values):
        if pd.isna(value):
            labels.append("Unknown occupation")
            continue
        code = int(value)
        label = "Unknown occupation"
        for low, high, group in OCCP_GROUPS:
            if low <= code <= high:
                label = group
                break
        labels.append(label)
    return pd.Series(labels, index=values.index)


def pobp_group(values: pd.Series, target_state: str) -> pd.Series:
    labels = []
    for value in _to_numeric(values):
        if pd.isna(value):
            labels.append("Unknown birthplace")
            continue
        code = int(value)
        if code in STATE_CODE_TO_ABBR:
            state = STATE_CODE_TO_ABBR[code]
            if state == target_state:
                labels.append(f"{target_state} birth state")
                continue
            labels.append(_birth_region_label(state))
        elif code in TERRITORIES:
            labels.append("U.S. territory")
        elif 100 <= code <= 169:
            labels.append("Europe-born")
        elif 200 <= code <= 254:
            labels.append("Asia-born")
        elif code in {301, 303}:
            labels.append("Canada/Mexico-born")
        elif code in CENTRAL_AMERICA or code in CARIBBEAN or code in SOUTH_AMERICA or code == 399:
            labels.append("Latin America/Caribbean/South America-born")
        elif 400 <= code <= 469:
            labels.append("Africa-born")
        elif 501 <= code <= 554:
            labels.append("Oceania/at sea-born")
        else:
            labels.append("Unknown birthplace")
    return pd.Series(labels, index=values.index)


def _birth_region_label(state: str) -> str:
    for label, states in BIRTH_REGIONS.items():
        if state in states:
            return label
    return "Other U.S. birth state"


def schl_group(values: pd.Series) -> pd.Series:
    labels = []
    for value in _to_numeric(values):
        if pd.isna(value):
            labels.append("Unknown education")
        elif value <= 15:
            labels.append("Less than high school")
        elif value in (16, 17):
            labels.append("High school/GED")
        elif value in (18, 19, 20):
            labels.append("Some college/associate")
        elif value == 21:
            labels.append("Bachelor degree")
        else:
            labels.append("Graduate degree")
    return pd.Series(labels, index=values.index)


def relp_group(values: pd.Series) -> pd.Series:
    mapping = {
        0: "Householder/spouse",
        1: "Householder/spouse",
        2: "Child",
        3: "Child",
        4: "Child",
        14: "Child",
        5: "Other relative",
        6: "Other relative",
        7: "Other relative",
        8: "Other relative",
        9: "Other relative",
        10: "Other relative",
        11: "Nonrelative/partner/roommate",
        12: "Nonrelative/partner/roommate",
        13: "Nonrelative/partner/roommate",
        15: "Nonrelative/partner/roommate",
        16: "Group quarters",
        17: "Group quarters",
    }
    labels = []
    for value in _to_numeric(values):
        if pd.isna(value):
            labels.append("Unknown relationship")
        else:
            labels.append(mapping.get(int(value), "Unknown relationship"))
    return pd.Series(labels, index=values.index)


def rac1p_group(values: pd.Series) -> pd.Series:
    mapping = {
        1: "White alone",
        2: "Black alone",
        3: "AIAN/NHPI/other race alone",
        4: "AIAN/NHPI/other race alone",
        5: "AIAN/NHPI/other race alone",
        6: "Asian alone",
        7: "AIAN/NHPI/other race alone",
        8: "AIAN/NHPI/other race alone",
        9: "Two or more races",
    }
    labels = []
    for value in _to_numeric(values):
        if pd.isna(value):
            labels.append("Unknown race")
        else:
            labels.append(mapping.get(int(value), "Unknown race"))
    return pd.Series(labels, index=values.index)


def cow_group(values: pd.Series) -> pd.Series:
    mapping = {
        1: "Private employee",
        2: "Private employee",
        3: "Government employee",
        4: "Government employee",
        5: "Government employee",
        6: "Self-employed",
        7: "Self-employed",
        8: "Unpaid family worker",
        9: "Unemployed/NILF code",
    }
    labels = []
    for value in _to_numeric(values):
        if pd.isna(value):
            labels.append("Unknown class of worker")
        else:
            labels.append(mapping.get(int(value), "Unknown class of worker"))
    return pd.Series(labels, index=values.index)


def add_semantic_features(frame: pd.DataFrame, target_state: str) -> pd.DataFrame:
    """Return a copy with semantic grouped ACSIncome features added."""

    result = frame.copy()
    result["COW_GROUP"] = cow_group(result["COW"])
    result["SCHL_GROUP"] = schl_group(result["SCHL"])
    result["OCCP_GROUP"] = occp_group(result["OCCP"])
    result["POBP_GROUP"] = pobp_group(result["POBP"], target_state=target_state)
    result["RELP_GROUP"] = relp_group(result["RELP"])
    result["RAC1P_GROUP"] = rac1p_group(result["RAC1P"])
    return result


def make_semantic_preprocessor() -> ColumnTransformer:
    """Build the 65-feature semantic encoder for the current PA split."""

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                SEMANTIC_CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0.3,
    )


def feature_block_dimensions(preprocessor: ColumnTransformer) -> dict[str, int]:
    """Return fitted feature counts by semantic block."""

    encoder = preprocessor.named_transformers_["cat"]
    dims = {"numeric": len(NUMERIC_FEATURES)}
    for name, categories in zip(SEMANTIC_CATEGORICAL_FEATURES, encoder.categories_):
        dims[name] = int(len(categories))
    dims["total"] = int(len(preprocessor.get_feature_names_out()))
    return dims
