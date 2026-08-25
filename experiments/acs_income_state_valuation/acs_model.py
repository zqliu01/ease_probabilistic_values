"""Modeling helpers for the ACSIncome state-source valuation experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import acs_data
import semantic_acs_encoder


NUMERIC_FEATURES = ["AGEP", "WKHP"]
CATEGORICAL_FEATURES = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]


@dataclass(frozen=True)
class EncodedACSSplit:
    encoded_train: dict[str, sparse.csr_matrix]
    train_labels: dict[str, np.ndarray]
    encoded_eval: sparse.csr_matrix
    eval_y: np.ndarray
    label_rates: dict[str, float]
    feature_block_dims: dict[str, int]
    processed_split: acs_data.ACSProcessedSplit
    preprocessor: ColumnTransformer


def make_preprocessor(encoder: str) -> ColumnTransformer:
    if encoder == "semantic65":
        return semantic_acs_encoder.make_semantic_preprocessor()
    if encoder != "full":
        raise ValueError(f"Unknown encoder {encoder!r}.")
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0.3,
    )


def _apply_encoder_features(
    *,
    train_frames: dict[str, pd.DataFrame],
    eval_x: pd.DataFrame,
    encoder: str,
    target_state: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    if encoder != "semantic65":
        return train_frames, eval_x
    return (
        {
            state: semantic_acs_encoder.add_semantic_features(frame, target_state)
            for state, frame in train_frames.items()
        },
        semantic_acs_encoder.add_semantic_features(eval_x, target_state),
    )


def prepare_encoded_acs_split(args: Any) -> EncodedACSSplit:
    processed = acs_data.load_or_prepare_processed_split(
        survey_year=args.survey_year,
        target_state=args.target_state,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
        data_dir=args.data_dir,
        download=args.download,
    )
    train_frames, eval_x = _apply_encoder_features(
        train_frames=processed.train_frames,
        eval_x=processed.eval_x,
        encoder=args.encoder,
        target_state=args.target_state,
    )

    preprocessor = make_preprocessor(args.encoder)
    preprocessor.fit(pd.concat(train_frames.values(), ignore_index=True))
    if args.encoder == "semantic65":
        feature_block_dims = semantic_acs_encoder.feature_block_dimensions(preprocessor)
    else:
        feature_block_dims = {"total": int(len(preprocessor.get_feature_names_out()))}

    encoded_train = {
        state: preprocessor.transform(frame)
        for state, frame in train_frames.items()
    }
    encoded_train = {
        state: matrix if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
        for state, matrix in encoded_train.items()
    }
    encoded_eval = preprocessor.transform(eval_x)
    if not sparse.issparse(encoded_eval):
        encoded_eval = sparse.csr_matrix(encoded_eval)

    return EncodedACSSplit(
        encoded_train=encoded_train,
        train_labels=processed.train_labels,
        encoded_eval=encoded_eval,
        eval_y=processed.eval_y,
        label_rates=processed.label_rates,
        feature_block_dims=feature_block_dims,
        processed_split=processed,
        preprocessor=preprocessor,
    )


def transform_eval_frame(
    eval_x: pd.DataFrame,
    *,
    preprocessor: ColumnTransformer,
    encoder: str,
    target_state: str,
) -> sparse.csr_matrix:
    """Encode an alternate evaluation frame with a fitted ACS preprocessor."""

    if encoder == "semantic65":
        eval_x = semantic_acs_encoder.add_semantic_features(eval_x, target_state)
    encoded_eval = preprocessor.transform(eval_x)
    if sparse.issparse(encoded_eval):
        return encoded_eval.tocsr()
    return sparse.csr_matrix(encoded_eval)


def constant_utility(eval_y: np.ndarray, p: float) -> tuple[float, float]:
    scores = np.full(len(eval_y), float(np.clip(p, 1e-6, 1 - 1e-6)))
    loss = float(log_loss(eval_y, scores, labels=[0, 1]))
    return -loss, loss
