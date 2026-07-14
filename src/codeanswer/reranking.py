"""Cross-encoder reranking and threshold calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
import pandas as pd


class PairScorer(Protocol):
    """Interface for query-document pair scoring."""

    model_name: str
    device: str

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> np.ndarray:
        """Return one relevance score per pair."""


@dataclass
class CrossEncoderScorer:
    """Sentence Transformers cross-encoder scorer."""

    model_name: str
    device: str | None = None
    max_length: int = 384

    _model: object = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        if self.device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
        )

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> np.ndarray:
        """Score query-document pairs."""

        scores = self._model.predict(
            list(pairs),
            batch_size=batch_size,
            show_progress_bar=False,
        )

        return np.asarray(
            scores,
            dtype=np.float32,
        ).reshape(-1)


def rerank_candidates(
    query: str,
    candidates: pd.DataFrame,
    *,
    scorer: PairScorer,
    top_k: int,
    batch_size: int,
    answer_score_weight: float = 0.03,
) -> pd.DataFrame:
    """Rerank retrieved questions with a cross-encoder."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    required_columns = {
        "title",
        "question_body",
        "answer_score",
    }

    missing = required_columns - set(
        candidates.columns
    )

    if missing:
        raise ValueError(
            "Candidates are missing columns: "
            + ", ".join(sorted(missing))
        )

    if candidates.empty:
        raise ValueError(
            "At least one candidate is required."
        )

    pairs = [
        (
            query,
            f"{row.title}\n{row.question_body}",
        )
        for row in candidates.itertuples(
            index=False
        )
    ]

    scores = scorer.predict(
        pairs,
        batch_size=batch_size,
    )

    if len(scores) != len(candidates):
        raise ValueError(
            "The scorer returned an invalid "
            "number of scores."
        )

    result = candidates.copy()
    result["reranker_score"] = scores

    quality_bonus = np.log1p(
        np.maximum(
            result["answer_score"].to_numpy(
                dtype=float
            ),
            0.0,
        )
    )

    result["final_score"] = (
        result["reranker_score"]
        + answer_score_weight * quality_bonus
    )

    return (
        result.sort_values(
            "final_score",
            ascending=False,
        )
        .head(top_k)
        .reset_index(drop=True)
    )


def threshold_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    """Evaluate one answerability threshold."""

    expected = np.asarray(
        labels,
        dtype=np.int64,
    )
    values = np.asarray(
        scores,
        dtype=np.float64,
    )

    if expected.ndim != 1 or values.ndim != 1:
        raise ValueError(
            "labels and scores must be one-dimensional."
        )

    if len(expected) != len(values):
        raise ValueError(
            "labels and scores must have equal length."
        )

    if not set(np.unique(expected)).issubset(
        {0, 1}
    ):
        raise ValueError(
            "labels must contain only zero and one."
        )

    predictions = (
        values >= threshold
    ).astype(np.int64)

    true_positive = int(
        (
            (predictions == 1)
            & (expected == 1)
        ).sum()
    )
    false_positive = int(
        (
            (predictions == 1)
            & (expected == 0)
        ).sum()
    )
    false_negative = int(
        (
            (predictions == 0)
            & (expected == 1)
        ).sum()
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(
            (predictions == expected).mean()
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def calibrate_answerability_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """Select the threshold with the best F1 and accuracy."""

    values = np.asarray(
        scores,
        dtype=np.float64,
    )

    if values.size == 0:
        raise ValueError(
            "At least one score is required."
        )

    candidates = [
        threshold_metrics(
            labels,
            values,
            threshold=float(threshold),
        )
        for threshold in sorted(
            np.unique(values)
        )
    ]

    return max(
        candidates,
        key=lambda item: (
            item["f1"],
            item["accuracy"],
            -item["threshold"],
        ),
    )
