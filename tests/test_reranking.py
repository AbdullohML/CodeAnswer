"""Tests for reranking and calibration."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from codeanswer.reranking import (
    calibrate_answerability_threshold,
    rerank_candidates,
)


class FakeScorer:
    model_name = "fake"
    device = "cpu"

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> np.ndarray:
        del batch_size

        return np.asarray(
            [
                float(len(document))
                for _, document in pairs
            ],
            dtype=np.float32,
        )


def test_rerank_candidates() -> None:
    candidates = pd.DataFrame(
        {
            "title": ["A", "Long title"],
            "question_body": [
                "short",
                "much longer body",
            ],
            "answer_score": [10, 0],
        }
    )

    result = rerank_candidates(
        "query",
        candidates,
        scorer=FakeScorer(),
        top_k=1,
        batch_size=2,
    )

    assert len(result) == 1
    assert result.iloc[0]["title"] == (
        "Long title"
    )
    assert "reranker_score" in result
    assert "final_score" in result


def test_calibrate_threshold() -> None:
    labels = np.asarray(
        [1, 1, 1, 0, 0],
    )
    scores = np.asarray(
        [8.0, 5.0, 2.0, -4.0, -8.0],
    )

    result = (
        calibrate_answerability_threshold(
            labels,
            scores,
        )
    )

    assert result["f1"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["threshold"] == 2.0
