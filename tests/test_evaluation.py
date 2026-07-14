"""Tests for retrieval evaluation metrics."""

from __future__ import annotations

import faiss
import numpy as np
import pytest

from codeanswer.evaluation import (
    benchmark_index,
    evaluate_known_item,
    known_item_ranks,
)


def test_known_item_ranks() -> None:
    retrieved = np.asarray(
        [
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
            [40, 41, 42],
        ],
        dtype=np.int64,
    )

    relevant = np.asarray(
        [10, 21, 32, 99],
        dtype=np.int64,
    )

    ranks = known_item_ranks(
        retrieved,
        relevant,
    )

    assert ranks.tolist() == [
        1,
        2,
        3,
        0,
    ]


def test_known_item_metrics() -> None:
    retrieved = np.asarray(
        [
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
            [40, 41, 42],
        ],
        dtype=np.int64,
    )

    relevant = np.asarray(
        [10, 21, 32, 99],
        dtype=np.int64,
    )

    metrics, per_query = evaluate_known_item(
        retrieved,
        relevant,
        recall_ks=(1, 2, 3),
        metric_cutoff=3,
    )

    assert metrics["queries"] == 4
    assert metrics["recall@1"] == 0.25
    assert metrics["recall@2"] == 0.5
    assert metrics["recall@3"] == 0.75

    expected_mrr = (
        1.0
        + 1.0 / 2
        + 1.0 / 3
    ) / 4

    expected_ndcg = (
        1.0
        + 1.0 / np.log2(3)
        + 1.0 / np.log2(4)
    ) / 4

    assert metrics["mrr@3"] == pytest.approx(
        expected_mrr,
        abs=1e-6,
    )
    assert metrics["ndcg@3"] == pytest.approx(
        expected_ndcg,
        abs=1e-6,
    )

    assert per_query["rank"].tolist() == [
        1,
        2,
        3,
        0,
    ]


def test_rejects_incompatible_rankings() -> None:
    retrieved = np.asarray(
        [
            [1, 2],
            [3, 4],
        ]
    )
    relevant = np.asarray([1])

    with pytest.raises(
        ValueError,
        match="number",
    ):
        known_item_ranks(
            retrieved,
            relevant,
        )


def test_benchmark_index() -> None:
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.70710677, 0.70710677],
        ],
        dtype=np.float32,
    )

    index = faiss.IndexFlatIP(2)
    index.add(vectors)

    benchmark = benchmark_index(
        index,
        vectors,
        top_k=2,
        latency_queries=3,
        warmup_queries=1,
    )

    assert benchmark["batch_queries"] == 3
    assert benchmark["latency_queries"] == 3
    assert benchmark["search_qps"] > 0
    assert benchmark["search_ms"]["mean"] >= 0
    assert benchmark["search_ms"]["p50"] >= 0
    assert benchmark["search_ms"]["p95"] >= 0
