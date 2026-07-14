"""Tests for retrieval-system comparison."""

from __future__ import annotations

import faiss
import numpy as np

from codeanswer.comparison import (
    ann_overlap_at_k,
    evaluate_retriever,
)


def test_ann_overlap_at_k() -> None:
    reference = np.asarray(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    candidate = np.asarray(
        [
            [1, 3, 8],
            [4, 9, 10],
        ]
    )

    overlap = ann_overlap_at_k(
        reference,
        candidate,
        k=3,
    )

    assert overlap == 0.5


def test_evaluate_retriever() -> None:
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

    relevant_ids = np.asarray(
        [0, 1, 2],
        dtype=np.int64,
    )

    result, retrieved_ids = evaluate_retriever(
        system_name="test",
        index=index,
        query_vectors=vectors,
        relevant_ids=relevant_ids,
        top_k=3,
        latency_queries=3,
    )

    assert result["system"] == "test"
    assert result["recall@1"] == 1.0
    assert result["recall@10"] == 1.0
    assert result["search_qps"] > 0
    assert retrieved_ids.shape == (3, 3)
