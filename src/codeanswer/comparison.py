"""Compare exact and approximate retrieval systems."""

from __future__ import annotations

from typing import Any

import numpy as np

from codeanswer.evaluation import (
    benchmark_index,
    evaluate_known_item,
)
from codeanswer.indexing import search_index


def ann_overlap_at_k(
    reference_ids: np.ndarray,
    candidate_ids: np.ndarray,
    *,
    k: int,
) -> float:
    """Calculate average top-k overlap between two rankings."""

    reference = np.asarray(reference_ids)
    candidate = np.asarray(candidate_ids)

    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError(
            "Both ranking arrays must be two-dimensional."
        )

    if reference.shape[0] != candidate.shape[0]:
        raise ValueError(
            "Ranking arrays must contain the same number of queries."
        )

    if k <= 0:
        raise ValueError("k must be positive.")

    if reference.shape[1] < k or candidate.shape[1] < k:
        raise ValueError(
            "Ranking arrays contain fewer than k results."
        )

    overlaps = []

    for reference_row, candidate_row in zip(
        reference[:, :k],
        candidate[:, :k],
        strict=True,
    ):
        overlaps.append(
            len(
                set(map(int, reference_row))
                & set(map(int, candidate_row))
            )
            / k
        )

    return round(float(np.mean(overlaps)), 6)


def evaluate_retriever(
    *,
    system_name: str,
    index: Any,
    query_vectors: np.ndarray,
    relevant_ids: np.ndarray,
    top_k: int = 10,
    latency_queries: int = 100,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate one retrieval configuration."""

    _, retrieved_ids = search_index(
        index,
        query_vectors,
        top_k=top_k,
    )

    quality, _ = evaluate_known_item(
        retrieved_ids,
        relevant_ids,
        recall_ks=(1, 5, 10),
        metric_cutoff=10,
    )

    performance = benchmark_index(
        index,
        query_vectors,
        top_k=top_k,
        latency_queries=latency_queries,
    )

    result = {
        "system": system_name,
        **quality,
        "batch_search_seconds": performance[
            "batch_search_seconds"
        ],
        "search_qps": performance["search_qps"],
        "search_mean_ms": performance[
            "search_ms"
        ]["mean"],
        "search_p50_ms": performance[
            "search_ms"
        ]["p50"],
        "search_p95_ms": performance[
            "search_ms"
        ]["p95"],
    }

    return result, retrieved_ids
