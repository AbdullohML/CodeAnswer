"""Retrieval quality and latency evaluation utilities."""

from __future__ import annotations

import math
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from codeanswer.indexing import search_index


def known_item_ranks(
    retrieved_ids: np.ndarray,
    relevant_ids: np.ndarray,
) -> np.ndarray:
    """Return the one-indexed rank of each relevant document.

    A rank of zero means that the relevant document was not retrieved.
    """

    retrieved = np.asarray(
        retrieved_ids,
        dtype=np.int64,
    )
    relevant = np.asarray(
        relevant_ids,
        dtype=np.int64,
    )

    if retrieved.ndim != 2:
        raise ValueError(
            "retrieved_ids must be a two-dimensional array."
        )

    if relevant.ndim != 1:
        raise ValueError(
            "relevant_ids must be a one-dimensional array."
        )

    if retrieved.shape[0] != relevant.shape[0]:
        raise ValueError(
            "The number of retrieved rankings must match "
            "the number of relevant document IDs."
        )

    matches = retrieved == relevant[:, None]
    found = matches.any(axis=1)

    ranks = np.zeros(
        relevant.shape[0],
        dtype=np.int64,
    )

    ranks[found] = (
        matches[found].argmax(axis=1) + 1
    )

    return ranks


def evaluate_known_item(
    retrieved_ids: np.ndarray,
    relevant_ids: np.ndarray,
    *,
    recall_ks: Iterable[int] = (1, 5, 10),
    metric_cutoff: int = 10,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate known-item retrieval rankings."""

    if metric_cutoff <= 0:
        raise ValueError(
            "metric_cutoff must be greater than zero."
        )

    recall_values = sorted(
        set(int(value) for value in recall_ks)
    )

    if not recall_values or recall_values[0] <= 0:
        raise ValueError(
            "All recall cutoffs must be greater than zero."
        )

    ranks = known_item_ranks(
        retrieved_ids,
        relevant_ids,
    )

    metrics: dict[str, float | int] = {
        "queries": int(len(ranks)),
    }

    for cutoff in recall_values:
        success = (
            (ranks > 0)
            & (ranks <= cutoff)
        )

        metrics[f"recall@{cutoff}"] = round(
            float(success.mean()),
            6,
        )

    reciprocal_ranks = np.where(
        (ranks > 0)
        & (ranks <= metric_cutoff),
        1.0 / np.maximum(ranks, 1),
        0.0,
    )

    discounted_gains = np.where(
        (ranks > 0)
        & (ranks <= metric_cutoff),
        1.0
        / np.log2(
            np.maximum(ranks, 1) + 1
        ),
        0.0,
    )

    metrics[f"mrr@{metric_cutoff}"] = round(
        float(reciprocal_ranks.mean()),
        6,
    )
    metrics[f"ndcg@{metric_cutoff}"] = round(
        float(discounted_gains.mean()),
        6,
    )

    per_query = pd.DataFrame(
        {
            "query_index": np.arange(
                len(ranks),
                dtype=np.int64,
            ),
            "relevant_doc_id": np.asarray(
                relevant_ids,
                dtype=np.int64,
            ),
            "rank": ranks,
            f"reciprocal_rank@{metric_cutoff}": (
                reciprocal_ranks
            ),
            f"ndcg@{metric_cutoff}": (
                discounted_gains
            ),
        }
    )

    for cutoff in recall_values:
        per_query[f"recall@{cutoff}"] = (
            (ranks > 0)
            & (ranks <= cutoff)
        ).astype(float)

    return metrics, per_query


def _latency_summary(
    latency_ms: list[float],
) -> dict[str, float]:
    """Summarize latency measurements."""

    if not latency_ms:
        raise ValueError(
            "At least one latency value is required."
        )

    values = np.asarray(
        latency_ms,
        dtype=np.float64,
    )

    return {
        "mean": round(
            float(values.mean()),
            3,
        ),
        "p50": round(
            float(np.quantile(values, 0.50)),
            3,
        ),
        "p95": round(
            float(np.quantile(values, 0.95)),
            3,
        ),
    }


def benchmark_index(
    index: Any,
    query_vectors: np.ndarray,
    *,
    top_k: int = 10,
    latency_queries: int = 100,
    warmup_queries: int = 3,
) -> dict[str, Any]:
    """Benchmark batch throughput and single-query latency."""

    queries = np.asarray(
        query_vectors,
        dtype=np.float32,
    )

    if queries.ndim != 2:
        raise ValueError(
            "query_vectors must be a two-dimensional array."
        )

    if queries.shape[0] == 0:
        raise ValueError(
            "At least one query vector is required."
        )

    if latency_queries <= 0:
        raise ValueError(
            "latency_queries must be greater than zero."
        )

    warmup_count = min(
        warmup_queries,
        queries.shape[0],
    )

    for query in queries[:warmup_count]:
        search_index(
            index,
            query,
            top_k=top_k,
        )

    batch_started_at = time.perf_counter()

    search_index(
        index,
        queries,
        top_k=top_k,
    )

    batch_seconds = (
        time.perf_counter()
        - batch_started_at
    )

    measured_queries = min(
        latency_queries,
        queries.shape[0],
    )

    latency_ms = []

    for query in queries[:measured_queries]:
        started_at = time.perf_counter()

        search_index(
            index,
            query,
            top_k=top_k,
        )

        latency_ms.append(
            (
                time.perf_counter()
                - started_at
            )
            * 1_000
        )

    query_rate = (
        queries.shape[0] / batch_seconds
        if batch_seconds > 0
        else math.inf
    )

    return {
        "batch_queries": int(
            queries.shape[0]
        ),
        "batch_search_seconds": round(
            batch_seconds,
            6,
        ),
        "search_qps": round(
            float(query_rate),
            2,
        ),
        "latency_queries": measured_queries,
        "search_ms": _latency_summary(
            latency_ms
        ),
    }
