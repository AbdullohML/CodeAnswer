"""Evaluate the exact CodeAnswer retrieval baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from codeanswer.config import load_config
from codeanswer.embeddings import (
    SentenceTransformerEncoder,
)
from codeanswer.evaluation import (
    benchmark_index,
    evaluate_known_item,
)
from codeanswer.indexing import (
    load_index,
    search_index,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate known-item retrieval quality and latency."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the project YAML configuration.",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Optional corpus Parquet path.",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="Optional FAISS index path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional result directory.",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=None,
        help="Number of evaluation queries.",
    )
    parser.add_argument(
        "--latency-queries",
        type=int,
        default=None,
        help="Queries used for latency measurement.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Bi-encoder inference device.",
    )

    return parser.parse_args()


def sql_path(path: Path) -> str:
    """Escape a path for a DuckDB SQL string."""

    return str(path.resolve()).replace(
        "'",
        "''",
    )


def load_evaluation_queries(
    corpus_path: Path,
    *,
    query_count: int,
    seed: int,
) -> pd.DataFrame:
    """Load a deterministic title-query evaluation set."""

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}"
        )

    if query_count <= 0:
        raise ValueError(
            "query_count must be greater than zero."
        )

    connection = duckdb.connect()

    try:
        evaluation = connection.execute(
            f"""
            SELECT
                doc_id,
                question_id,
                title
            FROM read_parquet(
                '{sql_path(corpus_path)}'
            )
            ORDER BY
                hash(
                    question_id
                    + {int(seed)}
                ),
                question_id
            LIMIT {int(query_count)}
            """
        ).df()

    finally:
        connection.close()

    if len(evaluation) != query_count:
        raise RuntimeError(
            f"Requested {query_count:,} queries, "
            f"but loaded {len(evaluation):,}."
        )

    return evaluation


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write formatted JSON."""

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    arguments = parse_arguments()
    config = load_config(
        arguments.config
    )

    corpus_path = (
        Path(arguments.corpus).expanduser()
        if arguments.corpus
        else (
            config.paths.artifacts_dir
            / "corpus"
            / "qa_corpus.parquet"
        )
    ).resolve()

    index_path = (
        Path(arguments.index).expanduser()
        if arguments.index
        else (
            config.paths.artifacts_dir
            / "indexes"
            / "flat_ip_384.faiss"
        )
    ).resolve()

    output_dir = (
        Path(arguments.output_dir).expanduser()
        if arguments.output_dir
        else (
            config.paths.results_dir
            / "iteration_1"
            / "exact_dense_baseline"
        )
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    query_count = (
        arguments.query_count
        or config.evaluation.retrieval_queries
    )
    latency_queries = (
        arguments.latency_queries
        or config.evaluation.latency_queries
    )

    evaluation_queries = (
        load_evaluation_queries(
            corpus_path,
            query_count=query_count,
            seed=config.dataset.random_seed,
        )
    )

    encoder = SentenceTransformerEncoder(
        model_name=(
            config.embedding.model_name
        ),
        max_sequence_length=(
            config.embedding
            .max_sequence_length
        ),
        device=arguments.device,
    )

    query_vectors = encoder.encode(
        evaluation_queries[
            "title"
        ].tolist(),
        batch_size=(
            config.embedding.batch_size
        ),
    )

    index = load_index(
        index_path,
        expected_objects=(
            config.dataset.target_objects
        ),
        expected_dimension=(
            config.embedding.dimension
        ),
    )

    _, retrieved_ids = search_index(
        index,
        query_vectors,
        top_k=10,
    )

    quality_metrics, per_query = (
        evaluate_known_item(
            retrieved_ids,
            evaluation_queries[
                "doc_id"
            ].to_numpy(),
            recall_ks=(1, 5, 10),
            metric_cutoff=10,
        )
    )

    benchmark = benchmark_index(
        index,
        query_vectors,
        top_k=10,
        latency_queries=latency_queries,
    )

    per_query.insert(
        1,
        "question_id",
        evaluation_queries[
            "question_id"
        ].to_numpy(),
    )
    per_query.insert(
        2,
        "query",
        evaluation_queries[
            "title"
        ].to_numpy(),
    )

    per_query_path = (
        output_dir
        / "per_query_metrics.csv"
    )
    results_path = (
        output_dir
        / "results.json"
    )

    per_query.to_csv(
        per_query_path,
        index=False,
    )

    results = {
        "system": "Flat-384",
        "evaluation_type": (
            "known_item_title_to_body"
        ),
        "corpus_path": str(corpus_path),
        "index_path": str(index_path),
        "embedding_model": (
            config.embedding.model_name
        ),
        "device": encoder.device,
        "quality": quality_metrics,
        "performance": benchmark,
        "artifacts": {
            "per_query_metrics": str(
                per_query_path
            ),
        },
    }

    write_json(
        results_path,
        results,
    )

    print(
        json.dumps(
            results,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
