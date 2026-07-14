"""Compare Flat-384, Flat-PCA128, and HNSW-PCA128."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import faiss
import numpy as np
import pandas as pd

from codeanswer.comparison import (
    ann_overlap_at_k,
    evaluate_retriever,
)
from codeanswer.config import load_config
from codeanswer.embeddings import (
    SentenceTransformerEncoder,
)
from codeanswer.indexing import (
    load_index,
    open_embeddings,
)
from codeanswer.optimization import (
    load_pca_model,
    transform_queries,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CodeAnswer retrieval systems."
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )

    return parser.parse_args()


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def load_queries(
    corpus_path: Path,
    *,
    query_count: int,
    seed: int,
) -> pd.DataFrame:
    connection = duckdb.connect()

    try:
        return connection.execute(
            f"""
            SELECT
                doc_id,
                question_id,
                title
            FROM read_parquet(
                '{sql_path(corpus_path)}'
            )
            ORDER BY
                hash(question_id + {int(seed)}),
                question_id
            LIMIT {int(query_count)}
            """
        ).df()

    finally:
        connection.close()


def build_flat_index(
    embeddings: np.ndarray,
    dimension: int,
) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(dimension)

    for start in range(
        0,
        embeddings.shape[0],
        10_000,
    ):
        end = min(
            start + 10_000,
            embeddings.shape[0],
        )

        index.add(
            np.ascontiguousarray(
                embeddings[start:end],
                dtype=np.float32,
            )
        )

    return index


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)

    corpus_path = (
        config.paths.artifacts_dir
        / "corpus"
        / "qa_corpus.parquet"
    )

    exact_index_path = (
        config.paths.artifacts_dir
        / "indexes"
        / "flat_ip_384.faiss"
    )

    optimized_dir = (
        config.paths.artifacts_dir
        / "optimized"
    )

    reduced_embeddings_path = (
        optimized_dir
        / "question_body_pca_128.f32"
    )
    pca_model_path = (
        optimized_dir
        / "pca_384_to_128.joblib"
    )
    hnsw_index_path = (
        optimized_dir
        / "hnsw_pca_128.faiss"
    )

    output_dir = (
        Path(arguments.output_dir).expanduser()
        if arguments.output_dir
        else (
            config.paths.results_dir
            / "iteration_2"
            / "retrieval_optimization"
        )
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    queries = load_queries(
        corpus_path,
        query_count=(
            config.evaluation.retrieval_queries
        ),
        seed=config.dataset.random_seed,
    )

    encoder = SentenceTransformerEncoder(
        model_name=config.embedding.model_name,
        max_sequence_length=(
            config.embedding.max_sequence_length
        ),
        device=arguments.device,
    )

    query_vectors_384 = encoder.encode(
        queries["title"].tolist(),
        batch_size=config.embedding.batch_size,
    )

    relevant_ids = queries[
        "doc_id"
    ].to_numpy(dtype=np.int64)

    exact_index = load_index(
        exact_index_path,
        expected_objects=(
            config.dataset.target_objects
        ),
        expected_dimension=(
            config.embedding.dimension
        ),
    )

    exact_result, exact_ids = evaluate_retriever(
        system_name="Flat-384",
        index=exact_index,
        query_vectors=query_vectors_384,
        relevant_ids=relevant_ids,
        latency_queries=(
            config.evaluation.latency_queries
        ),
    )

    exact_result["dimension"] = 384
    exact_result["index_mib"] = round(
        exact_index_path.stat().st_size
        / 1024**2,
        2,
    )
    exact_result["ann_overlap@10"] = 1.0

    pca = load_pca_model(
        pca_model_path
    )
    query_vectors_128 = transform_queries(
        pca,
        query_vectors_384,
    )

    reduced_embeddings = open_embeddings(
        reduced_embeddings_path,
        objects=config.dataset.target_objects,
        dimension=config.pca.output_dimension,
    )

    flat_128_index = build_flat_index(
        reduced_embeddings,
        config.pca.output_dimension,
    )

    flat_128_result, flat_128_ids = (
        evaluate_retriever(
            system_name="Flat-PCA128",
            index=flat_128_index,
            query_vectors=query_vectors_128,
            relevant_ids=relevant_ids,
            latency_queries=(
                config.evaluation.latency_queries
            ),
        )
    )

    flat_128_result["dimension"] = 128
    flat_128_result["index_mib"] = round(
        reduced_embeddings_path.stat().st_size
        / 1024**2,
        2,
    )
    flat_128_result["ann_overlap@10"] = (
        ann_overlap_at_k(
            exact_ids,
            flat_128_ids,
            k=10,
        )
    )

    hnsw_index = load_index(
        hnsw_index_path,
        expected_objects=(
            config.dataset.target_objects
        ),
        expected_dimension=(
            config.pca.output_dimension
        ),
    )

    results = [
        exact_result,
        flat_128_result,
    ]

    for ef_search in [16, 32, 64, 128]:
        hnsw_index.hnsw.efSearch = ef_search

        result, retrieved_ids = evaluate_retriever(
            system_name=(
                f"HNSW-PCA128-ef{ef_search}"
            ),
            index=hnsw_index,
            query_vectors=query_vectors_128,
            relevant_ids=relevant_ids,
            latency_queries=(
                config.evaluation.latency_queries
            ),
        )

        result["dimension"] = 128
        result["ef_search"] = ef_search
        result["index_mib"] = round(
            hnsw_index_path.stat().st_size
            / 1024**2,
            2,
        )
        result["ann_overlap@10"] = (
            ann_overlap_at_k(
                flat_128_ids,
                retrieved_ids,
                k=10,
            )
        )

        results.append(result)

    comparison = pd.DataFrame(results)

    comparison.to_csv(
        output_dir / "comparison.csv",
        index=False,
    )

    payload = {
        "evaluation_type": (
            "known_item_title_to_body"
        ),
        "queries": len(queries),
        "systems": results,
    }

    (
        output_dir / "results.json"
    ).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
