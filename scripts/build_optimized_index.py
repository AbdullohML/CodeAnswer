"""Build PCA-compressed embeddings and the HNSW index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codeanswer.config import load_config
from codeanswer.optimization import (
    build_hnsw_index,
    fit_pca_embeddings,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the optimized PCA and HNSW "
            "retrieval artifacts."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the project YAML configuration.",
    )
    parser.add_argument(
        "--embeddings",
        default=None,
        help="Optional 384-dimensional embedding path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing optimized artifacts.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)

    source_embeddings = (
        Path(arguments.embeddings).expanduser()
        if arguments.embeddings
        else (
            config.paths.artifacts_dir
            / "embeddings"
            / "question_body_minilm_384.f32"
        )
    )

    optimized_dir = (
        config.paths.artifacts_dir
        / "optimized"
    )

    reduced_embeddings = (
        optimized_dir
        / "question_body_pca_128.f32"
    )
    pca_model = (
        optimized_dir
        / "pca_384_to_128.joblib"
    )
    hnsw_index = (
        optimized_dir
        / "hnsw_pca_128.faiss"
    )

    pca_manifest = fit_pca_embeddings(
        embedding_path=source_embeddings,
        output_path=reduced_embeddings,
        model_path=pca_model,
        input_dimension=(
            config.embedding.dimension
        ),
        output_dimension=(
            config.pca.output_dimension
        ),
        sample_size=(
            config.pca.sample_size
        ),
        transform_batch_size=(
            config.pca.transform_batch_size
        ),
        seed=config.dataset.random_seed,
        overwrite=arguments.overwrite,
    )

    hnsw_manifest = build_hnsw_index(
        embedding_path=reduced_embeddings,
        output_path=hnsw_index,
        dimension=(
            config.pca.output_dimension
        ),
        m=config.hnsw.m,
        ef_construction=(
            config.hnsw.ef_construction
        ),
        overwrite=arguments.overwrite,
    )

    print(
        json.dumps(
            {
                "pca": pca_manifest,
                "hnsw": hnsw_manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
