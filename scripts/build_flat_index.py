"""Build the exact CodeAnswer FAISS index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codeanswer.config import load_config
from codeanswer.indexing import (
    build_flat_index,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the exact FAISS IndexFlatIP baseline."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the project configuration.",
    )
    parser.add_argument(
        "--embeddings",
        default=None,
        help="Optional input .f32 embedding path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output FAISS index path.",
    )
    parser.add_argument(
        "--add-batch-size",
        type=int,
        default=10_000,
        help="Vectors added to FAISS per batch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing index.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(
        arguments.config
    )

    embedding_path = (
        Path(arguments.embeddings).expanduser()
        if arguments.embeddings
        else (
            config.paths.artifacts_dir
            / "embeddings"
            / "question_body_minilm_384.f32"
        )
    )

    output_path = (
        Path(arguments.output).expanduser()
        if arguments.output
        else (
            config.paths.artifacts_dir
            / "indexes"
            / "flat_ip_384.faiss"
        )
    )

    manifest = build_flat_index(
        embedding_path=embedding_path,
        output_path=output_path,
        dimension=config.embedding.dimension,
        add_batch_size=(
            arguments.add_batch_size
        ),
        overwrite=arguments.overwrite,
    )

    print(
        json.dumps(
            manifest,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
