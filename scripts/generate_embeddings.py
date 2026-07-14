"""Generate CodeAnswer question-body embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codeanswer.config import load_config
from codeanswer.embeddings import (
    generate_embeddings,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate normalized MiniLM embeddings."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration.",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Optional corpus Parquet path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output .f32 path.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Model inference device.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the configured batch size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace completed embeddings.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)

    corpus_path = (
        Path(arguments.corpus).expanduser()
        if arguments.corpus
        else (
            config.paths.artifacts_dir
            / "corpus"
            / "qa_corpus.parquet"
        )
    )

    output_path = (
        Path(arguments.output).expanduser()
        if arguments.output
        else (
            config.paths.artifacts_dir
            / "embeddings"
            / "question_body_minilm_384.f32"
        )
    )

    manifest = generate_embeddings(
        corpus_path=corpus_path,
        output_path=output_path,
        model_name=(
            config.embedding.model_name
        ),
        expected_dimension=(
            config.embedding.dimension
        ),
        batch_size=(
            arguments.batch_size
            or config.embedding.batch_size
        ),
        max_sequence_length=(
            config.embedding
            .max_sequence_length
        ),
        device=arguments.device,
        overwrite=arguments.overwrite,
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
