"""Command-line entry point for building the Q&A corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codeanswer.config import load_config
from codeanswer.data import build_corpus


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the CodeAnswer Stack Overflow corpus."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the project YAML configuration.",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help=(
            "Directory containing Questions.csv "
            "and Answers.csv."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output Parquet path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing corpus.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)

    output_path = (
        Path(arguments.output).expanduser()
        if arguments.output
        else (
            config.paths.artifacts_dir
            / "corpus"
            / "qa_corpus.parquet"
        )
    )

    manifest = build_corpus(
        source_dir=arguments.source_dir,
        output_path=output_path,
        target_objects=(
            config.dataset.target_objects
        ),
        seed=config.dataset.random_seed,
        overwrite=arguments.overwrite,
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
