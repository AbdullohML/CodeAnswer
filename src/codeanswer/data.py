"""Build the Stack Overflow question-answer corpus."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb


def _find_csv(source_dir: Path, filename: str) -> Path:
    """Find a CSV file using a case-insensitive filename match."""

    source_dir = source_dir.expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {source_dir}"
        )

    exact_path = source_dir / filename

    if exact_path.exists():
        return exact_path

    matches = [
        path
        for path in source_dir.iterdir()
        if path.is_file()
        and path.name.casefold() == filename.casefold()
    ]

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} inside {source_dir}"
        )

    return matches[0]


def _sql_path(path: Path) -> str:
    """Escape a filesystem path for a DuckDB SQL string."""

    return str(path.resolve()).replace("'", "''")


def _clean_expression(column: str) -> str:
    """Return a DuckDB expression that removes HTML and extra spaces."""

    return (
        "trim("
        "regexp_replace("
        f"regexp_replace(coalesce({column}, ''), '<[^>]+>', ' ', 'g'), "
        "'[[:space:]]+', ' ', 'g'"
        ")"
        ")"
    )


def build_corpus(
    source_dir: str | Path,
    output_path: str | Path,
    target_objects: int = 520_000,
    seed: int = 20260714,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a deterministic question-answer corpus.

    Each question is joined to its highest-scored answer. The resulting
    corpus is sampled deterministically and written as a Parquet file.
    """

    if target_objects <= 0:
        raise ValueError("target_objects must be greater than zero.")

    source_path = Path(source_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()

    questions_path = _find_csv(
        source_path,
        "Questions.csv",
    )
    answers_path = _find_csv(
        source_path,
        "Answers.csv",
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. "
            "Use overwrite=True to replace it."
        )

    temporary_output = output.with_name(
        f".{output.name}.temporary"
    )

    if temporary_output.exists():
        temporary_output.unlink()

    connection = duckdb.connect()

    try:
        connection.execute(
            f"PRAGMA threads={max(1, os.cpu_count() or 1)}"
        )
        connection.execute("PRAGMA enable_progress_bar")

        question_title = _clean_expression("title_html")
        question_body = _clean_expression("question_html")
        answer_body = _clean_expression("answer_html")

        sampling_key = (
            f"((question_id * 1103515245 + {int(seed)}) "
            "% 2147483647)"
        )

        query = f"""
        COPY (
            WITH ranked_answers AS (
                SELECT
                    try_cast(ParentId AS BIGINT) AS question_id,
                    try_cast(Id AS BIGINT) AS answer_id,
                    coalesce(
                        try_cast(Score AS INTEGER),
                        0
                    ) AS answer_score,
                    Body AS answer_html,
                    row_number() OVER (
                        PARTITION BY try_cast(
                            ParentId AS BIGINT
                        )
                        ORDER BY
                            coalesce(
                                try_cast(Score AS INTEGER),
                                0
                            ) DESC,
                            try_cast(Id AS BIGINT) ASC
                    ) AS answer_rank
                FROM read_csv_auto(
                    '{_sql_path(answers_path)}',
                    header = true,
                    all_varchar = true,
                    ignore_errors = true
                )
                WHERE
                    ParentId IS NOT NULL
                    AND Id IS NOT NULL
                    AND Body IS NOT NULL
            ),
            joined AS (
                SELECT
                    try_cast(q.Id AS BIGINT) AS question_id,
                    coalesce(
                        try_cast(q.Score AS INTEGER),
                        0
                    ) AS question_score,
                    q.Title AS title_html,
                    q.Body AS question_html,
                    a.answer_id,
                    a.answer_score,
                    a.answer_html
                FROM read_csv_auto(
                    '{_sql_path(questions_path)}',
                    header = true,
                    all_varchar = true,
                    ignore_errors = true
                ) AS q
                INNER JOIN ranked_answers AS a
                    ON try_cast(q.Id AS BIGINT)
                        = a.question_id
                WHERE
                    a.answer_rank = 1
                    AND q.Id IS NOT NULL
                    AND q.Title IS NOT NULL
                    AND q.Body IS NOT NULL
            ),
            numbered AS (
                SELECT
                    row_number() OVER (
                        ORDER BY
                            {sampling_key},
                            question_id
                    ) - 1 AS doc_id,
                    question_id,
                    answer_id,
                    question_score,
                    answer_score,
                    {question_title} AS title,
                    {question_body} AS question_body,
                    {answer_body} AS answer,
                    'https://stackoverflow.com/questions/'
                        || cast(question_id AS VARCHAR)
                        AS url
                FROM joined
            )
            SELECT
                doc_id,
                question_id,
                answer_id,
                title,
                question_body,
                answer,
                question_score,
                answer_score,
                url
            FROM numbered
            WHERE doc_id < {int(target_objects)}
            ORDER BY doc_id
        )
        TO '{_sql_path(temporary_output)}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
        """

        connection.execute(query)

        statistics_row = connection.execute(
            f"""
            SELECT
                count(*) AS objects,
                round(avg(length(title)), 2)
                    AS title_chars_mean,
                round(avg(length(question_body)), 2)
                    AS body_chars_mean,
                round(avg(length(answer)), 2)
                    AS answer_chars_mean,
                round(avg(question_score), 3)
                    AS question_score_mean,
                round(avg(answer_score), 3)
                    AS answer_score_mean
            FROM read_parquet(
                '{_sql_path(temporary_output)}'
            )
            """
        ).fetchone()

        if statistics_row is None:
            raise RuntimeError(
                "DuckDB did not return corpus statistics."
            )

        statistics = {
            "objects": int(statistics_row[0]),
            "title_chars_mean": float(statistics_row[1]),
            "body_chars_mean": float(statistics_row[2]),
            "answer_chars_mean": float(statistics_row[3]),
            "question_score_mean": float(
                statistics_row[4]
            ),
            "answer_score_mean": float(
                statistics_row[5]
            ),
        }

        if statistics["objects"] < target_objects:
            raise RuntimeError(
                "The dataset contains only "
                f"{statistics['objects']:,} valid answered "
                f"questions, but {target_objects:,} were requested."
            )

    except Exception:
        if temporary_output.exists():
            temporary_output.unlink()

        raise

    finally:
        connection.close()

    temporary_output.replace(output)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_directory": str(source_path),
        "questions_file": str(questions_path),
        "answers_file": str(answers_path),
        "questions_file_gib": round(
            questions_path.stat().st_size / 1024**3,
            3,
        ),
        "answers_file_gib": round(
            answers_path.stat().st_size / 1024**3,
            3,
        ),
        "target_objects": target_objects,
        "random_seed": seed,
        "output_path": str(output),
        "output_mib": round(
            output.stat().st_size / 1024**2,
            2,
        ),
        "statistics": statistics,
    }

    manifest_path = output.with_suffix(
        ".manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return manifest
