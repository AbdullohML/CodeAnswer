"""Tests for Stack Overflow corpus construction."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from codeanswer.data import build_corpus


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_corpus_selects_best_answer(
    tmp_path: Path,
) -> None:
    questions = [
        {
            "Id": 1,
            "Score": 5,
            "Title": "First <b>question</b>",
            "Body": "<p>Question one</p>",
        },
        {
            "Id": 2,
            "Score": 2,
            "Title": "Second question",
            "Body": "<p>Question two</p>",
        },
        {
            "Id": 3,
            "Score": 1,
            "Title": "Third question",
            "Body": "<p>Question three</p>",
        },
        {
            "Id": 4,
            "Score": 0,
            "Title": "Fourth question",
            "Body": "<p>Question four</p>",
        },
    ]

    answers = [
        {
            "Id": 11,
            "ParentId": 1,
            "Score": 1,
            "Body": "<p>Low answer</p>",
        },
        {
            "Id": 12,
            "ParentId": 1,
            "Score": 9,
            "Body": "<p>Best <b>answer</b></p>",
        },
        {
            "Id": 21,
            "ParentId": 2,
            "Score": 3,
            "Body": "<p>Answer two</p>",
        },
        {
            "Id": 31,
            "ParentId": 3,
            "Score": 2,
            "Body": "<p>Answer three</p>",
        },
        {
            "Id": 41,
            "ParentId": 4,
            "Score": 1,
            "Body": "<p>Answer four</p>",
        },
    ]

    _write_csv(
        tmp_path / "Questions.csv",
        ["Id", "Score", "Title", "Body"],
        questions,
    )
    _write_csv(
        tmp_path / "Answers.csv",
        ["Id", "ParentId", "Score", "Body"],
        answers,
    )

    output_path = tmp_path / "qa_corpus.parquet"

    manifest = build_corpus(
        source_dir=tmp_path,
        output_path=output_path,
        target_objects=4,
        seed=7,
    )

    corpus = pd.read_parquet(output_path)

    assert manifest["statistics"]["objects"] == 4
    assert len(corpus) == 4
    assert corpus["doc_id"].tolist() == [0, 1, 2, 3]
    assert corpus["question_id"].is_unique

    first_question = corpus[
        corpus["question_id"] == 1
    ].iloc[0]

    assert first_question["answer_id"] == 12
    assert first_question["answer"] == "Best answer"
    assert "<" not in first_question["title"]
    assert first_question["url"].endswith("/1")

    with pytest.raises(FileExistsError):
        build_corpus(
            source_dir=tmp_path,
            output_path=output_path,
            target_objects=4,
            seed=7,
        )
