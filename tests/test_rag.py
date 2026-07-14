"""Tests for grounded answer generation."""

from __future__ import annotations

import pandas as pd

from codeanswer.rag import (
    citation_metrics,
    generate_grounded_answer,
)


class FakeGenerator:
    model_name = "fake"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> dict[str, object]:
        del system_prompt
        del user_prompt
        del max_tokens

        return {
            "text": self.text,
            "elapsed_ms": 10.0,
            "prompt_tokens": 20,
            "output_tokens": 10,
        }


def _sources() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": ["Python answer"],
            "question_body": [
                "How should this be done?"
            ],
            "answer": [
                "Use the supported implementation."
            ],
            "answer_score": [5],
            "url": [
                "https://stackoverflow.com/questions/1"
            ],
        }
    )


def test_citation_metrics() -> None:
    metrics = citation_metrics(
        "Use the implementation. [1]",
        source_count=1,
    )

    assert metrics["citation_coverage"] == 1.0
    assert metrics["citation_validity"] == 1.0


def test_valid_generation_is_preserved() -> None:
    result = generate_grounded_answer(
        "How?",
        _sources(),
        generator=FakeGenerator(
            "Use the implementation. [1]"
        ),
        maximum_words=50,
    )

    assert result["used_fallback"] is False
    assert result["citation_validity"] == 1.0


def test_invalid_generation_uses_fallback() -> None:
    result = generate_grounded_answer(
        "How?",
        _sources(),
        generator=FakeGenerator(
            "This answer has no citation."
        ),
        maximum_words=50,
    )

    assert result["used_fallback"] is True
    assert result["answer"].endswith("[1]")
    assert result["citation_validity"] == 1.0
