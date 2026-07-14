"""Tests for the embedding pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from codeanswer.embeddings import (
    generate_embeddings,
    load_embeddings,
)


class FakeEncoder:
    """Small deterministic encoder used in tests."""

    model_name = "fake-encoder"
    device = "cpu"
    dimension = 4

    def __init__(
        self,
        fail_on_call: int | None = None,
    ) -> None:
        self.fail_on_call = fail_on_call
        self.calls = 0

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> np.ndarray:
        del batch_size

        self.calls += 1

        if self.calls == self.fail_on_call:
            raise RuntimeError(
                "Simulated encoder failure."
            )

        vectors = []

        for text in texts:
            vector = np.asarray(
                [
                    len(text) + 1,
                    sum(map(ord, text)) % 17 + 1,
                    text.count("a") + 1,
                    text.count("e") + 1,
                ],
                dtype=np.float32,
            )

            vector /= np.linalg.norm(vector)
            vectors.append(vector)

        return np.stack(vectors).astype(
            np.float32
        )


def _write_corpus(
    path: Path,
    objects: int = 5,
) -> None:
    frame = pd.DataFrame(
        {
            "doc_id": list(range(objects)),
            "question_body": [
                f"Question body {index}"
                for index in range(objects)
            ],
        }
    )

    frame.to_parquet(
        path,
        index=False,
    )


def test_generate_and_load_embeddings(
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "corpus.parquet"
    output_path = tmp_path / "embeddings.f32"

    _write_corpus(corpus_path)

    manifest = generate_embeddings(
        corpus_path=corpus_path,
        output_path=output_path,
        model_name="fake-encoder",
        expected_dimension=4,
        batch_size=2,
        encoder=FakeEncoder(),
    )

    embeddings = load_embeddings(
        output_path,
        objects=5,
        dimension=4,
    )

    assert manifest["objects"] == 5
    assert manifest["dimension"] == 4
    assert manifest["normalized"] is True
    assert embeddings.shape == (5, 4)

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    assert np.allclose(
        norms,
        1.0,
        atol=1e-6,
    )

    with pytest.raises(FileExistsError):
        generate_embeddings(
            corpus_path=corpus_path,
            output_path=output_path,
            model_name="fake-encoder",
            expected_dimension=4,
            batch_size=2,
            encoder=FakeEncoder(),
        )


def test_embedding_generation_resumes(
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "corpus.parquet"
    output_path = tmp_path / "embeddings.f32"

    _write_corpus(corpus_path)

    with pytest.raises(
        RuntimeError,
        match="Simulated encoder failure",
    ):
        generate_embeddings(
            corpus_path=corpus_path,
            output_path=output_path,
            model_name="fake-encoder",
            expected_dimension=4,
            batch_size=2,
            encoder=FakeEncoder(
                fail_on_call=2
            ),
        )

    state_path = Path(
        f"{output_path}.state.json"
    )

    assert output_path.exists()
    assert state_path.exists()

    manifest = generate_embeddings(
        corpus_path=corpus_path,
        output_path=output_path,
        model_name="fake-encoder",
        expected_dimension=4,
        batch_size=2,
        encoder=FakeEncoder(),
    )

    embeddings = load_embeddings(
        output_path,
        objects=5,
        dimension=4,
    )

    assert manifest["objects"] == 5
    assert embeddings.shape == (5, 4)
    assert not state_path.exists()
