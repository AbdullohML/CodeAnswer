"""Tests for FAISS index construction and search."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codeanswer.indexing import (
    build_flat_index,
    infer_embedding_objects,
    load_index,
    open_embeddings,
    search_index,
)


def _normalized_vectors() -> np.ndarray:
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    vectors /= np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    return vectors


def _write_embeddings(
    path: Path,
    vectors: np.ndarray,
) -> None:
    output = np.memmap(
        path,
        dtype=np.float32,
        mode="w+",
        shape=vectors.shape,
    )

    output[:] = vectors
    output.flush()
    del output


def test_build_and_search_flat_index(
    tmp_path: Path,
) -> None:
    embedding_path = (
        tmp_path / "embeddings.f32"
    )
    index_path = (
        tmp_path / "flat.faiss"
    )

    vectors = _normalized_vectors()
    _write_embeddings(
        embedding_path,
        vectors,
    )

    manifest = build_flat_index(
        embedding_path=embedding_path,
        output_path=index_path,
        dimension=3,
        add_batch_size=2,
    )

    index = load_index(
        index_path,
        expected_objects=4,
        expected_dimension=3,
    )

    query = np.asarray(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    scores, document_ids = search_index(
        index,
        query,
        top_k=3,
    )

    assert manifest["objects"] == 4
    assert manifest["dimension"] == 3
    assert manifest["index_type"] == "IndexFlatIP"

    assert document_ids.shape == (1, 3)
    assert scores.shape == (1, 3)
    assert document_ids[0, 0] == 0
    assert scores[0, 0] == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_embedding_shape_helpers(
    tmp_path: Path,
) -> None:
    embedding_path = (
        tmp_path / "embeddings.f32"
    )

    vectors = _normalized_vectors()
    _write_embeddings(
        embedding_path,
        vectors,
    )

    objects = infer_embedding_objects(
        embedding_path,
        dimension=3,
    )

    embeddings = open_embeddings(
        embedding_path,
        objects=objects,
        dimension=3,
    )

    assert objects == 4
    assert embeddings.shape == (4, 3)


def test_rejects_unnormalized_embeddings(
    tmp_path: Path,
) -> None:
    embedding_path = (
        tmp_path / "embeddings.f32"
    )
    index_path = (
        tmp_path / "flat.faiss"
    )

    vectors = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=np.float32,
    )

    _write_embeddings(
        embedding_path,
        vectors,
    )

    with pytest.raises(
        ValueError,
        match="L2-normalized",
    ):
        build_flat_index(
            embedding_path=embedding_path,
            output_path=index_path,
            dimension=2,
        )


def test_search_rejects_wrong_dimension(
    tmp_path: Path,
) -> None:
    embedding_path = (
        tmp_path / "embeddings.f32"
    )
    index_path = (
        tmp_path / "flat.faiss"
    )

    vectors = _normalized_vectors()
    _write_embeddings(
        embedding_path,
        vectors,
    )

    build_flat_index(
        embedding_path=embedding_path,
        output_path=index_path,
        dimension=3,
    )

    index = load_index(index_path)

    with pytest.raises(
        ValueError,
        match="dimension",
    ):
        search_index(
            index,
            np.asarray(
                [1.0, 0.0],
                dtype=np.float32,
            ),
            top_k=2,
        )
