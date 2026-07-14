"""Tests for PCA compression and HNSW indexing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codeanswer.indexing import (
    load_index,
    open_embeddings,
    search_index,
)
from codeanswer.optimization import (
    build_hnsw_index,
    fit_pca_embeddings,
    load_pca_model,
    transform_queries,
)


def _normalized_random_vectors(
    *,
    objects: int,
    dimension: int,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    vectors = rng.normal(
        size=(objects, dimension)
    ).astype(np.float32)

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


def test_pca_compression_and_hnsw_search(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.f32"
    reduced_path = tmp_path / "reduced.f32"
    model_path = tmp_path / "pca.joblib"
    index_path = tmp_path / "hnsw.faiss"

    vectors = _normalized_random_vectors(
        objects=30,
        dimension=6,
    )

    _write_embeddings(
        source_path,
        vectors,
    )

    pca_manifest = fit_pca_embeddings(
        embedding_path=source_path,
        output_path=reduced_path,
        model_path=model_path,
        input_dimension=6,
        output_dimension=3,
        sample_size=25,
        transform_batch_size=7,
        seed=11,
    )

    reduced = open_embeddings(
        reduced_path,
        objects=30,
        dimension=3,
    )

    assert reduced.shape == (30, 3)
    assert pca_manifest["input_dimension"] == 6
    assert pca_manifest["output_dimension"] == 3
    assert pca_manifest[
        "memory_reduction_percent"
    ] == 50.0

    norms = np.linalg.norm(
        reduced,
        axis=1,
    )

    assert np.allclose(
        norms,
        1.0,
        atol=1e-5,
    )

    hnsw_manifest = build_hnsw_index(
        embedding_path=reduced_path,
        output_path=index_path,
        dimension=3,
        m=8,
        ef_construction=40,
        add_batch_size=10,
    )

    index = load_index(
        index_path,
        expected_objects=30,
        expected_dimension=3,
    )
    index.hnsw.efSearch = 64

    scores, document_ids = search_index(
        index,
        np.asarray(reduced[0]),
        top_k=3,
    )

    assert hnsw_manifest["objects"] == 30
    assert hnsw_manifest["m"] == 8
    assert document_ids[0, 0] == 0
    assert scores[0, 0] == pytest.approx(
        1.0,
        abs=1e-5,
    )


def test_transform_queries(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.f32"
    reduced_path = tmp_path / "reduced.f32"
    model_path = tmp_path / "pca.joblib"

    vectors = _normalized_random_vectors(
        objects=20,
        dimension=5,
    )

    _write_embeddings(
        source_path,
        vectors,
    )

    fit_pca_embeddings(
        embedding_path=source_path,
        output_path=reduced_path,
        model_path=model_path,
        input_dimension=5,
        output_dimension=2,
        sample_size=20,
        transform_batch_size=5,
        seed=3,
    )

    pca = load_pca_model(
        model_path
    )

    transformed = transform_queries(
        pca,
        vectors[:4],
    )

    assert transformed.shape == (4, 2)

    assert np.allclose(
        np.linalg.norm(
            transformed,
            axis=1,
        ),
        1.0,
        atol=1e-5,
    )


def test_rejects_invalid_pca_configuration(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.f32"

    vectors = _normalized_random_vectors(
        objects=10,
        dimension=4,
    )

    _write_embeddings(
        source_path,
        vectors,
    )

    with pytest.raises(
        ValueError,
        match="smaller than input_dimension",
    ):
        fit_pca_embeddings(
            embedding_path=source_path,
            output_path=tmp_path / "output.f32",
            model_path=tmp_path / "pca.joblib",
            input_dimension=4,
            output_dimension=4,
            sample_size=10,
            transform_batch_size=5,
            seed=1,
        )
