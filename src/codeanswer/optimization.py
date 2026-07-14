"""PCA compression and FAISS HNSW index construction."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
from sklearn.decomposition import PCA

from codeanswer.indexing import (
    infer_embedding_objects,
    open_embeddings,
)


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Atomically write a JSON file."""

    temporary_path = path.with_name(
        f".{path.name}.temporary"
    )

    temporary_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _manifest_path(path: Path) -> Path:
    return Path(f"{path}.manifest.json")


def _validate_normalized(
    vectors: np.ndarray,
    *,
    tolerance: float = 1e-3,
) -> None:
    """Validate finite, L2-normalized vectors."""

    if not np.isfinite(vectors).all():
        raise ValueError(
            "Vectors contain NaN or infinite values."
        )

    norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=tolerance,
    ):
        raise ValueError(
            "Vectors must be L2-normalized."
        )


def normalize_rows(
    vectors: np.ndarray,
) -> np.ndarray:
    """Return row-wise L2-normalized float32 vectors."""

    output = np.asarray(
        vectors,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        output,
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= 1e-12):
        raise ValueError(
            "Cannot normalize zero-length vectors."
        )

    output = output / norms

    return np.asarray(
        output,
        dtype=np.float32,
    )


def fit_pca_embeddings(
    embedding_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
    *,
    input_dimension: int,
    output_dimension: int,
    sample_size: int,
    transform_batch_size: int,
    seed: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit PCA and transform a raw float32 embedding matrix."""

    if input_dimension <= 0:
        raise ValueError(
            "input_dimension must be positive."
        )

    if not 0 < output_dimension < input_dimension:
        raise ValueError(
            "output_dimension must be positive "
            "and smaller than input_dimension."
        )

    if sample_size < output_dimension:
        raise ValueError(
            "sample_size must be at least "
            "output_dimension."
        )

    if transform_batch_size <= 0:
        raise ValueError(
            "transform_batch_size must be positive."
        )

    source = Path(
        embedding_path
    ).expanduser().resolve()

    output = Path(
        output_path
    ).expanduser().resolve()

    pca_model_path = Path(
        model_path
    ).expanduser().resolve()

    manifest_path = _manifest_path(output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    pca_model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if overwrite:
        for path in [
            output,
            pca_model_path,
            manifest_path,
        ]:
            if path.exists():
                path.unlink()

    existing = [
        path
        for path in [
            output,
            pca_model_path,
            manifest_path,
        ]
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "PCA artifacts already exist. "
            "Use overwrite=True to replace them."
        )

    objects = infer_embedding_objects(
        source,
        input_dimension,
    )

    embeddings = open_embeddings(
        source,
        objects=objects,
        dimension=input_dimension,
    )

    actual_sample_size = min(
        sample_size,
        objects,
    )

    if actual_sample_size < output_dimension:
        raise ValueError(
            "The dataset contains fewer rows than "
            "the requested PCA output dimension."
        )

    rng = np.random.default_rng(seed)

    sample_indices = rng.choice(
        objects,
        size=actual_sample_size,
        replace=False,
    )

    training_sample = np.asarray(
        embeddings[sample_indices],
        dtype=np.float32,
    )

    _validate_normalized(training_sample)

    pca = PCA(
        n_components=output_dimension,
        svd_solver="randomized",
        random_state=seed,
    )

    fit_started_at = time.perf_counter()
    pca.fit(training_sample)
    fit_seconds = (
        time.perf_counter()
        - fit_started_at
    )

    temporary_output = output.with_name(
        f".{output.name}.temporary"
    )
    temporary_model = pca_model_path.with_name(
        f".{pca_model_path.name}.temporary"
    )

    for path in [
        temporary_output,
        temporary_model,
    ]:
        if path.exists():
            path.unlink()

    transformed = np.memmap(
        temporary_output,
        dtype=np.float32,
        mode="w+",
        shape=(
            objects,
            output_dimension,
        ),
    )

    transform_started_at = time.perf_counter()

    try:
        for start in range(
            0,
            objects,
            transform_batch_size,
        ):
            end = min(
                start + transform_batch_size,
                objects,
            )

            batch = np.asarray(
                embeddings[start:end],
                dtype=np.float32,
            )

            projected = pca.transform(
                batch
            )

            transformed[start:end] = (
                normalize_rows(projected)
            )

        transformed.flush()
        del transformed

        joblib.dump(
            pca,
            temporary_model,
        )

        temporary_output.replace(output)
        temporary_model.replace(
            pca_model_path
        )

    except Exception:
        try:
            del transformed
        except UnboundLocalError:
            pass

        for path in [
            temporary_output,
            temporary_model,
        ]:
            if path.exists():
                path.unlink()

        raise

    transform_seconds = (
        time.perf_counter()
        - transform_started_at
    )

    reduced_embeddings = np.memmap(
        output,
        dtype=np.float32,
        mode="r",
        shape=(
            objects,
            output_dimension,
        ),
    )

    validation_indices = np.linspace(
        0,
        objects - 1,
        min(objects, 10_000),
        dtype=np.int64,
    )

    validation_sample = np.asarray(
        reduced_embeddings[
            validation_indices
        ],
        dtype=np.float32,
    )

    _validate_normalized(
        validation_sample
    )

    sample_norms = np.linalg.norm(
        validation_sample,
        axis=1,
    )

    manifest = {
        "created_at": (
            datetime.now(UTC).isoformat()
        ),
        "source_embedding_path": str(source),
        "output_embedding_path": str(output),
        "pca_model_path": str(
            pca_model_path
        ),
        "objects": objects,
        "input_dimension": input_dimension,
        "output_dimension": output_dimension,
        "sample_size": actual_sample_size,
        "seed": seed,
        "explained_variance": round(
            float(
                pca.explained_variance_ratio_.sum()
            ),
            6,
        ),
        "fit_seconds": round(
            fit_seconds,
            3,
        ),
        "transform_seconds": round(
            transform_seconds,
            3,
        ),
        "source_mib": round(
            source.stat().st_size / 1024**2,
            2,
        ),
        "output_mib": round(
            output.stat().st_size / 1024**2,
            2,
        ),
        "memory_reduction_percent": round(
            (
                1
                - output.stat().st_size
                / source.stat().st_size
            )
            * 100,
            2,
        ),
        "sample_norm_mean": round(
            float(sample_norms.mean()),
            6,
        ),
        "sample_norm_std": round(
            float(sample_norms.std()),
            6,
        ),
    }

    del reduced_embeddings

    _write_json(
        manifest_path,
        manifest,
    )

    return manifest


def load_pca_model(
    model_path: str | Path,
) -> PCA:
    """Load a saved PCA model."""

    path = Path(
        model_path
    ).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"PCA model not found: {path}"
        )

    model = joblib.load(path)

    if not isinstance(model, PCA):
        raise TypeError(
            "Loaded object is not a scikit-learn PCA model."
        )

    return model


def transform_queries(
    pca: PCA,
    query_vectors: np.ndarray,
) -> np.ndarray:
    """Project and normalize query embeddings."""

    queries = np.asarray(
        query_vectors,
        dtype=np.float32,
    )

    if queries.ndim == 1:
        queries = queries.reshape(1, -1)

    if queries.ndim != 2:
        raise ValueError(
            "query_vectors must be a 1D or 2D array."
        )

    if queries.shape[1] != pca.n_features_in_:
        raise ValueError(
            "Query dimension does not match "
            "the PCA input dimension."
        )

    transformed = pca.transform(
        queries
    )

    return normalize_rows(
        transformed
    )


def build_hnsw_index(
    embedding_path: str | Path,
    output_path: str | Path,
    *,
    dimension: int,
    m: int,
    ef_construction: int,
    add_batch_size: int = 10_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a FAISS HNSW inner-product index."""

    if dimension <= 0:
        raise ValueError(
            "dimension must be positive."
        )

    if m <= 0:
        raise ValueError(
            "m must be positive."
        )

    if ef_construction <= 0:
        raise ValueError(
            "ef_construction must be positive."
        )

    if add_batch_size <= 0:
        raise ValueError(
            "add_batch_size must be positive."
        )

    source = Path(
        embedding_path
    ).expanduser().resolve()

    output = Path(
        output_path
    ).expanduser().resolve()

    manifest_path = _manifest_path(output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if overwrite:
        for path in [
            output,
            manifest_path,
        ]:
            if path.exists():
                path.unlink()

    if output.exists() or manifest_path.exists():
        raise FileExistsError(
            "HNSW index or manifest already exists. "
            "Use overwrite=True to replace it."
        )

    objects = infer_embedding_objects(
        source,
        dimension,
    )

    embeddings = open_embeddings(
        source,
        objects=objects,
        dimension=dimension,
    )

    sample_indices = np.linspace(
        0,
        objects - 1,
        min(objects, 10_000),
        dtype=np.int64,
    )

    _validate_normalized(
        np.asarray(
            embeddings[sample_indices],
            dtype=np.float32,
        )
    )

    index = faiss.IndexHNSWFlat(
        dimension,
        m,
        faiss.METRIC_INNER_PRODUCT,
    )

    index.hnsw.efConstruction = (
        ef_construction
    )

    build_started_at = time.perf_counter()

    for start in range(
        0,
        objects,
        add_batch_size,
    ):
        end = min(
            start + add_batch_size,
            objects,
        )

        batch = np.ascontiguousarray(
            embeddings[start:end],
            dtype=np.float32,
        )

        index.add(batch)

    build_seconds = (
        time.perf_counter()
        - build_started_at
    )

    if index.ntotal != objects:
        raise RuntimeError(
            f"HNSW contains {index.ntotal:,} vectors, "
            f"expected {objects:,}."
        )

    temporary_output = output.with_name(
        f".{output.name}.temporary"
    )

    if temporary_output.exists():
        temporary_output.unlink()

    try:
        faiss.write_index(
            index,
            str(temporary_output),
        )
        temporary_output.replace(output)

    except Exception:
        if temporary_output.exists():
            temporary_output.unlink()

        raise

    manifest = {
        "created_at": (
            datetime.now(UTC).isoformat()
        ),
        "embedding_path": str(source),
        "index_path": str(output),
        "index_type": "IndexHNSWFlat",
        "metric": "inner_product",
        "objects": objects,
        "dimension": dimension,
        "m": m,
        "ef_construction": ef_construction,
        "add_batch_size": add_batch_size,
        "build_seconds": round(
            build_seconds,
            3,
        ),
        "embedding_mib": round(
            source.stat().st_size / 1024**2,
            2,
        ),
        "index_mib": round(
            output.stat().st_size / 1024**2,
            2,
        ),
    }

    _write_json(
        manifest_path,
        manifest,
    )

    return manifest
