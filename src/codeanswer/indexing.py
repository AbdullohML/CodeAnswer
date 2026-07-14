"""Build, load, and query FAISS vector indexes."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np


FLOAT32_BYTES = np.dtype(np.float32).itemsize


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


def _manifest_path(index_path: Path) -> Path:
    """Return the manifest path for an index."""

    return Path(f"{index_path}.manifest.json")


def infer_embedding_objects(
    embedding_path: str | Path,
    dimension: int,
) -> int:
    """Infer the number of vectors from a raw float32 file."""

    if dimension <= 0:
        raise ValueError(
            "dimension must be greater than zero."
        )

    path = Path(
        embedding_path
    ).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {path}"
        )

    bytes_per_vector = dimension * FLOAT32_BYTES
    file_size = path.stat().st_size

    if file_size == 0:
        raise ValueError(
            "Embedding file is empty."
        )

    if file_size % bytes_per_vector != 0:
        raise ValueError(
            "Embedding file size is incompatible "
            f"with dimension {dimension}."
        )

    return file_size // bytes_per_vector


def open_embeddings(
    embedding_path: str | Path,
    *,
    objects: int,
    dimension: int,
) -> np.memmap:
    """Open a raw float32 embedding matrix."""

    path = Path(
        embedding_path
    ).expanduser().resolve()

    expected_bytes = (
        objects
        * dimension
        * FLOAT32_BYTES
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {path}"
        )

    if path.stat().st_size != expected_bytes:
        raise ValueError(
            "Embedding file size does not match "
            "the requested matrix shape."
        )

    return np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        shape=(objects, dimension),
    )


def _validate_normalized_sample(
    embeddings: np.ndarray,
    *,
    sample_size: int = 10_000,
) -> dict[str, float]:
    """Check that a sample of vectors is L2-normalized."""

    objects = embeddings.shape[0]

    sample_indices = np.linspace(
        0,
        objects - 1,
        min(objects, sample_size),
        dtype=np.int64,
    )

    sample = np.asarray(
        embeddings[sample_indices],
        dtype=np.float32,
    )

    if not np.isfinite(sample).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    norms = np.linalg.norm(
        sample,
        axis=1,
    )

    norm_mean = float(norms.mean())
    norm_std = float(norms.std())

    if not np.allclose(
        norms,
        1.0,
        atol=1e-3,
    ):
        raise ValueError(
            "Embeddings must be L2-normalized "
            "before IndexFlatIP is used for "
            "cosine-similarity retrieval."
        )

    return {
        "sample_norm_mean": round(
            norm_mean,
            6,
        ),
        "sample_norm_std": round(
            norm_std,
            6,
        ),
    }


def build_flat_index(
    embedding_path: str | Path,
    output_path: str | Path,
    *,
    dimension: int,
    add_batch_size: int = 10_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build an exact FAISS inner-product index."""

    if add_batch_size <= 0:
        raise ValueError(
            "add_batch_size must be greater than zero."
        )

    embeddings_path = Path(
        embedding_path
    ).expanduser().resolve()

    index_path = Path(
        output_path
    ).expanduser().resolve()

    manifest_path = _manifest_path(
        index_path
    )

    if overwrite:
        for path in [
            index_path,
            manifest_path,
        ]:
            if path.exists():
                path.unlink()

    if index_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Index or manifest already exists. "
            "Use overwrite=True to replace it."
        )

    objects = infer_embedding_objects(
        embeddings_path,
        dimension,
    )

    embeddings = open_embeddings(
        embeddings_path,
        objects=objects,
        dimension=dimension,
    )

    normalization_stats = (
        _validate_normalized_sample(
            embeddings
        )
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    started_at = time.perf_counter()

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
        - started_at
    )

    if index.ntotal != objects:
        raise RuntimeError(
            "FAISS index contains "
            f"{index.ntotal:,} vectors, expected "
            f"{objects:,}."
        )

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_index_path = (
        index_path.with_name(
            f".{index_path.name}.temporary"
        )
    )

    if temporary_index_path.exists():
        temporary_index_path.unlink()

    try:
        faiss.write_index(
            index,
            str(temporary_index_path),
        )
        temporary_index_path.replace(
            index_path
        )

    except Exception:
        if temporary_index_path.exists():
            temporary_index_path.unlink()

        raise

    manifest = {
        "created_at": (
            datetime.now(UTC).isoformat()
        ),
        "embedding_path": str(
            embeddings_path
        ),
        "index_path": str(index_path),
        "index_type": "IndexFlatIP",
        "metric": "inner_product",
        "normalized_embeddings": True,
        "objects": objects,
        "dimension": dimension,
        "add_batch_size": add_batch_size,
        "build_seconds": round(
            build_seconds,
            3,
        ),
        "index_mib": round(
            index_path.stat().st_size
            / 1024**2,
            2,
        ),
        **normalization_stats,
    }

    _write_json(
        manifest_path,
        manifest,
    )

    return manifest


def load_index(
    index_path: str | Path,
    *,
    expected_objects: int | None = None,
    expected_dimension: int | None = None,
) -> Any:
    """Load and validate a FAISS index."""

    path = Path(
        index_path
    ).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {path}"
        )

    index = faiss.read_index(
        str(path)
    )

    if (
        expected_objects is not None
        and index.ntotal != expected_objects
    ):
        raise ValueError(
            "Index object count does not match: "
            f"expected {expected_objects:,}, "
            f"found {index.ntotal:,}."
        )

    if (
        expected_dimension is not None
        and index.d != expected_dimension
    ):
        raise ValueError(
            "Index dimension does not match: "
            f"expected {expected_dimension}, "
            f"found {index.d}."
        )

    return index


def search_index(
    index: Any,
    query_vectors: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Search a FAISS index with normalized float32 queries."""

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

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

    if queries.shape[1] != index.d:
        raise ValueError(
            "Query dimension does not match "
            f"index dimension {index.d}."
        )

    if not np.isfinite(queries).all():
        raise ValueError(
            "Query vectors contain NaN or "
            "infinite values."
        )

    norms = np.linalg.norm(
        queries,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-3,
    ):
        raise ValueError(
            "Query vectors must be L2-normalized."
        )

    queries = np.ascontiguousarray(
        queries,
        dtype=np.float32,
    )

    scores, document_ids = index.search(
        queries,
        min(top_k, index.ntotal),
    )

    return scores, document_ids
