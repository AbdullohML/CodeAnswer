"""Generate normalized dense embeddings for the question corpus."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import pyarrow.parquet as pq


class TextEncoder(Protocol):
    """Interface required by the embedding pipeline."""

    model_name: str
    device: str
    dimension: int

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> np.ndarray:
        """Encode text into normalized float32 vectors."""


@dataclass
class SentenceTransformerEncoder:
    """Sentence Transformers implementation of TextEncoder."""

    model_name: str
    max_sequence_length: int = 256
    device: str | None = None

    dimension: int = field(init=False)
    _model: Any = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        import torch
        from sentence_transformers import (
            SentenceTransformer,
        )

        selected_device = self.device

        if selected_device is None:
            selected_device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = selected_device

        self._model = SentenceTransformer(
            self.model_name,
            device=selected_device,
        )
        self._model.max_seq_length = (
            self.max_sequence_length
        )

        dimension = (
            self._model
            .get_sentence_embedding_dimension()
        )

        if dimension is None:
            raise RuntimeError(
                "Could not determine embedding dimension."
            )

        self.dimension = int(dimension)

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> np.ndarray:
        """Encode and L2-normalize a batch of text."""

        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            vectors,
            dtype=np.float32,
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


def _state_path(output_path: Path) -> Path:
    return Path(f"{output_path}.state.json")


def _manifest_path(output_path: Path) -> Path:
    return Path(f"{output_path}.manifest.json")


def _validate_state(
    state: dict[str, Any],
    *,
    corpus_path: Path,
    output_path: Path,
    model_name: str,
    objects: int,
    dimension: int,
) -> None:
    """Ensure a partial embedding run is compatible."""

    expected = {
        "corpus_path": str(corpus_path),
        "output_path": str(output_path),
        "model_name": model_name,
        "objects": objects,
        "dimension": dimension,
    }

    for key, expected_value in expected.items():
        if state.get(key) != expected_value:
            raise ValueError(
                "The existing embedding state is "
                f"incompatible for {key!r}: "
                f"expected {expected_value!r}, "
                f"found {state.get(key)!r}."
            )


def _validate_vectors(
    vectors: np.ndarray,
    *,
    expected_rows: int,
    expected_dimension: int,
) -> None:
    """Validate one encoded batch."""

    expected_shape = (
        expected_rows,
        expected_dimension,
    )

    if vectors.shape != expected_shape:
        raise ValueError(
            "Encoder returned shape "
            f"{vectors.shape}, expected {expected_shape}."
        )

    if vectors.dtype != np.float32:
        raise ValueError(
            "Encoder must return float32 vectors."
        )

    if not np.isfinite(vectors).all():
        raise ValueError(
            "Encoder returned NaN or infinite values."
        )

    norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-3,
    ):
        raise ValueError(
            "Embeddings must be L2-normalized."
        )


def generate_embeddings(
    corpus_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str,
    expected_dimension: int,
    batch_size: int,
    max_sequence_length: int = 256,
    device: str | None = None,
    overwrite: bool = False,
    encoder: TextEncoder | None = None,
) -> dict[str, Any]:
    """Generate resumable question-body embeddings.

    Rows in the output file follow the corpus ``doc_id`` order.
    The raw output format is contiguous float32 data with shape
    ``(objects, dimension)``.
    """

    if expected_dimension <= 0:
        raise ValueError(
            "expected_dimension must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    corpus = Path(
        corpus_path
    ).expanduser().resolve()
    output = Path(
        output_path
    ).expanduser().resolve()

    if not corpus.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus}"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_path = _state_path(output)
    manifest_path = _manifest_path(output)

    if overwrite:
        for path in [
            output,
            state_path,
            manifest_path,
        ]:
            if path.exists():
                path.unlink()

    if manifest_path.exists():
        raise FileExistsError(
            "Embeddings are already complete: "
            f"{output}. Use overwrite=True to replace them."
        )

    parquet_file = pq.ParquetFile(corpus)
    schema_names = set(
        parquet_file.schema_arrow.names
    )

    required_columns = {
        "doc_id",
        "question_body",
    }
    missing_columns = (
        required_columns - schema_names
    )

    if missing_columns:
        raise ValueError(
            "Corpus is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    objects = int(
        parquet_file.metadata.num_rows
    )

    if objects <= 0:
        raise ValueError(
            "The corpus does not contain any rows."
        )

    active_encoder = encoder

    if active_encoder is None:
        active_encoder = (
            SentenceTransformerEncoder(
                model_name=model_name,
                max_sequence_length=(
                    max_sequence_length
                ),
                device=device,
            )
        )

    if active_encoder.dimension != expected_dimension:
        raise ValueError(
            "Configured embedding dimension "
            f"is {expected_dimension}, but "
            f"{active_encoder.model_name} produces "
            f"{active_encoder.dimension}."
        )

    expected_bytes = (
        objects
        * expected_dimension
        * np.dtype(np.float32).itemsize
    )

    processed = 0
    previous_seconds = 0.0

    if state_path.exists():
        state = json.loads(
            state_path.read_text(
                encoding="utf-8"
            )
        )

        _validate_state(
            state,
            corpus_path=corpus,
            output_path=output,
            model_name=active_encoder.model_name,
            objects=objects,
            dimension=expected_dimension,
        )

        processed = int(
            state.get("processed", 0)
        )
        previous_seconds = float(
            state.get("seconds_total", 0.0)
        )

        if not output.exists():
            raise FileNotFoundError(
                "Embedding state exists, but the "
                f"embedding file is missing: {output}"
            )

        if output.stat().st_size != expected_bytes:
            raise ValueError(
                "Partial embedding file has an "
                "unexpected size."
            )

        embeddings = np.memmap(
            output,
            dtype=np.float32,
            mode="r+",
            shape=(
                objects,
                expected_dimension,
            ),
        )

    else:
        if output.exists():
            raise FileExistsError(
                "Embedding file exists without a "
                "state or manifest file. Use "
                "overwrite=True to replace it."
            )

        embeddings = np.memmap(
            output,
            dtype=np.float32,
            mode="w+",
            shape=(
                objects,
                expected_dimension,
            ),
        )

        _write_json(
            state_path,
            {
                "status": "running",
                "corpus_path": str(corpus),
                "output_path": str(output),
                "model_name": (
                    active_encoder.model_name
                ),
                "device": active_encoder.device,
                "objects": objects,
                "dimension": expected_dimension,
                "processed": 0,
                "seconds_total": 0.0,
            },
        )

    if not 0 <= processed <= objects:
        raise ValueError(
            f"Invalid processed row count: {processed}"
        )

    run_started_at = time.perf_counter()
    rows_seen = 0

    try:
        for record_batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=[
                "doc_id",
                "question_body",
            ],
        ):
            batch_rows = record_batch.num_rows
            batch_end = rows_seen + batch_rows

            if batch_end <= processed:
                rows_seen = batch_end
                continue

            skip_rows = max(
                0,
                processed - rows_seen,
            )

            doc_ids = np.asarray(
                record_batch
                .column(0)
                .to_numpy(
                    zero_copy_only=False
                ),
                dtype=np.int64,
            )[skip_rows:]

            texts = [
                str(value or "")
                for value in (
                    record_batch
                    .column(1)
                    .to_pylist()[skip_rows:]
                )
            ]

            expected_doc_ids = np.arange(
                processed,
                processed + len(doc_ids),
                dtype=np.int64,
            )

            if not np.array_equal(
                doc_ids,
                expected_doc_ids,
            ):
                raise ValueError(
                    "Corpus doc_id values must be "
                    "contiguous and ordered from zero."
                )

            vectors = active_encoder.encode(
                texts,
                batch_size=batch_size,
            )

            _validate_vectors(
                vectors,
                expected_rows=len(texts),
                expected_dimension=(
                    expected_dimension
                ),
            )

            new_processed = (
                processed + len(texts)
            )

            embeddings[
                processed:new_processed
            ] = vectors
            embeddings.flush()

            processed = new_processed
            rows_seen = batch_end

            elapsed_total = (
                previous_seconds
                + time.perf_counter()
                - run_started_at
            )

            _write_json(
                state_path,
                {
                    "status": "running",
                    "corpus_path": str(corpus),
                    "output_path": str(output),
                    "model_name": (
                        active_encoder.model_name
                    ),
                    "device": (
                        active_encoder.device
                    ),
                    "objects": objects,
                    "dimension": (
                        expected_dimension
                    ),
                    "processed": processed,
                    "seconds_total": round(
                        elapsed_total,
                        3,
                    ),
                },
            )

    except Exception as error:
        elapsed_total = (
            previous_seconds
            + time.perf_counter()
            - run_started_at
        )

        _write_json(
            state_path,
            {
                "status": "failed",
                "corpus_path": str(corpus),
                "output_path": str(output),
                "model_name": (
                    active_encoder.model_name
                ),
                "device": active_encoder.device,
                "objects": objects,
                "dimension": expected_dimension,
                "processed": processed,
                "seconds_total": round(
                    elapsed_total,
                    3,
                ),
                "last_error": str(error),
            },
        )

        raise

    embeddings.flush()
    del embeddings

    if processed != objects:
        raise RuntimeError(
            f"Processed {processed:,} of "
            f"{objects:,} corpus rows."
        )

    completed_embeddings = np.memmap(
        output,
        dtype=np.float32,
        mode="r",
        shape=(
            objects,
            expected_dimension,
        ),
    )

    sample_size = min(
        objects,
        10_000,
    )
    sample_indices = np.linspace(
        0,
        objects - 1,
        sample_size,
        dtype=np.int64,
    )
    sample = np.asarray(
        completed_embeddings[
            sample_indices
        ]
    )
    sample_norms = np.linalg.norm(
        sample,
        axis=1,
    )

    seconds_this_run = (
        time.perf_counter()
        - run_started_at
    )
    seconds_total = (
        previous_seconds
        + seconds_this_run
    )

    manifest = {
        "created_at": (
            datetime.now(UTC).isoformat()
        ),
        "corpus_path": str(corpus),
        "output_path": str(output),
        "model_name": (
            active_encoder.model_name
        ),
        "device": active_encoder.device,
        "objects": objects,
        "dimension": expected_dimension,
        "dtype": "float32",
        "normalized": True,
        "batch_size": batch_size,
        "max_sequence_length": (
            max_sequence_length
        ),
        "file_mib": round(
            output.stat().st_size
            / 1024**2,
            2,
        ),
        "seconds_this_run": round(
            seconds_this_run,
            3,
        ),
        "seconds_total": round(
            seconds_total,
            3,
        ),
        "sample_size": sample_size,
        "sample_norm_mean": round(
            float(sample_norms.mean()),
            6,
        ),
        "sample_norm_std": round(
            float(sample_norms.std()),
            6,
        ),
    }

    del completed_embeddings

    _write_json(
        manifest_path,
        manifest,
    )

    if state_path.exists():
        state_path.unlink()

    return manifest


def load_embeddings(
    embedding_path: str | Path,
    *,
    objects: int,
    dimension: int,
) -> np.memmap:
    """Open an existing raw float32 embedding matrix."""

    path = Path(
        embedding_path
    ).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {path}"
        )

    expected_bytes = (
        objects
        * dimension
        * np.dtype(np.float32).itemsize
    )

    if path.stat().st_size != expected_bytes:
        raise ValueError(
            "Embedding file size does not match "
            "the requested shape."
        )

    return np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        shape=(objects, dimension),
    )
