"""Project configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used by the project."""

    repository_root: Path
    data_dir: Path
    artifacts_dir: Path
    results_dir: Path

    def create_directories(self) -> None:
        """Create writable project directories."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetConfig:
    """Stack Overflow corpus settings."""

    name: str
    target_objects: int
    random_seed: int


@dataclass(frozen=True)
class EmbeddingConfig:
    """Bi-encoder configuration."""

    model_name: str
    dimension: int
    max_sequence_length: int
    batch_size: int


@dataclass(frozen=True)
class PCAConfig:
    """Dimensionality-reduction configuration."""

    output_dimension: int
    sample_size: int
    transform_batch_size: int


@dataclass(frozen=True)
class HNSWConfig:
    """FAISS HNSW index configuration."""

    m: int
    ef_construction: int
    ef_search: int


@dataclass(frozen=True)
class RerankerConfig:
    """Cross-encoder reranking configuration."""

    model_name: str
    retrieval_candidates: int
    final_sources: int
    batch_size: int
    answerability_threshold: float


@dataclass(frozen=True)
class OllamaConfig:
    """Local generator configuration."""

    base_url: str
    model_name: str
    temperature: float
    context_length: int
    maximum_answer_words: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Experiment and benchmark configuration."""

    retrieval_queries: int
    latency_queries: int
    calibration_in_domain: int
    calibration_out_of_domain: int
    rag_in_domain: int
    rag_out_of_domain: int


@dataclass(frozen=True)
class ProjectConfig:
    """Complete CodeAnswer configuration."""

    paths: PathsConfig
    dataset: DatasetConfig
    embedding: EmbeddingConfig
    pca: PCAConfig
    hnsw: HNSWConfig
    reranker: RerankerConfig
    ollama: OllamaConfig
    evaluation: EvaluationConfig


def _resolve_path(repository_root: Path, value: str) -> Path:
    path = Path(value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (repository_root / path).resolve()


def _require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)

    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid configuration section: {name}")

    return section


def load_config(
    config_path: str | Path = "configs/default.yaml",
) -> ProjectConfig:
    """Load project settings from a YAML file."""

    path = Path(config_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("The configuration file must contain a mapping.")

    repository_root = path.parent.parent

    paths_raw = _require_section(raw, "paths")
    dataset_raw = _require_section(raw, "dataset")
    embedding_raw = _require_section(raw, "embedding")
    pca_raw = _require_section(raw, "pca")
    hnsw_raw = _require_section(raw, "hnsw")
    reranker_raw = _require_section(raw, "reranker")
    ollama_raw = _require_section(raw, "ollama")
    evaluation_raw = _require_section(raw, "evaluation")

    paths = PathsConfig(
        repository_root=repository_root,
        data_dir=_resolve_path(repository_root, paths_raw["data_dir"]),
        artifacts_dir=_resolve_path(
            repository_root,
            paths_raw["artifacts_dir"],
        ),
        results_dir=_resolve_path(
            repository_root,
            paths_raw["results_dir"],
        ),
    )

    config = ProjectConfig(
        paths=paths,
        dataset=DatasetConfig(**dataset_raw),
        embedding=EmbeddingConfig(**embedding_raw),
        pca=PCAConfig(**pca_raw),
        hnsw=HNSWConfig(**hnsw_raw),
        reranker=RerankerConfig(**reranker_raw),
        ollama=OllamaConfig(**ollama_raw),
        evaluation=EvaluationConfig(**evaluation_raw),
    )

    config.paths.create_directories()
    return config
