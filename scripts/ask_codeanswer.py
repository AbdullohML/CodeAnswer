"""Ask CodeAnswer a programming question."""

from __future__ import annotations

import argparse
import json

from codeanswer.config import load_config
from codeanswer.embeddings import (
    SentenceTransformerEncoder,
)
from codeanswer.indexing import load_index
from codeanswer.optimization import (
    load_pca_model,
)
from codeanswer.rag import (
    CodeAnswerPipeline,
    OllamaGenerator,
)
from codeanswer.reranking import (
    CrossEncoderScorer,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run citation-grounded programming QA."
        )
    )

    parser.add_argument(
        "question",
        help="Programming question to answer.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)

    optimized_dir = (
        config.paths.artifacts_dir
        / "optimized"
    )

    encoder = SentenceTransformerEncoder(
        model_name=config.embedding.model_name,
        max_sequence_length=(
            config.embedding.max_sequence_length
        ),
        device=arguments.device,
    )

    reranker = CrossEncoderScorer(
        model_name=config.reranker.model_name,
        device=arguments.device,
    )

    generator = OllamaGenerator(
        model_name=config.ollama.model_name,
        base_url=config.ollama.base_url,
        temperature=config.ollama.temperature,
        context_length=(
            config.ollama.context_length
        ),
        seed=config.dataset.random_seed,
    )

    pipeline = CodeAnswerPipeline(
        corpus_path=(
            config.paths.artifacts_dir
            / "corpus"
            / "qa_corpus.parquet"
        ),
        pca=load_pca_model(
            optimized_dir
            / "pca_384_to_128.joblib"
        ),
        index=load_index(
            optimized_dir
            / "hnsw_pca_128.faiss",
            expected_objects=(
                config.dataset.target_objects
            ),
            expected_dimension=(
                config.pca.output_dimension
            ),
        ),
        encoder=encoder,
        reranker=reranker,
        generator=generator,
        answerability_threshold=(
            config.reranker
            .answerability_threshold
        ),
        retrieval_candidates=(
            config.reranker
            .retrieval_candidates
        ),
        final_sources=(
            config.reranker.final_sources
        ),
        reranker_batch_size=(
            config.reranker.batch_size
        ),
        ef_search=config.hnsw.ef_search,
        maximum_answer_words=(
            config.ollama.maximum_answer_words
        ),
    )

    result = pipeline.answer(
        arguments.question
    )

    if arguments.json:
        print(json.dumps(result, indent=2))
        return

    print(result["answer"])
    print("\nSources:")

    for number, source in enumerate(
        result["sources"],
        start=1,
    ):
        print(
            f"[{number}] {source['title']} — "
            f"{source['url']}"
        )


if __name__ == "__main__":
    main()
