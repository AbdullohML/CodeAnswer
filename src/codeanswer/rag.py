"""Citation-grounded RAG pipeline for CodeAnswer."""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import pyarrow.dataset as pads
import requests

from codeanswer.indexing import search_index
from codeanswer.optimization import transform_queries
from codeanswer.reranking import PairScorer, rerank_candidates


ABSTENTION_TEXT = (
    "I could not find enough reliable evidence "
    "in the Stack Overflow collection."
)

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class Generator(Protocol):
    """Interface required by the RAG pipeline."""

    model_name: str

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Generate one response."""


@dataclass
class OllamaGenerator:
    """Generate answers through a local Ollama server."""

    model_name: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    context_length: int = 4096
    seed: int = 20260714

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": self.temperature,
                    "num_predict": max_tokens,
                    "num_ctx": self.context_length,
                    "seed": self.seed,
                },
            },
            timeout=300,
        )
        response.raise_for_status()

        payload = response.json()

        return {
            "text": payload["message"]["content"].strip(),
            "elapsed_ms": round(
                (time.perf_counter() - started_at) * 1000,
                3,
            ),
            "prompt_tokens": payload.get(
                "prompt_eval_count",
                0,
            ),
            "output_tokens": payload.get(
                "eval_count",
                0,
            ),
        }


def clean_text(value: str) -> str:
    """Normalize HTML entities and whitespace."""

    return re.sub(
        r"\s+",
        " ",
        html.unescape(value or ""),
    ).strip()


def citation_metrics(
    answer: str,
    *,
    source_count: int,
) -> dict[str, float]:
    """Measure citation coverage and citation-number validity."""

    paragraphs = [
        paragraph.strip()
        for paragraph in answer.splitlines()
        if len(paragraph.strip()) >= 15
    ]

    if not paragraphs:
        return {
            "citation_coverage": 0.0,
            "citation_validity": 0.0,
        }

    cited_paragraphs = 0
    citation_numbers: list[int] = []

    for paragraph in paragraphs:
        numbers = [
            int(value)
            for value in CITATION_PATTERN.findall(paragraph)
        ]

        if numbers:
            cited_paragraphs += 1
            citation_numbers.extend(numbers)

    valid_citations = sum(
        1 <= number <= source_count
        for number in citation_numbers
    )

    return {
        "citation_coverage": (
            cited_paragraphs / len(paragraphs)
        ),
        "citation_validity": (
            valid_citations / len(citation_numbers)
            if citation_numbers
            else 0.0
        ),
    }


def build_context(sources: pd.DataFrame) -> str:
    """Format reranked sources for generation."""

    blocks = []

    for index, row in sources.iterrows():
        number = index + 1

        blocks.append(
            f"[{number}]\n"
            f"Title: {row['title']}\n"
            f"Question: {row['question_body'][:700]}\n"
            f"Answer: {row['answer'][:1400]}\n"
            f"URL: {row['url']}"
        )

    return "\n\n".join(blocks)


def extractive_fallback(
    sources: pd.DataFrame,
    *,
    maximum_words: int,
) -> str:
    """Return a shortened human-written answer."""

    if sources.empty:
        return ABSTENTION_TEXT

    words = clean_text(
        str(sources.iloc[0]["answer"])
    ).split()

    answer = " ".join(words[:maximum_words])

    if len(words) > maximum_words:
        answer += " ..."

    return f"{answer} [1]"


def generate_grounded_answer(
    query: str,
    sources: pd.DataFrame,
    *,
    generator: Generator,
    maximum_words: int,
) -> dict[str, Any]:
    """Generate a cited answer or use a safe extractive fallback."""

    system_prompt = f"""
You are CodeAnswer, a programming assistant.

Use only the supplied Stack Overflow sources.
Give the simplest supported answer.
Do not invent missing information.
Every paragraph must contain a citation such as [1].
Use only source numbers present in the context.
Keep the answer below {maximum_words} words.
Do not add a separate Sources section.
""".strip()

    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Sources:\n{build_context(sources)}\n\n"
        "Write the answer."
    )

    result = generator.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=320,
    )

    answer = str(result["text"]).strip()

    metrics = citation_metrics(
        answer,
        source_count=len(sources),
    )

    used_fallback = (
        metrics["citation_validity"] < 1.0
        or metrics["citation_coverage"] < 0.5
    )

    if used_fallback:
        answer = extractive_fallback(
            sources,
            maximum_words=maximum_words,
        )

        metrics = citation_metrics(
            answer,
            source_count=len(sources),
        )

    return {
        "answer": answer,
        "generation_ms": float(
            result.get("elapsed_ms", 0.0)
        ),
        "prompt_tokens": int(
            result.get("prompt_tokens", 0)
        ),
        "output_tokens": int(
            result.get("output_tokens", 0)
        ),
        "used_fallback": used_fallback,
        **metrics,
    }


class CodeAnswerPipeline:
    """Retrieve, rerank, and answer programming questions."""

    def __init__(
        self,
        *,
        corpus_path: str | Path,
        pca: Any,
        index: Any,
        encoder: Any,
        reranker: PairScorer,
        generator: Generator,
        answerability_threshold: float,
        retrieval_candidates: int = 40,
        final_sources: int = 3,
        reranker_batch_size: int = 32,
        ef_search: int = 128,
        maximum_answer_words: int = 140,
    ) -> None:
        self.corpus_path = Path(
            corpus_path
        ).expanduser().resolve()

        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus not found: {self.corpus_path}"
            )

        self.dataset = pads.dataset(
            self.corpus_path,
            format="parquet",
        )

        self.pca = pca
        self.index = index
        self.encoder = encoder
        self.reranker = reranker
        self.generator = generator
        self.answerability_threshold = answerability_threshold
        self.retrieval_candidates = retrieval_candidates
        self.final_sources = final_sources
        self.reranker_batch_size = reranker_batch_size
        self.maximum_answer_words = maximum_answer_words

        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = ef_search

    def _fetch_documents(
        self,
        document_ids: list[int],
    ) -> pd.DataFrame:
        table = self.dataset.to_table(
            filter=pads.field("doc_id").isin(document_ids)
        )

        frame = (
            table.to_pandas()
            .set_index("doc_id")
            .loc[document_ids]
            .reset_index()
        )

        for column in [
            "title",
            "question_body",
            "answer",
        ]:
            frame[column] = frame[column].map(clean_text)

        return frame

    def retrieve_and_rerank(
        self,
        query: str,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        retrieval_started_at = time.perf_counter()

        query_vector = self.encoder.encode(
            [query],
            batch_size=1,
        )

        reduced_query = transform_queries(
            self.pca,
            query_vector,
        )

        vector_scores, vector_ids = search_index(
            self.index,
            reduced_query,
            top_k=self.retrieval_candidates,
        )

        retrieval_ms = (
            time.perf_counter() - retrieval_started_at
        ) * 1000

        document_ids = [
            int(document_id)
            for document_id in vector_ids[0]
            if document_id >= 0
        ]

        candidates = self._fetch_documents(document_ids)

        candidates.insert(
            1,
            "vector_similarity",
            vector_scores[0][:len(candidates)],
        )

        rerank_started_at = time.perf_counter()

        sources = rerank_candidates(
            query,
            candidates,
            scorer=self.reranker,
            top_k=self.final_sources,
            batch_size=self.reranker_batch_size,
        )

        rerank_ms = (
            time.perf_counter() - rerank_started_at
        ) * 1000

        return sources, {
            "retrieval_ms": round(retrieval_ms, 3),
            "rerank_ms": round(rerank_ms, 3),
        }

    def answer(self, query: str) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("The query cannot be empty.")

        total_started_at = time.perf_counter()

        sources, timing = self.retrieve_and_rerank(query)

        top_score = float(
            sources.iloc[0]["reranker_score"]
        )

        if top_score < self.answerability_threshold:
            return {
                "query": query,
                "answer": ABSTENTION_TEXT,
                "abstained": True,
                "top_reranker_score": top_score,
                **timing,
                "generation_ms": 0.0,
                "total_ms": round(
                    (
                        time.perf_counter()
                        - total_started_at
                    )
                    * 1000,
                    3,
                ),
                "used_fallback": False,
                "sources": sources.to_dict(
                    orient="records"
                ),
            }

        generation = generate_grounded_answer(
            query,
            sources,
            generator=self.generator,
            maximum_words=self.maximum_answer_words,
        )

        return {
            "query": query,
            "abstained": False,
            "top_reranker_score": top_score,
            **timing,
            **generation,
            "total_ms": round(
                (
                    time.perf_counter()
                    - total_started_at
                )
                * 1000,
                3,
            ),
            "sources": sources.to_dict(
                orient="records"
            ),
        }
