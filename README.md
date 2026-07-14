# CodeAnswer

CodeAnswer is a semantic search and citation-grounded RAG system for
programming questions.

It searches a collection of 520,000 answered Stack Overflow questions,
reranks retrieved candidates, and produces a concise answer grounded in
the original Stack Overflow sources.

The repository is being built incrementally from the validated project
experiments. Full setup, architecture, evaluation results, and usage
instructions will be added as the implementation is completed.

## Corpus preparation

The corpus builder expects the StackSample files:

- `Questions.csv`
- `Answers.csv`

For each question, CodeAnswer selects the highest-scored answer,
removes HTML tags, and creates a deterministic collection of 520,000
question-answer objects.

```bash
PYTHONPATH=src python scripts/build_corpus.py \
  --source-dir ~/.cache/kagglehub/datasets/stackoverflow/stacksample/versions/2
```

Generated files are stored under:

```text
artifacts/corpus/
├── qa_corpus.parquet
└── qa_corpus.manifest.json
```

Generated datasets and model artifacts are excluded from Git.

## Embedding generation

CodeAnswer uses `sentence-transformers/all-MiniLM-L6-v2` as its
bi-encoder. Question bodies are encoded independently into normalized
384-dimensional vectors.

The embedding pipeline writes the vectors incrementally to a
memory-mapped float32 file and can resume after interruption.

```bash
PYTHONPATH=src python scripts/generate_embeddings.py
```

Select a device explicitly when needed:

```bash
PYTHONPATH=src python scripts/generate_embeddings.py \
  --device cuda \
  --batch-size 64
```

Generated artifacts:

```text
artifacts/embeddings/
├── question_body_minilm_384.f32
└── question_body_minilm_384.f32.manifest.json
```

The row at position `i` corresponds to corpus `doc_id = i`.
Generated embeddings are excluded from Git.

## Exact retrieval baseline

The first retrieval system uses a FAISS `IndexFlatIP` index over
normalized 384-dimensional MiniLM vectors.

Because both document and query vectors are L2-normalized, inner-product
ranking is equivalent to cosine-similarity ranking.

```bash
PYTHONPATH=src python scripts/build_flat_index.py
```

Generated artifacts:

```text
artifacts/indexes/
├── flat_ip_384.faiss
└── flat_ip_384.faiss.manifest.json
```

This exact index is used as the quality reference for the later PCA and
HNSW optimization experiments.

## Retrieval evaluation

Retrieval quality is evaluated using a known-item task. A Stack Overflow
question title is used as the search query, while the corresponding
question body is treated as the relevant document.

The evaluation reports:

- Recall@1, Recall@5, and Recall@10
- MRR@10
- nDCG@10
- batch search throughput
- mean, p50, and p95 single-query search latency

```bash
PYTHONPATH=src python scripts/evaluate_retrieval.py \
  --device cuda
```

Results are written to:

```text
results/iteration_1/exact_dense_baseline/
├── results.json
└── per_query_metrics.csv
```

## PCA and HNSW optimization

The optimized retrieval stage reduces MiniLM embeddings from 384 to 128
dimensions using PCA. The reduced vectors are normalized again and
stored in a FAISS HNSW index.

```bash
PYTHONPATH=src python scripts/build_optimized_index.py
```

Default parameters:

```text
PCA dimensions: 384 → 128
PCA training sample: 100,000 vectors
HNSW M: 16
HNSW efConstruction: 100
HNSW efSearch: 128
```

Generated artifacts:

```text
artifacts/optimized/
├── question_body_pca_128.f32
├── question_body_pca_128.f32.manifest.json
├── pca_384_to_128.joblib
├── hnsw_pca_128.faiss
└── hnsw_pca_128.faiss.manifest.json
```

The exact 384-dimensional index remains the quality reference, while the
PCA-HNSW index provides lower memory usage and substantially faster
search.
