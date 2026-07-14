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
