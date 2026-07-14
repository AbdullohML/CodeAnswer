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
