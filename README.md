# CodeAnswer

CodeAnswer is a semantic search and citation-grounded RAG system for
programming questions.

It searches 520,000 answered Stack Overflow questions, reranks the most
relevant candidates, and produces a concise answer grounded in the
original human-written sources.

The repository contains reusable Python modules and command-line scripts.
Notebooks, datasets, embeddings, indexes, and model files are not tracked
in Git.

## Architecture

```text
User question
    ↓
all-MiniLM-L6-v2 bi-encoder
    ↓
PCA projection: 384 → 128 dimensions
    ↓
FAISS HNSW retrieval: top 40
    ↓
ms-marco-MiniLM-L-6-v2 cross-encoder
    ↓
Top 3 Stack Overflow sources
    ↓
Calibrated answerability threshold
    ├── weak evidence → abstain
    └── sufficient evidence
            ↓
        Qwen2.5:3B through Ollama
            ↓
        cited answer or extractive fallback
```

## Validated iterations

### 1. Exact dense-search baseline

Normalized 384-dimensional MiniLM embeddings are stored in
`FAISS IndexFlatIP`.

### 2. PCA and HNSW optimization

PCA reduces the vectors to 128 dimensions. HNSW provides approximate
nearest-neighbor search with a controllable latency-quality trade-off.

### 3. Reranked citation-grounded RAG

The optimized retriever returns 40 candidates, a cross-encoder reranks
them, and local Qwen generates an answer from the best three sources.
The system validates citations and abstains when evidence is weak.

## Main results

### Retrieval

| System | Recall@10 | Search p95 | Index size |
|---|---:|---:|---:|
| Flat-384 | 0.7070 | 71.059 ms | 761.72 MiB |
| Flat-PCA128 | 0.6655 | 11.170 ms | 253.91 MiB |
| HNSW-PCA128, ef=128 | 0.6555 | 0.514 ms | 325.45 MiB |

The selected HNSW retriever has approximately 138 times lower search p95
than the exact baseline and uses approximately 57% less index storage.

### Final RAG pipeline

| Metric | Result |
|---|---:|
| Behavior accuracy | 96.43% |
| In-domain answer rate | 100% |
| Out-of-domain abstention | 87.5% |
| Citation coverage | 100% |
| Citation validity | 97.28% |
| Total latency p50 | 2.94 s |
| Total latency p95 | 5.48 s |

Detailed results are available in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Repository structure

```text
CodeAnswer/
├── configs/
│   └── default.yaml
├── docs/
│   └── EXPERIMENTS.md
├── results/
│   ├── experiment_summary.json
│   └── retrieval_comparison.csv
├── scripts/
│   ├── ask_codeanswer.py
│   ├── build_corpus.py
│   ├── build_flat_index.py
│   ├── build_optimized_index.py
│   ├── compare_retrievers.py
│   ├── evaluate_retrieval.py
│   └── generate_embeddings.py
├── src/codeanswer/
│   ├── comparison.py
│   ├── config.py
│   ├── data.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── indexing.py
│   ├── optimization.py
│   ├── rag.py
│   └── reranking.py
├── tests/
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.11–3.13
- Approximately 15 GiB RAM for the validated laptop setup
- Optional NVIDIA GPU for faster embedding and reranking
- Ollama with `qwen2.5:3b` for answer generation

The FAISS indexes use CPU search. MiniLM inference can use CPU or CUDA.

## Installation

```bash
git clone git@github.com:AbdullohML/CodeAnswer.git
cd CodeAnswer
python -m pip install -e .
```

For development and testing:

```bash
python -m pip install -e ".[dev]"
```

## Data preparation

Download StackSample and locate:

```text
Questions.csv
Answers.csv
```

Build the 520,000-object corpus:

```bash
PYTHONPATH=src python scripts/build_corpus.py \
  --source-dir /path/to/stacksample
```

Each question is joined to its highest-scored answer. HTML is removed and
the resulting collection is saved as Parquet.

## Generate embeddings

```bash
PYTHONPATH=src python scripts/generate_embeddings.py \
  --device cuda
```

The pipeline writes normalized vectors incrementally to a resumable
memory-mapped float32 file.

## Build the exact baseline

```bash
PYTHONPATH=src python scripts/build_flat_index.py
```

Evaluate it:

```bash
PYTHONPATH=src python scripts/evaluate_retrieval.py \
  --device cuda
```

## Build the optimized retriever

```bash
PYTHONPATH=src python scripts/build_optimized_index.py
```

Compare all retrieval configurations:

```bash
PYTHONPATH=src python scripts/compare_retrievers.py \
  --device cuda
```

## Ask CodeAnswer

Start Ollama:

```bash
ollama serve
```

Install the local generator:

```bash
ollama pull qwen2.5:3b
```

Ask a programming question:

```bash
PYTHONPATH=src python scripts/ask_codeanswer.py \
  "Why does an asynchronous JavaScript loop return the wrong result?" \
  --device cuda
```

Return full JSON containing scores, latency, and sources:

```bash
PYTHONPATH=src python scripts/ask_codeanswer.py \
  "How do I split a string in SQL?" \
  --device cuda \
  --json
```

## Testing

```bash
PYTHONPATH=src python -m pytest -q
```

The test suite covers corpus construction, resumable embeddings, FAISS
indexing, retrieval metrics, PCA, HNSW, reranking, threshold calibration,
citation validation, and safe fallback behavior.

## Generated artifacts

The following are intentionally excluded from Git:

```text
data/
artifacts/
models/
*.parquet
*.f32
*.faiss
*.index
```

## Limitations

- The source collection is historical and may contain outdated answers.
- Retrieval evaluation uses a known-item task rather than live user labels.
- The answerability and RAG evaluation sets are relatively small.
- The reported faithfulness value is an embedding-based proxy.
- Qwen generation is the main latency bottleneck.
- Invalid citation formatting often triggers the extractive fallback.

## License

MIT
