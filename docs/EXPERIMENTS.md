# CodeAnswer Experiments

## Evaluation protocol

The experiments use 520,000 answered Stack Overflow questions.

- Query: question title
- Indexed document: question body
- Returned content: highest-scored answer
- Relevant result: document belonging to the same question
- Retrieval metrics: Recall@k, MRR@10, and nDCG@10
- Performance metrics: QPS, latency, memory, and index size

This known-item task provides a reproducible offline evaluation, although
it does not fully represent real user queries.

## Iteration 1: exact dense retrieval

The baseline uses `all-MiniLM-L6-v2` as a bi-encoder and stores normalized
384-dimensional vectors in `FAISS IndexFlatIP`.

| Metric | Result |
|---|---:|
| Objects | 520,000 |
| Recall@1 | 0.4850 |
| Recall@5 | 0.6465 |
| Recall@10 | 0.7070 |
| MRR@10 | 0.5547 |
| nDCG@10 | 0.5912 |
| Search p95 | 71.059 ms |
| Index size | 761.72 MiB |

The exact index is used as the retrieval-quality reference.

## Iteration 2: PCA and HNSW

PCA reduces the vectors from 384 to 128 dimensions and preserves 79.53%
of the measured variance. HNSW then replaces exhaustive vector scanning.

| System | Recall@10 | QPS | p95 | Size |
|---|---:|---:|---:|---:|
| Flat-384 | 0.7070 | 193.13 | 71.059 ms | 761.72 MiB |
| Flat-PCA128 | 0.6655 | 590.26 | 11.170 ms | 253.91 MiB |
| HNSW-PCA128, ef=64 | 0.6430 | 14,345.05 | 0.310 ms | 325.45 MiB |
| HNSW-PCA128, ef=128 | 0.6555 | 7,672.06 | 0.514 ms | 325.45 MiB |

The final retriever uses `efSearch=128`. Compared with Flat-384, its
single-query search p95 is approximately 138 times lower and its index is
approximately 57% smaller. Recall@10 decreases by 0.0515.

## Iteration 3: reranked citation-grounded RAG

The final pipeline retrieves 40 candidates, reranks them using
`cross-encoder/ms-marco-MiniLM-L-6-v2`, and gives the best three sources
to local `qwen2.5:3b`.

A threshold calibrated on separate in-domain and out-of-domain queries
decides whether the retrieved evidence is answerable.

| Metric | Result |
|---|---:|
| Calibration F1 | 0.9877 |
| Behavior accuracy | 0.9643 |
| In-domain answer rate | 1.0000 |
| Out-of-domain abstention | 0.8750 |
| Citation coverage | 1.0000 |
| Citation validity | 0.9728 |
| Faithfulness proxy | 0.6667 |
| Extractive fallback rate | 0.6190 |
| Retrieval p50 | 7.619 ms |
| Reranking p50 | 167.517 ms |
| Generation p50 | 2,493.236 ms |
| Total p50 | 2,941.517 ms |
| Total p95 | 5,476.624 ms |

When generated citation formatting fails validation, CodeAnswer returns a
shortened version of a retrieved human-written answer instead of exposing
an unsupported generated response.

## Limitations

- StackSample is historical, so some answers may be outdated.
- The known-item task does not fully represent natural user queries.
- The faithfulness metric is an embedding-based proxy, not official RAGAS.
- The answerability evaluation set is relatively small.
- One tested out-of-domain query incorrectly passed the answerability gate.
- Generation is the dominant latency component.
- The extractive fallback rate is high.
