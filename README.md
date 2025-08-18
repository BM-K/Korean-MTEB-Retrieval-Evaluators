# Korean-MTEB-Retrieval-Evaluators
This repository provides two command-line evaluators for Korean retrieval tasks on [MTEB](https://huggingface.co/spaces/mteb/leaderboard): <br>
- `evaluate_splade.py` — a sparse evaluator that builds an inverted index over SPLADE document vectors for fast lookup. 
- `evaluate_dense.py` — a dense evaluator that encodes the corpus once with `SentenceTransformer` and runs batched semantic search.
  
Both scripts plug into MTEB and BEIR’s `EvaluateRetrieval`, reporting standard IR metrics (NDCG, MAP, Recall, Precision, and MRR) and the measured search time.

## Features
### Common
- **MTEB integration** restricted to retrieval tasks via a custom evaluator (`*MTEB.select_tasks`). 
- **Metrics & outputs**: NDCG@{1,3,5,10}, MAP@{…}, Recall@{…}, Precision@{…}, MRR@{…}; adds `"search_time"` to the result dict. Optional `--save_predictions` writes top-k results to JSON. 
- **Output folder layout**: results are written under `<output_folder>/<normalized_model_name>/`.

### SPLADE evaluator (evaluate_splade.py)
- **Inverted index**: documents are encoded with `SparseEncoder.encode_document` and only non-zero token weights are stored per token id. 
- **Query-time scoring**: queries are encoded with `encode_query`; scores are accumulated as *Σ(q_token_weight × d_token_weight)* over shared non-zero tokens. 
- **On-disk caching**: the token → postings map is cached to `./cache/<model>_<task>_<subset>_splade_index.pkl` and re-used between runs.

### Dense evaluator (evaluate_dense.py)
- **Single-pass corpus encoding** with `SentenceTransformer`; keeps the tensor in memory. Optionally applies `passage_prompt=True` for specific backbones.
- **Query prompts(prefix) for specific models**: uses `query_prompt=True` for certain embedding backbones (e.g., `telepix/PIXIE-Rune-Preview`, `Snowflake/snowflake-arctic-embed-l-v2.0`).
- **Ranking** via `util.semantic_search` over the encoded corpus.

## Installation
```
# Python 3.10+ recommended
pip install -U sentence-transformers mteb beir numpy
```

## Quick Start
### SPLADE retrieval
```
python evaluate_splade.py \
  --tasks Ko-StrategyQA AutoRAGRetrieval PublicHealthQA BelebeleRetrieval XPQARetrieval MultiLongDocRetrieval MIRACLRetrieval \
  --model telepix/PIXIE-Splade-Preview \
  --output_folder ./results_splade \
  --batch_size 32 \
```
### Dense retrieval
```
python evaluate_dense.py \
  --tasks Ko-StrategyQA AutoRAGRetrieval PublicHealthQA BelebeleRetrieval XPQARetrieval MultiLongDocRetrieval MIRACLRetrieval \
  --model telepix/PIXIE-Rune-Preview \
  --output_folder ./results_dense \
  --batch_size 32 \
```
> **Tip:**
> If you keep large corpora, prefer running with a GPU for both SPLADE encoding and dense encoding.

## How it works
### SPLADE internals
1. **Indexing**: `encode_document` returns sparse bag-of-token weights; for each document, only **non-zero token ids** are appended to the postings list. 
2. **Retrieval**: a query is encoded with `encode_query`, its non-zero token ids intersect the index, and scores accumulate as a weighted inner product on shared tokens. 
3. **Caching**: the inverted index is serialized to a pkl file under `./cache/…` and re-loaded on subsequent runs.
### Dense internals
1. **Corpus encoding**: all texts are encoded once to a tensor and kept in memory. Optionally applies `passage_prompt=True` for specific backbones.
2. **Query encoding**: optionally applies `query_prompt=True` for specific backbones; otherwise encodes raw text.
3. **Search**: ranks via `util.semantic_search` over the cached embeddings.

## Output & Saved Predictions
Each evaluator produces a per-subset `scores` dictionary with IR metrics and `search_time` in seconds. With `--save_predictions`, it writes `*_preds.json` containing the top-k doc ids and scores for each query to the configured output folder.

## Leaderboard
The table below presents the retrieval performance of several embedding models evaluated on a variety of Korean MTEB benchmarks. 
We report **Normalized Discounted Cumulative Gain (NDCG)** scores, 
which measure how well a ranked list of documents aligns with ground truth relevance. 
Higher values indicate better retrieval quality.
- **Avg. NDCG**: Average of NDCG@1, @3, @5, and @10 across all benchmark datasets.  
- **NDCG@k**: Relevance quality of the top-*k* retrieved results.

Descriptions of the benchmark datasets used for evaluation are as follows:
- **Ko-StrategyQA**  
  A Korean multi-hop open-domain question answering dataset designed for complex reasoning over multiple documents.
- **AutoRAGRetrieval**  
  A domain-diverse retrieval dataset covering finance, government, healthcare, legal, and e-commerce sectors.
- **MIRACLRetrieval**  
  A document retrieval benchmark built on Korean Wikipedia articles.
- **PublicHealthQA**  
  A retrieval dataset focused on medical and public health topics.
- **BelebeleRetrieval**  
  A dataset for retrieving relevant content from web and news articles in Korean.
- **MultiLongDocRetrieval**  
  A long-document retrieval benchmark based on Korean Wikipedia and mC4 corpus.

> **Tip:**
> While many benchmark datasets are available for evaluation, in this project we chose to use only those that contain clean positive documents for each query. Keep in mind that a benchmark dataset is just that—a benchmark. For real-world applications, it is best to construct an evaluation dataset tailored to your specific domain and evaluate embedding models, such as PIXIE, in that environment to determine the most suitable one.
  
### Dense Embedding
| Model Name | # params | Avg. NDCG | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| telepix/PIXIE-Spell-Preview-1.7B | 1.7B | 0.7567 | 0.7149 | 0.7541 | 0.7696 | 0.7882 |
| telepix/PIXIE-Spell-Preview-0.6B | 0.6B | 0.7280 | 0.6804 | 0.7258 | 0.7448 | 0.7612 |
| telepix/PIXIE-Rune-Preview | 0.5B | 0.7383 | 0.6936 | 0.7356 | 0.7545 | 0.7698 |
|  |  |  |  |  |  |  |
| nlpai-lab/KURE-v1 | 0.5B | 0.7312 | 0.6826 | 0.7303 | 0.7478 | 0.7642 |
| BAAI/bge-m3 | 0.5B | 0.7126 | 0.6613 | 0.7107 | 0.7301 | 0.7483 |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 0.5B | 0.7050 | 0.6570 | 0.7015 | 0.7226 | 0.7390 |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | 0.6872 | 0.6423 | 0.6833 | 0.7017 | 0.7215 |
| jinaai/jina-embeddings-v3 | 0.5B | 0.6731 | 0.6224 | 0.6715 | 0.6899 | 0.7088 |
| Alibaba-NLP/gte-multilingual-base | 0.3B | 0.6679 | 0.6068 | 0.6673 | 0.6892 | 0.7084 |
| openai/text-embedding-3-large | N/A | 0.6465 | 0.5895 | 0.6467 | 0.6646 | 0.6853 |

### Sparse Embedding
| Model Name | # params | Avg. NDCG | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| telepix/PIXIE-Splade-Preview | 0.1B | 0.7253 | 0.6799 | 0.7217 | 0.7416 | 0.7579 |
|  |  |  |  |  |  |  |
| [BM25](https://github.com/xhluca/bm25s) | N/A | 0.4714 | 0.4194 | 0.4708 | 0.4886 | 0.5071 |
| naver/splade-v3 | 0.1B | 0.0582 | 0.0462 | 0.0566 | 0.0612 | 0.0685 |

## License
The Korean-MTEB-Retrieval-Evaluators is licensed under MIT License.

## Citation
```
@software{Korean-MTEB-Retrieval-Evaluators,
  title={Korean MTEB Retrieval Evaluators - SPLADE & Dense},
  author={TelePIX AI Research Team},
  year={2025},
  url={https://github.com/BM-K/Korean-MTEB-Retrieval-Evaluators}
}
```

## References
[1] [telepix/PIXIE-Splade-Preview](https://huggingface.co/telepix/PIXIE-Splade-Preview) <br>
[2] [telepix/PIXIE-Rune-Preview](https://huggingface.co/telepix/PIXIE-Rune-Preview) <br>
[3] [nlpai-lab/KURE](https://github.com/nlpai-lab/KURE) <br>
