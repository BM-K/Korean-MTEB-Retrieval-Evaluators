# Korean-MTEB-Retrieval-Evaluators
This repository provides two command-line evaluators for Korean (and multilingual) retrieval tasks on [MTEB]: <br>
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
- **Query-time scoring**: queries are encoded with `encode_query`; scores are accumulated as Σ(q_token_weight × d_token_weight) over shared non-zero tokens. 
- **On-disk caching**: the token → postings map is cached to `./cache/<model>_<task>_<subset>_splade_index.pkl` and re-used between runs.

### Dense evaluator (evaluate_dense.py)
- **Single-pass corpus encoding** with `SentenceTransformer`; keeps the tensor in memory.
- **Query prompts(prefix) for specific models**: uses prompt_name="query" for certain embedding backbones (e.g., `telepix/PIXIE-Rune-Preview`, `Snowflake/snowflake-arctic-embed-l-v2.0`).
- **Ranking** via `util.semantic_search` over the encoded corpus.

## Installation
```
# Python 3.10+ recommended
pip install -U sentence-transformers mteb beir numpy
```
> **Tip:**
> If you keep large corpora, prefer running with a GPU for both SPLADE encoding and dense encoding.

## Quick Start
### SPLADE retrieval
```
python evaluate_splade.py \
  --tasks Ko-StrategyQA AutoRAGRetrieval PublicHealthQA BelebeleRetrieval XPQARetrieval MultiLongDocRetrieval MIRACLRetrieval \
  --model telepix/PIXIE-Splade-Preview \
  --output_folder ./results_splade \
  --save_predictions
```
The script builds or loads an inverted index, retrieves top-k per query, and reports metrics and `search_time`. 
### Dense retrieval
```
python evaluate_dense.py \
  --tasks Ko-StrategyQA AutoRAGRetrieval PublicHealthQA BelebeleRetrieval XPQARetrieval MultiLongDocRetrieval MIRACLRetrieval \
  --model telepix/PIXIE-Rune-Preview \
  --output_folder ./results_dense \
  --save_predictions
```
The script encodes the corpus once, then performs semantic search for each query and evaluates with BEIR.
