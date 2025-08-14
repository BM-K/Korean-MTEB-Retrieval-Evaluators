import os
import torch
import pickle
import logging
import argparse
import numpy as np
from time import time
from pathlib import Path
from mteb.evaluation import MTEB
from collections import defaultdict
from torch.nn.functional import normalize
from mteb.abstasks import AbsTaskRetrieval
from typing import List, Optional, Union, Dict
from sentence_transformers import SparseEncoder
from beir.retrieval.evaluation import EvaluateRetrieval

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger("splade_ko")

class SPLADESearch:
    def __init__(self, model_name: str = "telepix/PIXIE-Splade-Preview", device: Optional[str] = None, index_cache_path: Optional[str] = None, batch_size: int = 32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading SPLADE model on {self.device}...")
        self.model = SparseEncoder(model_name, model_kwargs={}).to(self.device)
        self.inverted_index: Dict[int, List[tuple[int, float]]] = {}
        self.doc_ids: List[str] = []
        self.index_cache_path = index_cache_path
        self.batch_size = batch_size

    def _build_index(self, texts: List[str]):
        logger.info("Building inverted index from scratch...*^*...")
        with torch.no_grad():
            doc_emb = self.model.encode_document(texts, batch_size=self.batch_size).to("cpu")
        doc_emb = doc_emb.to_dense().numpy()
        self.doc_ids = list(range(len(texts)))
        
        index = defaultdict(list)
        for doc_idx, vec in enumerate(doc_emb):
            nz = np.nonzero(vec)[0]

            for token_id in nz:
                index[token_id].append((doc_idx, float(vec[token_id])))
        self.inverted_index = index
        logger.info(f"Inverted index built: {len(self.doc_ids)} docs, {len(index)} tokens.")

        if self.index_cache_path:
            with open(self.index_cache_path, "wb") as f:
                pickle.dump((self.inverted_index, self.doc_ids), f)
            logger.info(f"Inverted index cached to: {self.index_cache_path}")

    def _load_index(self):
        if self.index_cache_path and os.path.exists(self.index_cache_path):
            logger.info(f"Loading cached inverted index from: {self.index_cache_path}")
            with open(self.index_cache_path, "rb") as f:
                self.inverted_index, self.doc_ids = pickle.load(f)
            return True
        return False

    def search(self, corpus: dict, queries: dict, top_k: int = 10, scores_function=None, **kwargs) -> dict:
        docs = []
        texts = []
        top_k = 10
        for did, info in corpus.items():
            if isinstance(info, dict):
                text = (info.get("title", "") + " " + info.get("text", info.get("content", ""))).strip()
            else:
                text = str(info)
            docs.append(did)
            texts.append(text)

        if not self._load_index():
            self._build_index(texts)
        
        # Prepare query texts
        q_ids = list(queries.keys())
        q_texts = [queries[qid] for qid in q_ids]
        _ = self.model.encode_query('warmup')

        results = {}
        search_start = time()

        # Encode all queries in a batch
        with torch.no_grad():
            q_embs = self.model.encode_query(q_texts, batch_size=self.batch_size).to("cpu")
        q_embs = q_embs.to_dense().numpy()
        
        # Iterate through the batch of query embeddings to get search results
        for i, vec in enumerate(q_embs):
            q_id = q_ids[i]
            # Get non-zero dimensions (tokens) for the current query
            nz = np.nonzero(vec)[0]
            scores: Dict[int, float] = defaultdict(float)
            
            # Calculate dot product scores using the inverted index
            for token_id in nz:
                qw = float(vec[token_id]) # Query weight for the token
                # Retrieve documents containing this token
                for doc_idx, dw in self.inverted_index.get(token_id, []):
                    scores[doc_idx] += qw * dw
            
            # Get the top-k documents
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            results[q_id] = {docs[doc_idx]: score for doc_idx, score in top}
        
        search_end = time()
        search_time = search_end - search_start
        logger.info(f"[SPLADEMTEB] Search time only:     {search_time:.2f}s")

        return results, search_time


class SPLADEMTEB(MTEB):
    def select_tasks(self, **kwargs):
        super().select_tasks(**kwargs)
        self.tasks = [t for t in self.tasks if isinstance(t, AbsTaskRetrieval)]
    
    def _run_eval(self, task, model, split, output_folder, batch_size, **kwargs):
        if not isinstance(task, AbsTaskRetrieval):
            raise ValueError("Only retrieval tasks can be evaluated with SPLADEMTEB")
        
        tick = time()
        scores = self.evaluate_task(model, task, batch_size, split, output_folder=output_folder, **kwargs)
        tock = time()
        return scores, tick, tock

    def evaluate_task(self, model, task, batch_size, split="test", **kwargs):
        scores = {}
        subsets = list(task.hf_subsets) if getattr(task, "is_multilingual", False) else ["default"]

        for hf in subsets:
            logger.info(f"[SPLADEMTEB] Evaluating subset: {hf}")
            if hf == "default":
                corpus = task.corpus[split]
                queries = task.queries[split]
                qrels = task.relevant_docs[split]
            else:
                corpus = task.corpus[hf][split]
                queries = task.queries[hf][split]
                qrels = task.relevant_docs[hf][split]

            scores[hf] = self._evaluate_subset(
                model, task, corpus, queries, qrels, hf, task.metadata.main_score, batch_size, **kwargs
            )
        return scores

    def _evaluate_subset(
        self,
        model,
        task,
        corpus: dict,
        queries: dict,
        relevant_docs: dict,
        hf_subset: str,
        main_score: str,
        batch_size: int,
        k_values=[1, 3, 5, 10],
        **kwargs,
    ):
        if '/' in model:
            model_name = f"{model.split('/')[0]}_{model.split('/')[-1]}"
        else:
            model_name = model
        
        index_cache_file = f"./cache/{model_name}_{task.metadata.name}_{hf_subset}_splade_index.pkl"
        os.makedirs("./cache", exist_ok=True)

        splade = SPLADESearch(model_name=model, index_cache_path=index_cache_file, batch_size=batch_size)

        index_start = time()
        beir_ret = EvaluateRetrieval(retriever=splade)
        index_end = time()
        
        logger.info(f"[SPLADEMTEB] Index load/build time: {index_end - index_start:.2f}s")

        results, search_time = beir_ret.retrieve(corpus, queries)

        ndcg, _map, recall, precision = beir_ret.evaluate(
            relevant_docs, results, k_values, ignore_identical_ids=True
        )
        mrr = beir_ret.evaluate_custom(relevant_docs, results, k_values, "mrr")

        scores = {f"ndcg_at_{k.split('@')[-1]}": v for k, v in ndcg.items()}
        scores.update({f"map_at_{k.split('@')[-1]}": v for k, v in _map.items()})
        scores.update({f"recall_at_{k.split('@')[-1]}": v for k, v in recall.items()})
        scores.update({f"precision_at_{k.split('@')[-1]}": v for k, v in precision.items()})
        scores.update({f"mrr_at_{k.split('@')[-1]}": v for k, v in mrr.items()})
        scores["main_score"] = scores.get(main_score, next(iter(scores.values())))
        scores["search_time"] = search_time

        if kwargs.get("save_predictions", False):
            out_path = Path(kwargs.get("output_folder", ".")) / f"{self.metadata_dict['name']}_{hf_subset}_preds.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate SPLADE on Korean MTEB Retrieval Tasks")
    parser.add_argument("--tasks", nargs="+", required=True, help="MTEB retrieval task names")
    parser.add_argument(
        "--task_langs", nargs="+", default=["kor-Kore", "kor-Hang", "kor_Hang", "ko", "korean", "kor-kor"], 
        help="HF language subsets for Korean"
    )
    parser.add_argument(
        "--eval_splits", nargs="+", default=None,
        help="Splits to evaluate (default: all available)"
    )
    parser.add_argument(
        "--model", type=str, default="telepix/PIXIE-Splade-Preview",
        help="SPLADE model name or path"
    )
    parser.add_argument(
        "--output_folder", type=str, default="./results_splade/",
        help="Directory to write results"
    )
    parser.add_argument(
        "--verbosity", type=int, default=2,
        help="Logging level: 0-CRITICAL,1-WARNING,2-INFO,3-DEBUG"
    )
    parser.add_argument(
        "--save_predictions", action="store_true",
        help="Save top-k predictions to JSON"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.verbosity <= 1:
        logging.getLogger("mteb").setLevel(logging.WARNING)
    elif args.verbosity == 0:
        logging.getLogger("mteb").setLevel(logging.CRITICAL)

    logger.info(f"Tasks: {args.tasks}, Splits: {args.eval_splits or 'all'}")
    evaluator = SPLADEMTEB(tasks=args.tasks, task_langs=args.task_langs)

    if '/' in args.model:
        model_name = f"{args.model.split('/')[0]}_{args.model.split('/')[-1]}"
    else:
        model_name = args.model

    output_folder = f"{args.output_folder}/{model_name}"

    run_kwargs = {"model": args.model, "output_folder": output_folder, "verbosity": args.verbosity, "batch_size": args.batch_size}
    if args.save_predictions:
        run_kwargs["save_predictions"] = True
    if args.eval_splits is not None:
        run_kwargs["eval_splits"] = args.eval_splits

    evaluator.run(**run_kwargs)


if __name__ == "__main__":
    main()


