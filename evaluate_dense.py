import os
import time
import json
import logging
import argparse
import numpy as np
from time import time
from pathlib import Path
from typing import Dict, List
from mteb.evaluation import MTEB
from typing import Dict, List, Tuple
from mteb.abstasks import AbsTaskRetrieval
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("dense_ko")

class DenseSearch:
    """
    Wrapper around SentenceTransformer for dense retrieval.
    Attributes:
        model: Loaded SentenceTransformer model
        doc_embeddings: Cached tensor of document embeddings
        doc_ids: Corresponding list of document IDs
    """
    def __init__(self, model_name: str = "telepix/PIXIE-Rune-Preview", batch_size: int = 8):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None
        self.doc_ids = None
        self.bch = batch_size

    def search(self, 
        corpus: Dict[str, Dict],
        queries: Dict[str, Dict],
        top_k: int = 10, 
        scores_function=None, 
        **kwargs
    ) -> Tuple[Dict[str, Dict[str, float]], float]:
        # Prepare document texts and IDs
        self.doc_ids = list(corpus.keys())
        documents = []
        for did in self.doc_ids:
            info = corpus[did]
            if isinstance(info, dict):
                # Use title + text or content
                text = (info.get("title", "") + " " + info.get("text", info.get("content", ""))).strip()
            else:
                text = str(info).strip()
        
            documents.append(text)

        logger.info("Encoding corpus...")
        # Encode all documents once
        self.doc_embeddings = self.model.encode(
            documents, 
            convert_to_tensor=True, 
            batch_size=self.bch, 
            show_progress_bar=True
        )
        
        results = {}
        start = time()
        for qid, qtext in queries.items():
            if self.model_name == 'telepix/PIXIE-Rune-Preview':
                q_emb = self.model.encode(qtext, convert_to_tensor=True, prompt_name="query")
            elif self.model_name == 'Snowflake/snowflake-arctic-embed-l-v2.0':
                q_emb = self.model.encode(qtext, convert_to_tensor=True, prompt_name="query")
            else:
                q_emb = self.model.encode(qtext, convert_to_tensor=True)

            hits = util.semantic_search(q_emb, self.doc_embeddings, top_k=10)[0]
            results[qid] = {self.doc_ids[hit['corpus_id']]: float(hit['score']) for hit in hits}
        end = time()

        return results, end-start

class DenseMTEB(MTEB):
    def select_tasks(self, **kwargs):
        super().select_tasks(**kwargs)
        self.tasks = [t for t in self.tasks if isinstance(t, AbsTaskRetrieval)]

    def _run_eval(self, task, model, split, output_folder, **kwargs):
        if not isinstance(task, AbsTaskRetrieval):
            raise ValueError("Only retrieval tasks can be evaluated.")
        tick = time()
        scores = self.evaluate_task(model, task, split, output_folder=output_folder, **kwargs)
        tock = time()
        return scores, tick, tock

    def evaluate_task(self, model, task, split="test", **kwargs):
        scores = {}
        subsets = list(task.hf_subsets) if getattr(task, "is_multilingual", False) else ["default"]

        for hf in subsets:
            logger.info(f"[DenseMTEB] Evaluating subset: {hf}")
            corpus = task.corpus[hf][split] if hf != "default" else task.corpus[split]
            queries = task.queries[hf][split] if hf != "default" else task.queries[split]
            qrels = task.relevant_docs[hf][split] if hf != "default" else task.relevant_docs[split]

            scores[hf] = self._evaluate_subset(model, task, corpus, queries, qrels, hf, task.metadata.main_score, **kwargs)
        return scores

    def _evaluate_subset(self, model, task, corpus, queries, qrels, hf_subset, main_score, k_values=[1, 3, 5, 10], **kwargs):
        retriever = EvaluateRetrieval(model)
        results, total_search_time = retriever.retrieve(corpus, queries)
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)
        mrr = retriever.evaluate_custom(qrels, results, k_values, "mrr")

        scores = {f"ndcg_at_{k.split('@')[-1]}": v for k, v in ndcg.items()}
        scores.update({f"map_at_{k.split('@')[-1]}": v for k, v in _map.items()})
        scores.update({f"recall_at_{k.split('@')[-1]}": v for k, v in recall.items()})
        scores.update({f"precision_at_{k.split('@')[-1]}": v for k, v in precision.items()})
        scores.update({f"mrr_at_{k.split('@')[-1]}": v for k, v in mrr.items()})
        scores["main_score"] = scores.get(main_score, next(iter(scores.values())))
        scores["search_time"] = total_search_time

        if kwargs.get("save_predictions", False):
            out_path = Path(kwargs.get("output_folder", ".")) / f"{task.metadata.name}_{hf_subset}_preds.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--task_langs", nargs="+", default=["kor-Kore", "kor-Hang", "kor_Hang", "ko", "korean", "kor-kor"])
    parser.add_argument("--eval_splits", nargs="+", default=None)
    parser.add_argument("--model", type=str, default="telepix/PIXIE-Rune-Preview")
    parser.add_argument("--output_folder", type=str, default="./results_dense")
    parser.add_argument("--verbosity", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_predictions", action="store_true")
    args = parser.parse_args()

    if args.verbosity <= 1:
        logging.getLogger("mteb").setLevel(logging.WARNING)
    elif args.verbosity == 0:
        logging.getLogger("mteb").setLevel(logging.CRITICAL)
    
    # Initialize evaluator and retrieval model
    evaluator = DenseMTEB(tasks=args.tasks, task_langs=args.task_langs)
    model = DenseSearch(model_name=args.model, batch_size=args.batch_size)
    
    # Normalize model name for output path
    if '/' in args.model:
        model_name = f"{args.model.split('/')[0]}_{args.model.split('/')[-1]}"
    else:
        model_name = args.model

    output_folder = f"{args.output_folder}/{model_name}"
    run_kwargs = {
        "model": model,
        "output_folder": output_folder,
        "verbosity": args.verbosity
    }
    if args.eval_splits:
        run_kwargs["eval_splits"] = args.eval_splits
    if args.save_predictions:
        run_kwargs["save_predictions"] = True

    evaluator.run(**run_kwargs)


if __name__ == "__main__":
    main()

