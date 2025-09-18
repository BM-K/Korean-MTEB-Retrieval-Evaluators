import os
import re
import json
import time
import pickle
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch

from mteb.evaluation import MTEB
from mteb.abstasks import AbsTaskRetrieval
from beir.retrieval.evaluation import EvaluateRetrieval

from sentence_transformers import SparseEncoder
from sentence_transformers import CrossEncoder as STCrossEncoder

from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("rerank_splade_qwen3")


# ---------------------------
# Utilities
# ---------------------------
def build_doc_text(entry) -> str:
    if isinstance(entry, dict):
        return (entry.get("title", "") + " " + entry.get("text", entry.get("content", ""))).strip()
    return str(entry).strip()


def get_query_text(qv) -> str:
    if isinstance(qv, dict):
        return qv.get("text", qv.get("content", "")).strip()
    return str(qv).strip()


def safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", name)


# ---------------------------
# SPLADE Retriever (Candidate Generator)
# ---------------------------
class SPLADESearch:
    """
    SPLADE 기반 후보 생성기
    index cache
    """
    def __init__(
        self,
        model_name: str = "telepix/PIXIE-Splade-Preview",
        device: Optional[str] = None,
        index_cache_path: Optional[str] = None,
        batch_size: int = 32,
        candidate_k: int = 100,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[SPLADE] Loading model on {self.device}: {model_name}")
        self.model = SparseEncoder(model_name, model_kwargs={}).to(self.device)

        self.inverted_index: Dict[int, List[Tuple[int, float]]] = {}
        self.doc_ids_seq: List[str] = []
        self.index_cache_path = index_cache_path
        self.batch_size = batch_size
        self.candidate_k = candidate_k

    def _build_index(self, texts: List[str]):
        logger.info("[SPLADE] Building inverted index from scratch...")
        with torch.no_grad():
            doc_emb = self.model.encode_document(texts, batch_size=self.batch_size)
        doc_emb = doc_emb.to("cpu").to_dense().numpy()

        self.doc_ids_seq = list(range(len(texts)))
        index = defaultdict(list)
        for doc_idx, vec in enumerate(doc_emb):
            nz = np.nonzero(vec)[0]
            for token_id in nz:
                index[token_id].append((doc_idx, float(vec[token_id])))
        self.inverted_index = index
        logger.info(f"[SPLADE] Index ready: {len(self.doc_ids_seq)} docs, {len(index)} tokens.")

        if self.index_cache_path:
            os.makedirs(os.path.dirname(self.index_cache_path), exist_ok=True)
            with open(self.index_cache_path, "wb") as f:
                pickle.dump((self.inverted_index, self.doc_ids_seq), f)
            logger.info(f"[SPLADE] Cached to: {self.index_cache_path}")

    def _load_index(self) -> bool:
        if self.index_cache_path and os.path.exists(self.index_cache_path):
            logger.info(f"[SPLADE] Loading cached index: {self.index_cache_path}")
            with open(self.index_cache_path, "rb") as f:
                self.inverted_index, self.doc_ids_seq = pickle.load(f)
            return True
        return False

    def search(self, corpus: dict, queries: dict, *args, **kwargs):
        """
        Returns:
            results: Dict[qid, Dict[doc_id, score]]
            search_time: float
        """
        top_k = self.candidate_k

        # 문서 준비
        docs: List[str] = []
        texts: List[str] = []
        for did, info in corpus.items():
            docs.append(did)
            texts.append(build_doc_text(info))

        # 인덱스 로드/빌드
        if not self._load_index():
            self._build_index(texts)

        # 쿼리 준비
        q_ids = list(queries.keys())
        q_texts = [get_query_text(queries[qid]) for qid in q_ids]

        # 워밍업
        _ = self.model.encode_query("warmup")

        results: Dict[str, Dict[str, float]] = {}
        search_start = time.time()

        with torch.no_grad():
            q_embs = self.model.encode_query(q_texts, batch_size=self.batch_size)
        q_embs = q_embs.to("cpu").to_dense().numpy()

        for i, vec in enumerate(q_embs):
            q_id = q_ids[i]
            nz = np.nonzero(vec)[0]
            scores: Dict[int, float] = defaultdict(float)
            for token_id in nz:
                qw = float(vec[token_id])
                for doc_idx, dw in self.inverted_index.get(token_id, []):
                    scores[doc_idx] += qw * dw

            # 상위 top_k
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            results[q_id] = {docs[doc_idx]: float(score) for doc_idx, score in top}

        search_time = time.time() - search_start
        logger.info(f"[SPLADE] Search time: {search_time:.2f}s (top_k={top_k})")
        return results, search_time


def qwen3_format_queries(query: str, instruction: Optional[str]) -> str:
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"


def qwen3_format_document(document: str) -> str:
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<Document>: {document}{suffix}"


class Reranker:
    def __init__(self, args):
        self.model_name = args.reranker_model
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.batch_size = args.rerank_batch_size
        self.max_length = args.rerank_max_length
        self.engine = args.reranker_engine
        self.qwen3_mode = args.qwen3_mode
        self.qwen3_instruction = args.qwen3_instruction

        if self.engine == "crossencoder":
            logger.info(f"[Rerank] Loading sentence-transformers CrossEncoder on {self.device}: {self.model_name}")
            self.model = STCrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": "float16"}
            )
            self.model.eval()
           
        elif self.engine == "hf":
            logger.info(f"[Rerank] Loading HF transformers model on {self.device}: {self.model_name}")
            use_fp16 = (self.device == "cuda") and args.fp16
            dtype = torch.float16 if use_fp16 else None
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, trust_remote_code=True, torch_dtype=dtype
            ).to(self.device)
            self.model.eval()
        else:
            raise ValueError("--reranker_engine must be one of {'crossencoder', 'hf'}")

    def _build_pairs(self, query_text: str, doc_texts: List[str]) -> List[List[str]]:
        if self.qwen3_mode:
            q = qwen3_format_queries(query_text, self.qwen3_instruction)
            return [[q, qwen3_format_document(dt)] for dt in doc_texts]
        else:
            return [[query_text, dt] for dt in doc_texts]

    @torch.no_grad()
    def score_pairs(self, pairs: List[List[str]]) -> List[float]:
        if self.engine == "crossencoder":
            scores = self.model.predict(pairs, batch_size=self.batch_size)
            return scores.tolist() if hasattr(scores, "tolist") else list(scores)
        else:
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
            return logits.detach().cpu().tolist()

    def rerank(
        self,
        initial_results: Dict[str, Dict[str, float]],
        corpus: Dict[str, Dict],
        queries: Dict[str, Dict]
    ) -> Tuple[Dict[str, Dict[str, float]], float]:
        logger.info("[Rerank] Scoring pairs with cross-encoder / HF...")
        t0 = time.time()
        final_results: Dict[str, Dict[str, float]] = {}

        qtext_cache = {qid: get_query_text(qv) for qid, qv in queries.items()}

        for qid, doc_scores in initial_results.items():
            doc_ids = list(doc_scores.keys())
            doc_texts = [build_doc_text(corpus[did]) for did in doc_ids]
            pairs = self._build_pairs(qtext_cache[qid], doc_texts)

            pair_scores = self.score_pairs(pairs)

            ranked = list(zip(doc_ids, pair_scores))
            ranked.sort(key=lambda x: x[1], reverse=True)
            final_results[qid] = {did: float(score) for did, score in ranked}

        t1 = time.time()
        return final_results, (t1 - t0)


class RerankMTEB(MTEB):
    """SPLADE 후보 + Cross-Encoder/HF rerank 평가 파이프라인"""
    def select_tasks(self, **kwargs):
        super().select_tasks(**kwargs)
        self.tasks = [t for t in self.tasks if isinstance(t, AbsTaskRetrieval)]

    def _run_eval(self, task, reranker, split, output_folder, **kwargs):
        tick = time.time()
        scores = self.evaluate_task(reranker, task, split, output_folder=output_folder, **kwargs)
        tock = time.time()
        return scores, tick, tock

    def evaluate_task(self, reranker: Reranker, task, split="test", **kwargs):
        scores = {}
        subsets = list(task.hf_subsets) if getattr(task, "is_multilingual", False) else ["default"]
        for hf in subsets:
            logger.info(f"[RerankMTEB] Evaluating subset: {hf}")
            corpus = task.corpus[hf][split] if hf != "default" else task.corpus[split]
            queries = task.queries[hf][split] if hf != "default" else task.queries[split]
            qrels = task.relevant_docs[hf][split] if hf != "default" else task.relevant_docs[split]

            res = self._evaluate_subset(
                reranker,
                task,
                corpus,
                queries,
                qrels,
                hf,
                task.metadata.main_score,
                **kwargs,
            )
            scores[hf] = res
        return scores

    def _evaluate_subset(
        self,
        reranker: Reranker,
        task,
        corpus: dict,
        queries: dict,
        qrels: dict,
        hf_subset: str,
        main_score: str,
        k_values=[1, 3, 5, 10],
        **kwargs,
    ):
        # ---- SPLADE 후보 생성기 구성 (subset별 캐시) ----
        splade_model = kwargs.get("splade_model", "telepix/PIXIE-Splade-Preview")
        splade_batch_size = kwargs.get("splade_batch_size", 32)
        index_cache_dir = kwargs.get("index_cache_dir", "./cache")
        candidate_k = kwargs.get("candidate_k", 100)

        os.makedirs(index_cache_dir, exist_ok=True)
        model_name = safe_name(splade_model)
        cache_file = os.path.join(
            index_cache_dir, f"{model_name}_{safe_name(task.metadata.name)}_{safe_name(hf_subset)}_splade_index.pkl"
        )

        splade = SPLADESearch(
            model_name=splade_model,
            index_cache_path=cache_file,
            batch_size=splade_batch_size,
            candidate_k=candidate_k,
        )

        beir_ret = EvaluateRetrieval(retriever=splade)

        # SPLADE 검색 (top_k=candidate_k)
        t0 = time.time()
        initial_results, search_time = beir_ret.retrieve(corpus, queries, top_k=candidate_k)
        t1 = time.time()
        logger.info(f"[RerankMTEB] SPLADE retrieve time: {search_time:.2f}s (overhead {t1 - t0:.2f}s)")

        # Cross-Encoder/HF rerank
        rerank_results, rerank_time = reranker.rerank(initial_results, corpus, queries)

        # 메트릭 계산
        ndcg, _map, recall, precision = beir_ret.evaluate(qrels, rerank_results, k_values)
        mrr = beir_ret.evaluate_custom(qrels, rerank_results, k_values, "mrr")

        scores = {f"ndcg_at_{k.split('@')[-1]}": v for k, v in ndcg.items()}
        scores.update({f"map_at_{k.split('@')[-1]}": v for k, v in _map.items()})
        scores.update({f"recall_at_{k.split('@')[-1]}": v for k, v in recall.items()})
        scores.update({f"precision_at_{k.split('@')[-1]}": v for k, v in precision.items()})
        scores.update({f"mrr_at_{k.split('@')[-1]}": v for k, v in mrr.items()})
        scores["main_score"] = scores.get(main_score, next(iter(scores.values())))
        scores["splade_search_time"] = search_time
        scores["rerank_time"] = rerank_time
        scores["total_time"] = search_time + rerank_time
        scores["candidate_k"] = candidate_k

        if kwargs.get("save_predictions", False):
            out_path = Path(kwargs.get("output_folder", ".")) / f"{task.metadata.name}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rerank_results, f, ensure_ascii=False, indent=2)

        return scores


def main():
    parser = argparse.ArgumentParser(description="SPLADE + (HF | ST CrossEncoder) Reranker on MTEB Retrieval Tasks")

    # 공통
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--task_langs", nargs="+", default=["kor-Kore", "kor-Hang", "kor_Hang", "ko", "korean", "kor-kor"])
    parser.add_argument("--eval_splits", nargs="+", default=None)
    parser.add_argument("--output_folder", type=str, default="./results_rerank")
    parser.add_argument("--verbosity", type=int, default=2)
    parser.add_argument("--save_predictions", action="store_true")

    # SPLADE 후보 생성기
    parser.add_argument("--splade_model", type=str, default="telepix/PIXIE-Splade-Preview")
    parser.add_argument("--splade_batch_size", type=int, default=16)
    parser.add_argument("--index_cache_dir", type=str, default="./cache_rerank")
    parser.add_argument("--candidate_k", type=int, default=100)

    # Reranker
    parser.add_argument("--reranker_model", type=str, default="Alibaba-NLP/gte-multilingual-reranker-base")
    parser.add_argument("--reranker_engine", type=str, choices=["hf", "crossencoder"], default="hf",
                        help="hf: Transformers, crossencoder: sentence-transformers CrossEncoder")
    parser.add_argument("--rerank_batch_size", type=int, default=32)
    parser.add_argument("--rerank_max_length", type=int, default=8192)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")

    # Qwen3 전용 포맷
    parser.add_argument("--qwen3_mode", action="store_true",
                        help="Qwen3 프롬프트 포맷 적용 (<|im_start|>... <Instruct>/<Query>/<Document>)")
    parser.add_argument("--qwen3_instruction", type=str,
                        default="Given a web search query, retrieve relevant passages that answer the query")

    args = parser.parse_args()

    if args.verbosity <= 1:
        logging.getLogger("mteb").setLevel(logging.WARNING)
    elif args.verbosity == 0:
        logging.getLogger("mteb").setLevel(logging.CRITICAL)

    evaluator = RerankMTEB(tasks=args.tasks, task_langs=args.task_langs)
    reranker = Reranker(args)

    splade_name = safe_name(args.splade_model)
    rerank_name = safe_name(args.reranker_model)
    output_folder = f"{args.output_folder}/{splade_name}__{rerank_name}"

    run_kwargs = {
        "model": reranker,
        "output_folder": output_folder,
        "verbosity": args.verbosity,
        "splade_model": args.splade_model,
        "splade_batch_size": args.splade_batch_size,
        "index_cache_dir": args.index_cache_dir,
        "candidate_k": args.candidate_k,
    }
    if args.eval_splits:
        run_kwargs["eval_splits"] = args.eval_splits
    if args.save_predictions:
        run_kwargs["save_predictions"] = True
    
    evaluator.run(**run_kwargs)


if __name__ == "__main__":
    main()
