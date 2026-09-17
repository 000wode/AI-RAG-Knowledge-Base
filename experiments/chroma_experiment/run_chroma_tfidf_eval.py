# -*- coding: utf-8 -*-
"""Isolated Chroma experiment using the existing TF-IDF vectors.

This script reads the existing chunks.json and eval set, stores explicit
TF-IDF dense vectors in a real Chroma collection, and reruns the existing
20-question retrieval checks. It does not import or modify the main RAG path.
"""

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import chromadb
import jieba
import numpy
import sklearn

SCRIPT_VERSION = "chroma_tfidf_experiment v1 (2026-09-17)"


def configure_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def tokenize(text: str) -> list[str]:
    return list(jieba.cut(text))


def dense_vector(sparse: dict[str, float], vocabulary: dict[str, int]) -> list[float]:
    vector = [0.0] * len(vocabulary)
    for token, value in sparse.items():
        index = vocabulary.get(token)
        if index is not None:
            vector[index] = float(value)
    return vector


def judge_retrieval(expected: str, top1: int, ranked: list[int]) -> str:
    if expected == "none":
        return "N/A"
    if "+" in expected:
        expected_blocks = [int(value) for value in expected.split("+")]
        top2 = ranked[:2]
        if top1 in expected_blocks:
            remaining = [value for value in expected_blocks if value != top1]
            return "FULL_HIT" if all(value in top2 for value in remaining) else "PARTIAL_HIT"
        return "MISS"
    return "HIT" if str(top1) == expected else "MISS"


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(int(len(ordered) * 0.95) - 1, 0)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-json", required=True, type=Path)
    parser.add_argument("--eval-set", required=True, type=Path)
    parser.add_argument("--baseline-results", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = args.outdir / f"run_{run_id}_chroma_tfidf.log"
    configure_logging(log_path)

    started_at = datetime.now().isoformat(timespec="seconds")
    logging.info("脚本版本: %s", SCRIPT_VERSION)
    logging.info("运行 ID: %s", run_id)
    logging.info("开始时间: %s", started_at)
    logging.info("Python: %s", sys.version.split()[0])
    logging.info("chromadb: %s", chromadb.__version__)
    logging.info("jieba: %s", jieba.__version__)
    logging.info("numpy: %s", numpy.__version__)
    logging.info("scikit-learn: %s", sklearn.__version__)

    with args.chunks_json.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    chunks = source["chunks"]
    chunk_sparse_vectors = source["vectors"]

    if len(chunks) != 2:
        raise RuntimeError(f"预期 2 个文本块，实际为 {len(chunks)}")
    if len(chunk_sparse_vectors) != len(chunks):
        raise RuntimeError("文本块数量与向量数量不一致")

    with args.eval_set.open("r", encoding="utf-8", newline="") as handle:
        eval_rows = list(csv.DictReader(handle))
    if len(eval_rows) != 20:
        raise RuntimeError(f"预期 20 道评测题，实际为 {len(eval_rows)}")

    query_tokens = {row["id"]: tokenize(row["question"]) for row in eval_rows}
    vocabulary_tokens = set()
    for vector in chunk_sparse_vectors:
        vocabulary_tokens.update(vector.keys())
    for tokens in query_tokens.values():
        vocabulary_tokens.update(tokens)
    vocabulary = {token: index for index, token in enumerate(sorted(vocabulary_tokens))}
    logging.info("共享词表维度: %s", len(vocabulary))

    dense_chunks = [dense_vector(vector, vocabulary) for vector in chunk_sparse_vectors]
    chroma_path = args.outdir / f"chroma_store_{run_id}"
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.create_collection(
        name=f"rag_tfidf_{run_id.replace('-', '_')}",
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    collection.add(
        ids=[f"chunk_{index + 1}" for index in range(len(chunks))],
        documents=chunks,
        embeddings=dense_chunks,
        metadatas=[{"chunk_index": index + 1} for index in range(len(chunks))],
    )
    logging.info("Chroma 持久化目录: %s", chroma_path)
    logging.info("Chroma collection 数量: %s", collection.count())

    results = []
    elapsed_values = []
    for row in eval_rows:
        tokens = query_tokens[row["id"]]
        query_counter = Counter(tokens)
        query_vector = [float(query_counter[token]) for token in vocabulary]

        started = time.perf_counter()
        response = collection.query(
            query_embeddings=[query_vector],
            n_results=len(chunks),
            include=["distances", "documents", "metadatas"],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        elapsed_values.append(elapsed_ms)

        ids = response["ids"][0]
        distances = response["distances"][0]
        documents = response["documents"][0]
        ranked = [int(item.rsplit("_", 1)[1]) - 1 for item in ids]
        similarity = 1.0 - float(distances[0])
        top1 = ranked[0] + 1
        verdict = judge_retrieval(row["expected_chunk"], top1, [index + 1 for index in ranked])

        logging.info("=" * 60)
        logging.info("[%s] %s", row["id"], row["question"])
        logging.info("预期块: %s", row["expected_chunk"])
        logging.info("Chroma 排名: %s", [index + 1 for index in ranked])
        logging.info("Top-1: %s | cosine similarity: %.4f", top1, similarity)
        logging.info("判定: %s | 耗时: %.2f ms", verdict, elapsed_ms)

        results.append(
            {
                "id": row["id"],
                "type": row["type"],
                "question": row["question"],
                "expected_chunk": row["expected_chunk"],
                "top1_chunk": top1,
                "top1_score": round(similarity, 4),
                "retrieval_verdict": verdict,
                "retrieval_ms": round(elapsed_ms, 2),
                "context": documents[0],
            }
        )

    results_path = args.outdir / f"results_{run_id}_chroma_tfidf.csv"
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with args.baseline_results.open("r", encoding="utf-8", newline="") as handle:
        baseline_rows = {row["id"]: row for row in csv.DictReader(handle)}

    baseline_differences = []
    for row in results:
        baseline = baseline_rows[row["id"]]
        baseline_top1 = int(baseline["top1_chunk"])
        baseline_verdict = baseline["retrieval_verdict"]
        if row["top1_chunk"] != baseline_top1 or row["retrieval_verdict"] != baseline_verdict:
            baseline_differences.append(
                {
                    "id": row["id"],
                    "baseline_top1": baseline_top1,
                    "chroma_top1": row["top1_chunk"],
                    "baseline_verdict": baseline_verdict,
                    "chroma_verdict": row["retrieval_verdict"],
                }
            )

    verdict_distribution = Counter(row["retrieval_verdict"] for row in results)
    summary = {
        "status": "completed",
        "script_version": SCRIPT_VERSION,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "chromadb_version": chromadb.__version__,
        "jieba_version": jieba.__version__,
        "numpy_version": numpy.__version__,
        "sklearn_version": sklearn.__version__,
        "chunks": len(chunks),
        "vocabulary_size": len(vocabulary),
        "questions": len(eval_rows),
        "chroma_collection_count": collection.count(),
        "verdict_distribution": dict(verdict_distribution),
        "retrieval_ms": {
            "min": round(min(elapsed_values), 2),
            "mean": round(statistics.mean(elapsed_values), 2),
            "p95": round(percentile_95(elapsed_values), 2),
            "max": round(max(elapsed_values), 2),
        },
        "baseline_match_count": len(results) - len(baseline_differences),
        "baseline_differences": baseline_differences,
        "results_csv": str(results_path),
        "log_file": str(log_path),
        "chroma_store": str(chroma_path),
        "boundary": "知识库仅 7 行、2 个文本块；结果为隔离实验的流程检查，不代表泛化检索质量。",
    }
    summary_path = args.outdir / f"summary_{run_id}_chroma_tfidf.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    logging.info("=" * 60)
    logging.info("判定分布: %s", dict(verdict_distribution))
    logging.info(
        "耗时: 最小 %.2fms | 平均 %.2fms | P95 %.2fms | 最大 %.2fms",
        summary["retrieval_ms"]["min"],
        summary["retrieval_ms"]["mean"],
        summary["retrieval_ms"]["p95"],
        summary["retrieval_ms"]["max"],
    )
    logging.info("与现有 TF-IDF 基线一致题数: %s / %s", summary["baseline_match_count"], len(results))
    logging.info("差异: %s", json.dumps(baseline_differences, ensure_ascii=False))
    logging.info("结果 CSV: %s", results_path)
    logging.info("摘要 JSON: %s", summary_path)
    logging.info("结束时间: %s", summary["finished_at"])
    logging.info("边界: %s", summary["boundary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
