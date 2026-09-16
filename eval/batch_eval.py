# -*- coding: utf-8 -*-
"""RAG 检索评测脚本（只评检索，不调 API）

版本: v2 (2026-09-16) — 判定口径对齐 eval_methodology.md §1
判定符号: HIT / FULL_HIT / PARTIAL_HIT / MISS / N/A
"""
import json, math, csv, sys, time
from datetime import datetime
from collections import Counter
import jieba

SCRIPT_VERSION = "batch_eval v2 (2026-09-16)"

# ── 加载向量库 ──
with open("./chroma_db/chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)
chunks = data["chunks"]
vectors = [{k: float(v) for k, v in vec.items()} for vec in data["vectors"]]

print(f"脚本版本: {SCRIPT_VERSION}")
print(f"运行开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"共加载 {len(chunks)} 个文本块\n")

def tokenize(text):
    return list(jieba.cut(text))

def get_query_vector(query):
    tokens = tokenize(query)
    n = len(tokens)
    tf = Counter(tokens)
    return {w: c / n for w, c in tf.items()}

def cosine(vec1, vec2):
    words = set(vec1) & set(vec2)
    if not words:
        return 0
    dot = sum(vec1[w] * vec2[w] for w in words)
    n1 = math.sqrt(sum(v*v for v in vec1.values()))
    n2 = math.sqrt(sum(v*v for v in vec2.values()))
    return 0 if n1 == 0 or n2 == 0 else dot / (n1 * n2)

def judge_retrieval(expected, top1, ranked):
    """按 eval_methodology.md §1 判定检索命中

    - 无答案题 (expected == "none") → N/A
    - 跨块题 (含 "+") → FULL_HIT / PARTIAL_HIT / MISS
    - 单块题 → HIT / MISS
    """
    if expected == "none":
        return "N/A"
    if "+" in expected:
        expected_list = [int(x) for x in expected.split("+")]
        top2 = [i + 1 for i in ranked[:2]]
        if top1 in expected_list:
            others = [e for e in expected_list if e != top1]
            return "FULL_HIT" if all(o in top2 for o in others) else "PARTIAL_HIT"
        return "MISS"
    return "HIT" if str(top1) == expected else "MISS"

# ── 读取测试集 ──
infile = sys.argv[1] if len(sys.argv) > 1 else "./eval/eval_set.csv"
outfile = sys.argv[2] if len(sys.argv) > 2 else "./eval/results.csv"

rows = []
times = []

with open(infile, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        q = row["question"]

        t0 = time.perf_counter()
        q_vec = get_query_vector(q)
        scores = [cosine(q_vec, v) for v in vectors]
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        top1 = ranked[0] + 1
        top1_score = scores[ranked[0]]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

        verdict = judge_retrieval(row["expected_chunk"], top1, ranked)

        print("=" * 60)
        print(f"[{row['id']}] {q}")
        print(f"  预期块: {row['expected_chunk']}")
        print(f"  Top-1 实际块: {top1} (相似度 {top1_score:.4f})")
        print(f"  全部排名: {[(i+1, round(scores[i],4)) for i in ranked]}")
        print(f"  判定: {verdict}")
        print(f"  耗时: {elapsed_ms:.2f} ms")
        print(f"  上下文全文:\n---\n{chunks[top1-1]}\n---")

        rows.append({
            "id": row["id"],
            "type": row["type"],
            "question": q,
            "expected_chunk": row["expected_chunk"],
            "top1_chunk": top1,
            "top1_score": round(top1_score, 4),
            "retrieval_verdict": verdict,
            "retrieval_ms": round(elapsed_ms, 2),
            "context": chunks[top1-1],
        })

with open(outfile, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ── 统计 ──
def p95(vals):
    s = sorted(vals)
    return s[max(int(len(s) * 0.95) - 1, 0)]

print("\n" + "=" * 60)
print("检索耗时统计")
print(f"  平均 {sum(times)/len(times):.2f}ms | 最小 {min(times):.2f}ms | P95 {p95(times):.2f}ms | 最大 {max(times):.2f}ms")
print(f"  说明: 第 1 题含 jieba 词典首次加载（冷启动），不计入稳态统计")

print("\n判定分布")
from collections import Counter as C2
dist = C2(r["retrieval_verdict"] for r in rows)
for k, v in dist.items():
    print(f"  {k}: {v}")

print(f"\n脚本版本: {SCRIPT_VERSION}")
print(f"运行结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"结果已保存到 {outfile}")
