# -*- coding: utf-8 -*-
"""RAG 检索评测脚本（只评检索，不调 API）"""
import json, math, csv, sys
from collections import Counter
import jieba

# ── 加载向量库 ──
with open("./chroma_db/chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)
chunks = data["chunks"]
vectors = [{k: float(v) for k, v in vec.items()} for vec in data["vectors"]]

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

# ── 读取测试集 ──
infile = sys.argv[1] if len(sys.argv) > 1 else "./eval/eval_set.csv"
outfile = sys.argv[2] if len(sys.argv) > 2 else "./eval/results.csv"

rows = []
with open(infile, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        q = row["question"]
        q_vec = get_query_vector(q)
        scores = [cosine(q_vec, v) for v in vectors]
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        top1 = ranked[0] + 1
        top1_score = scores[ranked[0]]

        hit = "YES" if str(top1) in row["expected_chunk"] else "NO"

        print("=" * 60)
        print(f"[{row['id']}] {q}")
        print(f"  预期块: {row['expected_chunk']}")
        print(f"  Top-1 实际块: {top1} (相似度 {top1_score:.4f})")
        print(f"  全部排名: {[(i+1, round(scores[i],4)) for i in ranked]}")
        print(f"  命中: {hit}")
        print(f"  上下文全文:\n---\n{chunks[top1-1]}\n---")

        rows.append({
            "id": row["id"],
            "type": row["type"],
            "question": q,
            "expected_chunk": row["expected_chunk"],
            "top1_chunk": top1,
            "top1_score": round(top1_score, 4),
            "hit": hit,
            "context": chunks[top1-1],
        })

with open(outfile, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\n结果已保存到 {outfile}")