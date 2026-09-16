# -*- coding: utf-8 -*-
"""RAG 生成层评测（含耗时统计）"""
import json, math, csv, os, time
from collections import Counter
import jieba
from openai import OpenAI

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    print("请先设置环境变量 DEEPSEEK_API_KEY")
    exit(1)

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

with open("./chroma_db/chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)
chunks = data["chunks"]
vectors = [{k: float(v) for k, v in vec.items()} for vec in data["vectors"]]

def tokenize(text):
    return list(jieba.cut(text))

def get_query_vector(q):
    tokens = tokenize(q)
    n = len(tokens)
    return {w: c/n for w, c in Counter(tokens).items()}

def cosine(v1, v2):
    words = set(v1) & set(v2)
    if not words:
        return 0
    dot = sum(v1[w]*v2[w] for w in words)
    n1 = math.sqrt(sum(x*x for x in v1.values()))
    n2 = math.sqrt(sum(x*x for x in v2.values()))
    return 0 if n1 == 0 or n2 == 0 else dot/(n1*n2)

def judge_retrieval(expected, top1, ranked):
    """按 eval_methodology.md §1 判定检索命中"""
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

rows = []
with open("./eval/eval_set.csv", "r", encoding="utf-8") as f:
    testset = list(csv.DictReader(f))

for row in testset:
    q = row["question"]

    # ── 检索计时 ──
    t0 = time.perf_counter()
    q_vec = get_query_vector(q)
    scores = [cosine(q_vec, v) for v in vectors]
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    top1 = ranked[0]
    context = chunks[top1]
    t_retrieval = (time.perf_counter() - t0) * 1000   # ms

    verdict = judge_retrieval(row["expected_chunk"], top1 + 1, ranked)

    prompt = f"""基于以下信息回答问题。如果信息不够，直接说不知道。

参考信息：
{context}

问题：{q}

回答："""

    # ── 生成计时 ──
    t1 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"[API错误] {e}"
    t_gen = (time.perf_counter() - t1) * 1000

    print(f"[{row['id']}] 检索 {t_retrieval:.1f}ms | 生成 {t_gen:.0f}ms | {verdict}")

    rows.append({
        "id": row["id"],
        "type": row["type"],
        "question": q,
        "expected_chunk": row["expected_chunk"],
        "top1_chunk": top1 + 1,
        "retrieval_verdict": verdict,
        "retrieval_ms": round(t_retrieval, 2),
        "generation_ms": round(t_gen, 1),
        "context": context,
        "answer": answer,
    })

with open("./eval/results_gen_timed.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ── P95 延迟统计 ──
def p95(vals):
    s = sorted(vals)
    idx = int(len(s) * 0.95) - 1
    return s[max(idx, 0)]

r_times = [r["retrieval_ms"] for r in rows]
g_times = [r["generation_ms"] for r in rows]

print("\n" + "=" * 50)
print("延迟统计（20 题）")
print(f"检索  平均 {sum(r_times)/len(r_times):.1f}ms | P95 {p95(r_times):.1f}ms | 最大 {max(r_times):.1f}ms")
print(f"生成  平均 {sum(g_times)/len(g_times):.0f}ms | P95 {p95(g_times):.0f}ms | 最大 {max(g_times):.0f}ms")
print(f"\n结果已保存到 ./eval/results_gen_timed.csv")