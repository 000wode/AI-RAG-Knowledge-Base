# -*- coding: utf-8 -*-
"""RAG 生成层评测：批量跑 20 题，记录回答"""
import json, math, csv, os
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

rows = []
with open("./eval/eval_set.csv", "r", encoding="utf-8") as f:
    testset = list(csv.DictReader(f))

for row in testset:
    q = row["question"]
    q_vec = get_query_vector(q)
    scores = [cosine(q_vec, v) for v in vectors]
    top1 = scores.index(max(scores))
    context = chunks[top1]

    prompt = f"""基于以下信息回答问题。如果信息不够，直接说不知道。

参考信息：
{context}

问题：{q}

回答："""

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"[API错误] {e}"

    print("=" * 60)
    print(f"[{row['id']}] ({row['type']}) {q}")
    print(f"  检索块: {top1+1}")
    print(f"  回答: {answer}")

    rows.append({
        "id": row["id"],
        "type": row["type"],
        "question": q,
        "expected_chunk": row["expected_chunk"],
        "top1_chunk": top1 + 1,
        "context": context,
        "answer": answer,
    })

with open("./eval/results_gen.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("\n结果已保存到 ./eval/results_gen.csv")