from openai import OpenAI
import json
import math
from collections import Counter
import jieba

import os
API_KEY = os.environ.get("DEEPSEEK_API_KEY")   # 从环境变量读取，切勿硬编码
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# 加载向量库
with open("./chroma_db/chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)
chunks = data["chunks"]
vectors = [{k: float(v) for k, v in vec.items()} for vec in data["vectors"]]

def tokenize(text):
    return list(jieba.cut(text))

def get_query_vector(query):
    tokens = tokenize(query)
    word_count = len(tokens)
    tf = Counter(tokens)
    vec = {}
    for word, count in tf.items():
        vec[word] = count / word_count
    return vec

def cosine_similarity(vec1, vec2):
    words = set(vec1.keys()) & set(vec2.keys())
    if not words:
        return 0
    dot = sum(vec1[w] * vec2[w] for w in words)
    norm1 = math.sqrt(sum(v*v for v in vec1.values()))
    norm2 = math.sqrt(sum(v*v for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

print("RAG知识库已启动（输入quit退出）\n")

while True:
    question = input("问题：")
    if question.lower() == "quit":
        break
    
    q_vec = get_query_vector(question)
    scores = [cosine_similarity(q_vec, v) for v in vectors]
    best_idx = scores.index(max(scores))
    context = chunks[best_idx]
    
    prompt = f"""基于以下信息回答问题。如果信息不够，直接说不知道。

参考信息：
{context}

问题：{question}

回答："""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    answer = response.choices[0].message.content
    print(f"回答：{answer}\n")