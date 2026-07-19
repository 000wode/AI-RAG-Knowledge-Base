import jieba
import json
import os
import math
from collections import Counter

with open("knowledge_base/ai_intro.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 切分文档
chunks = []
for i in range(0, len(text), 200):
    chunks.append(text[i:i+200])
print("已切分为" + str(len(chunks)) + "个文本块")

# 用jieba分词，计算每个chunk的关键词权重（TF-IDF）
def tokenize(text):
    return list(jieba.cut(text))

# 计算所有文档的词频
chunk_tokens = [tokenize(c) for c in chunks]
all_words = []
for t in chunk_tokens:
    all_words.extend(t)

# IDF计算
idf = {}
total_docs = len(chunks)
for word in set(all_words):
    doc_count = sum(1 for t in chunk_tokens if word in t)
    idf[word] = math.log((total_docs + 1) / (doc_count + 1)) + 1

# 保存向量（TF-IDF向量）
def get_tfidf_vector(tokens):
    word_count = len(tokens)
    tf = Counter(tokens)
    vec = {}
    for word, count in tf.items():
        vec[word] = (count / word_count) * idf.get(word, 1)
    return vec

vectors = [get_tfidf_vector(t) for t in chunk_tokens]
print("TF-IDF向量计算完成")

os.makedirs("./chroma_db", exist_ok=True)
with open("./chroma_db/chunks.json", "w", encoding="utf-8") as f:
    json.dump({"chunks": chunks, "vectors": vectors}, f, ensure_ascii=False)
print("向量库已构建并保存到 ./chroma_db/")