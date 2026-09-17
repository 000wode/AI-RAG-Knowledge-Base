# Chroma 隔离实验

实验性实现：将既有 TF-IDF 向量写入 chromadb，验证建库/入库/查询/持久化路径；未接入主链路，不带来语义检索能力；知识库 7 行、2 块，结果不代表泛化质量。

## 范围

- 使用现有 `chroma_db/chunks.json` 中的 2 个文本块和 TF-IDF 稀疏向量。
- 将稀疏向量展开到共享词表，以显式 embedding 写入真实 `chromadb.PersistentClient` collection。
- 使用同一份 20 题评测集和现有 `HIT / FULL_HIT / PARTIAL_HIT / MISS / N/A` 判定规则查询并统计。
- 未调用第三方 Embedding API，未下载本地大模型，未修改现有 TF-IDF 主链路。

## 环境

| 组件 | 版本 |
|---|---|
| Python | 3.14.6 |
| chromadb | 1.5.9 |
| jieba | 0.42.1 |
| numpy | 2.5.1 |
| scikit-learn | 1.9.0 |

## 安装依赖

```powershell
python -m pip install chromadb==1.5.9 jieba==0.42.1 numpy==2.5.1 scikit-learn==1.9.0
```

## 复跑命令

在仓库根目录运行：

```powershell
python experiments/chroma_experiment/run_chroma_tfidf_eval.py `
  --chunks-json "chroma_db/chunks.json" `
  --eval-set "eval/eval_set.csv" `
  --baseline-results "eval/results_batch2.csv" `
  --outdir "experiments/chroma_experiment"
```

脚本会在输出目录重建 `chroma_store_<run_id>/`、运行日志、结果 CSV 和摘要 JSON。持久化 store 未纳入仓库。

## 已归档运行

- 运行 ID：`20260917-102408`
- 日志：`run_20260917-102408_chroma_tfidf.log`
- 结果：`results_20260917-102408_chroma_tfidf.csv`
- 摘要：`summary_20260917-102408_chroma_tfidf.json`

## 已归档结果

- Chroma collection 数量：2。
- 共享词表维度：108。
- 20 题判定分布：`HIT 14 / FULL_HIT 2 / N/A 4`。
- 与现有 TF-IDF 基线的 Top-1 块和判定一致：20/20。
- 查询耗时：最小 0.90 ms、平均 1.37 ms、P95 1.81 ms、最大 2.11 ms。

## 边界

知识库仅 7 行、2 个文本块；本实验只证明真实 Chroma 建库、入库和查询流程可以跑通，并使用既有 TF-IDF 向量完成 20 题流程检查。未接入主链路，不代表语义检索能力、泛化检索质量或长期性能。
