# AI-RAG-Knowledge-Base
基于大模型 API 的检索增强生成（RAG）学习项目。
## 项目内容
| 文件 | 功能 |
|------|------|
| 01_first_api.py | 大模型 API 首次调用 |
| 02_chat_with_memory.py | 带上下文记忆的多轮对话 |
| 03_build_vector_store.py | 文档切块 + jieba/TF-IDF 权重向量构建；当前输出为 `chroma_db/chunks.json`，不是接入主链路的 Chroma 查询实现 |
| 04_rag_query.py | TF-IDF + 余弦相似度检索，拼接上下文后生成回答 |
| 05_rag_app.py | 完整的 RAG 问答应用 |
## 技术栈
- Python
- 大模型 API
- jieba + TF-IDF + 余弦相似度
- RAG 架构（检索增强生成）
## 学习路径
1. 理解 LLM API 调用 → 2. 对话记忆 → 3. 向量化 → 4. RAG 问答

## 评测复现

安装依赖：

```bash
python -m pip install jieba openai python-dotenv
```

配置 API Key。脚本只从环境变量或仓库根目录下已被 `.gitignore` 忽略的 `.env` 读取：

```text
DEEPSEEK_API_KEY=<your-key>
```

运行评测：

```bash
python eval/batch_eval.py eval/eval_set.csv eval/results_rerun.csv
python eval/gen_eval_timed.py
```

当前 v2 生成层结果、统计口径和 v1 归档说明见 `eval/README.md`。知识库仅 7 行、2 个文本块，评测结果不代表泛化检索质量。

## 补充材料

- [失败与边界案例](eval/failure_cases.md)
- [Chroma 隔离实验](experiments/chroma_experiment/README.md)：仅验证既有 TF-IDF 向量写入真实 `chromadb.PersistentClient` 的建库、入库、查询和持久化路径；未接入主链路，不代表语义检索能力或泛化质量。
