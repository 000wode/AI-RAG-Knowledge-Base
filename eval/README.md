# RAG 评测说明

## 当前版本

- 生成层脚本：`gen_eval_timed.py`
- 脚本版本：`gen_eval_timed v2 (2026-09-17)`
- 运行 ID：`20260917-1`
- 运行日期：2026-09-17
- 评测规模：20 题
- 当前结果：20/20 题执行成功，0 个运行错误

## v1 与 v2 差异

| 项目 | v1 | v2 |
|---|---|---|
| 检索判定字段 | `retrieval_hit=YES/NO` | `retrieval_verdict=HIT/FULL_HIT/PARTIAL_HIT/MISS/N/A` |
| 运行时间 | 日志无开始/结束时间戳 | 含开始和结束时间戳 |
| 脚本版本 | 未记录 | 记录脚本版本 |
| 逐题耗时 | 有 | 有 |
| 汇总耗时 | 仅平均/P95/最大 | 平均/最小/P95/最大 |
| 失败信息 | 混在回答文本中 | 独立 `status` 和 `error_message` 字段 |
| 运行标识 | 无独立 `run_id` 字段 | 每题含 `run_id` |
| 敏感信息保护 | 无专门脱敏层 | stdout、日志和 CSV 统一经过 `[REDACTED]` 脱敏 |

v1 旧产物已移至 `eval/archive/`。

## v2 运行结果

| 指标 | 结果 |
|---|---|
| 题目总数 | 20 |
| 成功题数 | 20 |
| 失败题数 | 0 |
| 检索判定 | HIT 14 / FULL_HIT 2 / N/A 4 |
| 检索耗时 | 平均 33.72ms / 最小 0.07ms / P95 0.38ms / 最大 671.48ms |
| 生成耗时 | 平均 695.0ms / 最小 494.5ms / P95 918.0ms / 最大 1648.3ms |

结果文件：

- `eval/results_gen_timed.csv`
- `eval/logs/run_20260917-1_generation.log`

## 安全与边界

- API Key 只从 `DEEPSEEK_API_KEY` 环境变量或已被 `.gitignore` 忽略的 `.env` 读取。
- 脚本会把 Key 和 Authorization 内容脱敏为 `[REDACTED]`。
- 知识库仅 `knowledge_base/ai_intro.txt` 7 行、2 个文本块；上述结果只用于检查评测流程和失败模式，不代表泛化检索质量。

## 复跑

```bash
python -m pip install jieba openai python-dotenv

# 方式一：设置环境变量
# PowerShell: $env:DEEPSEEK_API_KEY="<your-key>"

# 方式二：在仓库根目录创建 .env
# DEEPSEEK_API_KEY=<your-key>

python eval/batch_eval.py eval/eval_set.csv eval/results_rerun.csv
python eval/gen_eval_timed.py
```
