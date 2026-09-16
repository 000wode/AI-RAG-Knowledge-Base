# -*- coding: utf-8 -*-
"""RAG 生成层评测 v2（含耗时统计、完整日志与敏感信息脱敏）"""
import json, math, csv, os, re, sys, time
from collections import Counter
from datetime import datetime
from pathlib import Path

import jieba
from openai import OpenAI

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

SCRIPT_VERSION = "gen_eval_timed v2 (2026-09-17)"
LOG_DIR = Path("./eval/logs")
OUTPUT_FILE = Path("./eval/results_gen_timed.csv")


def redact(text):
    """Remove API keys and authorization headers from all persisted output."""
    value = str(text)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        value = value.replace(api_key, "[REDACTED]")
    value = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
        r"\1[REDACTED]",
        value,
    )
    value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "sk-[REDACTED]", value)
    return value


class Tee:
    """Write the same sanitized output to stdout and the run log."""

    def __init__(self, log_file, console):
        self.log_file = log_file
        self.console = console

    def write(self, text):
        safe = redact(text)
        self.console.write(safe)
        self.log_file.write(safe)
        self.log_file.flush()

    def flush(self):
        self.console.flush()
        self.log_file.flush()


def next_run_id():
    date = datetime.now().strftime("%Y%m%d")
    existing = [
        p.name for p in LOG_DIR.glob(f"run_{date}-*_generation.log")
    ]
    serial = len(existing) + 1
    return f"{date}-{serial}"


def p95(values):
    ordered = sorted(values)
    index = int(len(ordered) * 0.95) - 1
    return ordered[max(index, 0)]


API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    print("请先设置环境变量 DEEPSEEK_API_KEY")
    raise SystemExit(1)

LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = next_run_id()
LOG_FILE = LOG_DIR / f"run_{RUN_ID}_generation.log"

with LOG_FILE.open("w", encoding="utf-8", newline="\n") as log_handle:
    original_stdout = sys.stdout
    sys.stdout = Tee(log_handle, original_stdout)
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

        with open("./chroma_db/chunks.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = data["chunks"]
        vectors = [
            {k: float(v) for k, v in vec.items()} for vec in data["vectors"]
        ]

        def tokenize(text):
            return list(jieba.cut(text))

        def get_query_vector(query):
            tokens = tokenize(query)
            count = len(tokens)
            return {word: n / count for word, n in Counter(tokens).items()}

        def cosine(vec1, vec2):
            words = set(vec1) & set(vec2)
            if not words:
                return 0
            dot = sum(vec1[word] * vec2[word] for word in words)
            norm1 = math.sqrt(sum(value * value for value in vec1.values()))
            norm2 = math.sqrt(sum(value * value for value in vec2.values()))
            return 0 if norm1 == 0 or norm2 == 0 else dot / (norm1 * norm2)

        def judge_retrieval(expected, top1, ranked):
            if expected == "none":
                return "N/A"
            if "+" in expected:
                expected_list = [int(value) for value in expected.split("+")]
                top2 = [index + 1 for index in ranked[:2]]
                if top1 in expected_list:
                    others = [value for value in expected_list if value != top1]
                    return (
                        "FULL_HIT"
                        if all(value in top2 for value in others)
                        else "PARTIAL_HIT"
                    )
                return "MISS"
            return "HIT" if str(top1) == expected else "MISS"

        rows = []
        with open("./eval/eval_set.csv", "r", encoding="utf-8") as f:
            testset = list(csv.DictReader(f))

        print(f"脚本版本: {SCRIPT_VERSION}")
        print(f"运行 ID: {RUN_ID}")
        print(f"运行开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"评测题数: {len(testset)}")
        print("边界声明: 知识库仅 7 行、2 块，结果不代表泛化检索质量。")
        print()

        for row in testset:
            question = row["question"]

            t0 = time.perf_counter()
            q_vec = get_query_vector(question)
            scores = [cosine(q_vec, vector) for vector in vectors]
            ranked = sorted(range(len(scores)), key=lambda index: -scores[index])
            top1 = ranked[0]
            context = chunks[top1]
            retrieval_ms = (time.perf_counter() - t0) * 1000
            verdict = judge_retrieval(
                row["expected_chunk"], top1 + 1, ranked
            )

            prompt = f"""基于以下信息回答问题。如果信息不够，直接说不知道。

参考信息：
{context}

问题：{question}

回答："""

            t1 = time.perf_counter()
            status = "success"
            error_message = ""
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                answer = response.choices[0].message.content.strip()
            except Exception as exc:
                status = "error"
                error_message = redact(str(exc))
                answer = ""
            generation_ms = (time.perf_counter() - t1) * 1000

            print(
                f"[{row['id']}] {status} | 检索 {retrieval_ms:.2f}ms | "
                f"生成 {generation_ms:.1f}ms | {verdict}"
            )
            print(f"  问题: {question}")
            print(f"  回答: {redact(answer) if answer else '[ERROR]'}")
            if error_message:
                print(f"  错误: {error_message}")

            rows.append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "question": question,
                    "expected_chunk": row["expected_chunk"],
                    "top1_chunk": top1 + 1,
                    "retrieval_verdict": verdict,
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": round(generation_ms, 1),
                    "context": context,
                    "status": status,
                    "answer": redact(answer),
                    "error_message": error_message,
                    "run_id": RUN_ID,
                }
            )

        with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(rows[0].keys()), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)

        retrieval_times = [row["retrieval_ms"] for row in rows]
        generation_times = [row["generation_ms"] for row in rows]

        print("\n" + "=" * 50)
        print("延迟统计（20 题）")
        print(
            f"检索  平均 {sum(retrieval_times) / len(retrieval_times):.2f}ms | "
            f"最小 {min(retrieval_times):.2f}ms | "
            f"P95 {p95(retrieval_times):.2f}ms | "
            f"最大 {max(retrieval_times):.2f}ms"
        )
        print(
            f"生成  平均 {sum(generation_times) / len(generation_times):.1f}ms | "
            f"最小 {min(generation_times):.1f}ms | "
            f"P95 {p95(generation_times):.1f}ms | "
            f"最大 {max(generation_times):.1f}ms"
        )
        print(f"\n结果已保存到 {OUTPUT_FILE}")
        print(f"日志已保存到 {LOG_FILE}")
        print(f"运行结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as exc:
        print(f"\n运行失败: {redact(exc)}")
        print(f"运行结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        raise
    finally:
        sys.stdout = original_stdout
