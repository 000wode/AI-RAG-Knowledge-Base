# 01_first_api.py
# 你的第一个AI程序：调用DeepSeek API

from openai import OpenAI

# API Key 从环境变量读取（运行前先设置 DEEPSEEK_API_KEY）
import os
API_KEY = os.environ.get("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

# 发一条消息给AI
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "用一句话解释什么是人工智能"}
    ]
)

print(response.choices[0].message.content)