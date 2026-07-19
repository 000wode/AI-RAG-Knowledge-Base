# 01_first_api.py
# 你的第一个AI程序：调用DeepSeek API

from openai import OpenAI

# 把你的API Key填在这里（正式发到GitHub前记得删掉这个Key）
API_KEY = "***REMOVED***"

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