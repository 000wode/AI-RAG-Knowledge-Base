# 02_chat_with_memory.py
# 带对话记忆的聊天机器人

from openai import OpenAI

import os
API_KEY = os.environ.get("DEEPSEEK_API_KEY")   # 从环境变量读取，切勿硬编码
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

messages = [
    {"role": "system", "content": "你是一个知识渊博的助手，回答简洁准确。"}
]

print("💬 AI助手已启动（输入quit退出）\n")

while True:
    user_input = input("你：")
    if user_input.lower() == "quit":
        break
    
    messages.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    
    reply = response.choices[0].message.content
    print(f"AI：{reply}\n")
    messages.append({"role": "assistant", "content": reply})
    