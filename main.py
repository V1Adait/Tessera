import os

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],  # never hardcode the key
)

response = client.chat.completions.create(
    model="openrouter/free",  # OpenRouter picks a currently available free model
    max_tokens=20,
    messages=[
        {"role": "user", "content": "In two sentences, what is an AI agent?"}
    ],
)

choice = response.choices[0]
print("model used:", response.model)
print(choice.message.content)
print("finish_reason:", choice.finish_reason)
if response.usage:
    print("tokens:", response.usage.prompt_tokens, "in,", response.usage.completion_tokens, "out")