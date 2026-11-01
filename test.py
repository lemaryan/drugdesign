
import os
from openai import OpenAI

# Check environment variables
api_key = os.getenv("NRP_API_KEY")
base_url = os.getenv("NRP_BASE_URL")
model = os.getenv("LLM_MODEL", "qwen3")

if not api_key:
    raise ValueError("NRP_API_KEY is missing")

if not base_url:
    raise ValueError("NRP_BASE_URL is missing")

print("API key: Found")
print("Base URL:", base_url)
print("Model:", model)

# Initialize client
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=30.0,
    max_retries=2
)

try:
    print("Connecting to NRP API...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Say hello"}
        ],
        max_tokens=50
    )

    print("Connection successful!")
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print("Error type:", type(e).__name__)
    print("Error details:", repr(e))

    print("Underlying cause:", repr(e.__cause__))
    print("Context:", repr(e.__context__))
