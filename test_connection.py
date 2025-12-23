import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model = os.getenv("MODEL_NAME", "ecnu-max")

print(f"Testing connection to: {base_url}")
print(f"Model: {model}")
print(f"API Key: {api_key[:5]}...")

async def test():
    try:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=10.0)
        print("Client created. Sending request...")
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("Response received:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
