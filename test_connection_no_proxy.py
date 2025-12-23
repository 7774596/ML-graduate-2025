import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import httpx

load_dotenv()

# Unset proxy env vars for this test
if "HTTP_PROXY" in os.environ:
    del os.environ["HTTP_PROXY"]
if "HTTPS_PROXY" in os.environ:
    del os.environ["HTTPS_PROXY"]
if "http_proxy" in os.environ:
    del os.environ["http_proxy"]
if "https_proxy" in os.environ:
    del os.environ["https_proxy"]

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model = os.getenv("MODEL_NAME", "ecnu-max")

print(f"Testing connection to: {base_url}")
print(f"Model: {model}")

async def test():
    try:
        # Explicitly disable proxies in httpx client
        http_client = httpx.AsyncClient(trust_env=False)
        
        client = AsyncOpenAI(
            api_key=api_key, 
            base_url=base_url, 
            timeout=10.0,
            http_client=http_client
        )
        print("Client created with trust_env=False. Sending request...")
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
