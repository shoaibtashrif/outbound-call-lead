import os
import httpx
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def list_tools():
    api_key = os.getenv("ULTRAVOX_API_KEY")
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.ultravox.ai/api/tools", headers={"X-API-Key": api_key})
        print(f"Status: {resp.status_code}")
        print(f"Tools: {resp.text}")

if __name__ == "__main__":
    asyncio.run(list_tools())
