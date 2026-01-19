import os
import httpx
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Tool

DATABASE_URL = "sqlite:///./outbound_agents_v2.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

async def check():
    db = SessionLocal()
    tool = db.query(Tool).filter(Tool.name == 'googleSheetsAppend').first()
    if tool:
        print(f"Local DB Tool: {tool.name}, URL: {tool.base_url}")
    else:
        print("Local DB Tool: NOT FOUND")
    db.close()

    api_key = os.getenv("ULTRAVOX_API_KEY")
    if not api_key:
        api_key = "1vCSStI6.8VL2fVYMFN9Pokgcy3kUNReMUEeTQBlB"
    
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.ultravox.ai/api/tools", headers={"X-API-Key": api_key})
        if resp.status_code == 200:
            tools = resp.json().get("results", [])
            found = False
            for t in tools:
                if t['name'] == 'googleSheetsAppend':
                    print(f"Ultravox API Tool: FOUND, URL: {t.get('definition', {}).get('http', {}).get('baseUrlPattern')}")
                    found = True
                    break
            if not found:
                print("Ultravox API Tool: NOT FOUND")
        else:
            print(f"Ultravox API Error: {resp.status_code} {resp.text}")

if __name__ == "__main__":
    asyncio.run(check())
