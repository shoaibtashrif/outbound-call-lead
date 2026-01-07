import os
import httpx
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def register_tool():
    api_key = os.getenv("ULTRAVOX_API_KEY")
    tool_name = "googleSheetsAppend"
    tool_url = "https://agent.cabex.co.uk/outbound/api/tools/google-sheets/append"
    
    payload = {
        "name": tool_name,
        "definition": {
            "description": "Append data to a Google Sheet.",
            "dynamicParameters": [
                {"name": "spreadsheet_id", "location": "PARAMETER_LOCATION_QUERY", "schema": {"type": "string"}, "required": True},
                {"name": "sheet_name", "location": "PARAMETER_LOCATION_QUERY", "schema": {"type": "string"}, "required": True},
                {"name": "data", "location": "PARAMETER_LOCATION_QUERY", "schema": {"type": "object"}, "required": True}
            ],
            "http": {
                "baseUrlPattern": tool_url,
                "httpMethod": "POST"
            }
        }
    }
    
    async with httpx.AsyncClient() as client:
        print(f"🚀 Registering tool: {tool_name}")
        resp = await client.post("https://api.ultravox.ai/api/tools", headers={"X-API-Key": api_key}, json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")

if __name__ == "__main__":
    asyncio.run(register_tool())
