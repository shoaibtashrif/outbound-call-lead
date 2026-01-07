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
            "description": "Append data to a Google Sheet. Use this to save lead information like name, email, and phone number.",
            "modelToolName": "googleSheetsAppend",
            "dynamicParameters": [
                {
                    "name": "data",
                    "location": "PARAMETER_LOCATION_BODY",
                    "schema": {
                        "type": "object",
                        "description": "The data to append to the sheet. Keys should be column headers (e.g., 'name', 'email', 'phone')."
                    },
                    "required": True
                }
            ],
            "http": {
                "baseUrlPattern": tool_url,
                "httpMethod": "POST"
            }
        }
    }
    
    async with httpx.AsyncClient() as client:
        # Delete first to update
        # We don't have the toolId easily here, but we can just try to overwrite if the API supports it.
        # Actually, Ultravox API usually requires toolId for updates.
        # But we can just create a new one if we want, or find the ID.
        
        # Let's just try to POST again, it might allow it or we might need to delete.
        print(f"🚀 Registering tool: {tool_name}")
        resp = await client.post("https://api.ultravox.ai/api/tools", headers={"X-API-Key": api_key}, json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")

if __name__ == "__main__":
    asyncio.run(register_tool())
