import os
import httpx
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def delete_and_register():
    api_key = os.getenv("ULTRAVOX_API_KEY")
    tool_id = "b096ea38-3043-4b85-8a52-4ca647cf4cfe"
    tool_name = "googleSheetsAppend"
    tool_url = "https://agent.cabex.co.uk/outbound/api/tools/google-sheets/append"
    
    async with httpx.AsyncClient() as client:
        # Delete
        print(f"🗑️ Deleting tool: {tool_id}")
        await client.delete(f"https://api.ultravox.ai/api/tools/{tool_id}", headers={"X-API-Key": api_key})
        
        # Register
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
        print(f"🚀 Registering tool: {tool_name}")
        resp = await client.post("https://api.ultravox.ai/api/tools", headers={"X-API-Key": api_key}, json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")

if __name__ == "__main__":
    asyncio.run(delete_and_register())
