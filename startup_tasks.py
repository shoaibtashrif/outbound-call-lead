"""
Startup tasks for the application
"""
import os
import logging
import httpx
from models import Tool
from config import SessionLocal

logger = logging.getLogger(__name__)

async def register_google_sheets_tool():
    """Register Google Sheets Tool if not exists"""
    db = SessionLocal()
    try:
        tool_name = "googleSheetsAppend"
        db_tool = db.query(Tool).filter(Tool.name == tool_name).first()
        if not db_tool:
            # Get server host for webhook
            host = os.getenv("SERVER_HOST")
            if host:
                if not host.startswith("http"): host = f"https://{host}"
                path_prefix = "/outbound"
                if path_prefix not in host:
                    tool_url = f"{host}{path_prefix}/api/tools/google-sheets/append"
                else:
                    tool_url = f"{host}/api/tools/google-sheets/append"
                
                logger.info(f"🛠️ Registering Google Sheets Tool at {tool_url}")
                
                # Register in local DB
                new_tool = Tool(
                    name=tool_name,
                    description="Append data (name, phone, email, etc.) to a Google Sheet. Requires spreadsheet_id and sheet_name.",
                    base_url=tool_url,
                    http_method="POST"
                )
                db.add(new_tool)
                db.commit()
                
                # Register in Ultravox
                api_key = os.getenv("ULTRAVOX_API_KEY")
                payload = {
                    "name": tool_name,
                    "definition": {
                        "description": "Append data to a Google Sheet. Use this to save contact info or lead data.",
                        "dynamicParameters": [
                            {"name": "spreadsheet_id", "location": "body", "schema": {"type": "string"}, "required": True},
                            {"name": "sheet_name", "location": "body", "schema": {"type": "string"}, "required": True},
                            {"name": "data", "location": "body", "schema": {"type": "object"}, "required": True}
                        ],
                        "http": {
                            "baseUrlPattern": tool_url,
                            "httpMethod": "POST"
                        }
                    }
                }
                async with httpx.AsyncClient() as client:
                    resp = await client.post("https://api.ultravox.ai/api/tools", headers={"X-API-Key": api_key}, json=payload)
                    logger.info(f"🛠️ Ultravox Tool Registration Response: {resp.status_code} - {resp.text}")
        else:
            logger.info(f"🛠️ Google Sheets Tool already registered")
    except Exception as e:
        logger.error(f"❌ Error registering tool: {e}")
    finally:
        db.close()