#!/usr/bin/env python3
"""
Verify the new custom tool configuration for Alex agent
"""
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "outbound_agents_v2.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get Alex agent details
cursor.execute("""
    SELECT a.id, a.name, a.transfer_number, t.name as tool_name, t.is_builtin
    FROM agents a 
    LEFT JOIN agent_tools at ON a.id = at.agent_id 
    LEFT JOIN tools t ON at.tool_name = t.name 
    WHERE a.name = 'Alex'
""")

results = cursor.fetchall()
print("Agent: Alex")
print("-" * 50)

agent_id = None
transfer_number = None
tools = []

for row in results:
    agent_id, name, transfer_number, tool_name, is_builtin = row
    if tool_name:
        tools.append({"name": tool_name, "is_builtin": is_builtin})

print(f"ID: {agent_id}")
print(f"Transfer Number: {transfer_number}")
print(f"Tools: {[t['name'] for t in tools]}")
print()

# Simulate the new build_selected_tools logic
def get_transfer_tool(transfer_number, base_url, service_api_key):
    return {
        "temporaryTool": {
            "modelToolName": "transferCall",
            "description": "Transfers call to a human...",
            "staticParameters": [
                {"name": "destinationNumber", "value": transfer_number},
                {"name": "useWhisper", "value": True}
            ],
            "http": {
                "baseUrlPattern": f"{base_url}/api/transfer",
                "httpMethod": "POST"
            }
        }
    }

selected_tools = []
for tool in tools:
    if tool['name'] in ["coldTransfer", "warmTransfer"]:
        continue
    selected_tools.append({"toolName": tool['name']})

if transfer_number:
    host = os.getenv("SERVER_HOST")
    if host:
        if not host.startswith("http"): host = f"https://{host}"
        base_url = f"{host}/outbound"
        service_api_key = os.getenv("SERVICE_API_KEY", "change-this-key")
        transfer_tool = get_transfer_tool(transfer_number, base_url, service_api_key)
        selected_tools.append(transfer_tool)

print("Expected payload for Ultravox (Simplified):")
print("-" * 50)
import json
print(json.dumps(selected_tools, indent=4))

conn.close()
