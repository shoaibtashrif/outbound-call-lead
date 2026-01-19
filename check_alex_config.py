#!/usr/bin/env python3
"""
Quick script to verify the tool configuration for Alex agent
"""
import sqlite3

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
for row in results:
    agent_id, name, transfer_number, tool_name, is_builtin = row
    print(f"ID: {agent_id}")
    print(f"Transfer Number: {transfer_number}")
    print(f"Tool: {tool_name} (builtin: {is_builtin})")
    print()

# Simulate what build_selected_tools would create
if transfer_number:
    transfer_target = transfer_number
    if not transfer_target.startswith("sip:"):
        transfer_target = f"sip:{transfer_target}@sip.twilio.com"
    
    print("Expected payload for Ultravox:")
    print("-" * 50)
    print(f"""
{{
    "selectedTools": [
        {{
            "toolName": "coldTransfer",
            "parameterOverrides": {{
                "target": "{transfer_target}"
            }}
        }},
        {{
            "toolName": "googleSheetsAppend"
        }}
    ]
}}
""")

conn.close()
