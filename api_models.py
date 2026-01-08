"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class GoogleSheetsAppendRequest(BaseModel):
    spreadsheet_id: Optional[str] = None
    sheet_name: Optional[str] = None
    data: Dict[str, Any]

class CallRequest(BaseModel):
    system_prompt: str
    to_number: str
    tools: Optional[List[Dict[str, Any]]] = None
    from_number: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    server_host: Optional[str] = None
    voice: Optional[str] = "a656a751-b754-4621-b571-e1298cb7e5bb"

class AgentCallRequest(BaseModel):
    agent_id: str
    to_number: str
    from_number: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    server_host: Optional[str] = None

class ToolParameter(BaseModel):
    name: str
    location: str
    schema_data: Dict[str, Any] = Field(alias="schema")
    required: bool

class ToolAuthentication(BaseModel):
    type: str
    header_name: Optional[str] = None
    api_key: Optional[str] = None
    token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

class ToolDefinition(BaseModel):
    name: str
    description: str
    base_url: str
    http_method: str = "POST"
    parameters: Optional[List[Dict[str, Any]]] = None
    authentication: Optional[Dict[str, Any]] = None

class CreateAgentRequest(BaseModel):
    name: str
    system_prompt: str
    voice: Optional[str] = "a656a751-b754-4621-b571-e1298cb7e5bb"
    tool_ids: Optional[List[str]] = None  # List of tool IDs to assign
    language: Optional[str] = "en"

class BatchScheduleRequest(BaseModel):
    system_prompt: str
    to_numbers: List[str]
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    mode: str = "parallel"  # "parallel" or "sequential"

class ScheduleRequest(BaseModel):
    system_prompt: str
    to_numbers: List[str] # List of numbers to call
    window_start: Optional[str] = None # ISO string
    window_end: Optional[str] = None # ISO string
    tools: Optional[List[Dict[str, Any]]] = None
    voice: Optional[str] = "a656a751-b754-4621-b571-e1298cb7e5bb"
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    from_number: Optional[str] = None