"""
Call management routes - agents, calls, tools, etc.
This file contains the core business logic routes
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import Response
from sqlalchemy import func
from sqlalchemy.orm import Session
import httpx
import pandas as pd
import io
from twilio.rest import Client

from models import (User, Agent, Tool, Call, TwilioNumber, 
                   AgentCreate, CallRequest, AgentCallRequest, 
                   ToolDefinition, ScheduleRequest)
from auth import get_current_user
from config import get_db
from google_sheets_service import google_sheets_service

router = APIRouter()
logger = logging.getLogger(__name__)

# Agent management routes
@router.get("/api/agents")
async def list_agents(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    agents = db.query(Agent).filter(Agent.user_id == user.id).all()
    results = []
    for agent in agents:
        results.append({
            "agentId": agent.ultravox_agent_id or str(agent.id),
            "id": str(agent.id),
            "name": agent.name,
            "systemPrompt": agent.system_prompt,
            "model": agent.model,
            "voice": agent.voice,
            "languageHint": agent.language,
            "twilio_number_id": agent.twilio_number_id,
            "twilio_phone_number": agent.twilio_number.phone_number if agent.twilio_number else None,
            "google_spreadsheet_id": agent.google_spreadsheet_id,
            "google_sheet_name": agent.google_sheet_name,
            "google_webhook_url": agent.google_webhook_url,
            "selectedTools": [{"toolName": t.name} for t in agent.tools],
            "created": agent.created_at.isoformat() if agent.created_at else None
        })
    return {"results": results}