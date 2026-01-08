"""
Main FastAPI application - Outbound Call Service
Organized and refactored for better maintainability
"""
import os
import asyncio
import json
import logging
import sys
from openai import OpenAI
from groq import Groq
import requests
import io
import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException, UploadFile, File, Depends
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from twilio.rest import Client
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import httpx
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session
from models import Base, Agent, Tool, AgentCreate, AgentResponse, User, UserCreate, UserLogin, UserResponse, Call, TwilioNumber, TwilioNumberCreate, TwilioNumberResponse
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from google_sheets_service import google_sheets_service

# Import organized modules
from config import logger, get_db
from auth import get_current_user, verify_password, get_password_hash, create_access_token
from call_monitor import monitor_calls_and_balance
from sms_service import send_call_summary_sms
from api_models import *

app = FastAPI(title="Outbound Call Service")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting up Outbound Call Service...")
    asyncio.create_task(monitor_calls_and_balance())
    
    # Register Google Sheets Tool if not exists
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
    
    logger.info("✅ Outbound Call Service startup complete!")

# Import the database session from config
from config import SessionLocal

# Basic page routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "outbound-call-service"}
# 
--- Auth API ---
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=60 * 24)  # 24 hours
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/register", response_model=UserResponse)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        business_type=user.business_type,
        subscription_type=user.subscription_type,
        hashed_password=hashed_password,
        balance=10.0 # Free trial balance
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.get("/api/check-username")
async def check_username(username: str, db: Session = Depends(get_db)):
    """Check if username is available"""
    user = db.query(User).filter(User.username == username).first()
    return {"available": user is None}

# --- Dashboard API ---
@app.get("/api/dashboard")
async def get_dashboard_data(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    agent_count = db.query(Agent).filter(Agent.user_id == user.id).count()
    
    # Count actual live calls
    live_calls = db.query(Call).filter(
        Call.user_id == user.id, 
        Call.status.in_(["started", "active"])
    ).count()

    # Real graph data - last 7 days
    usage_data = []
    satisfaction_data = []
    for i in range(6, -1, -1):
        target_date = datetime.utcnow() - timedelta(days=i)
        day_label = target_date.strftime("%a")
        date_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)
        
        # Call count
        count = db.query(Call).filter(Call.user_id == user.id, Call.created_at >= date_start, Call.created_at < date_end).count()
        usage_data.append({"day": day_label, "calls": count})
        
        # Satisfaction average
        avg_score = db.query(func.avg(Call.satisfaction_score)).filter(
            Call.user_id == user.id, 
            Call.created_at >= date_start, 
            Call.created_at < date_end,
            Call.satisfaction_score != None
        ).scalar() or 0
        satisfaction_data.append({"day": day_label, "score": round(float(avg_score), 1)})
    
    # Satisfaction distribution for Pie Chart
    successful_calls = db.query(Call).filter(Call.user_id == user.id, Call.satisfaction_score >= 7).count()
    unsuccessful_calls = db.query(Call).filter(Call.user_id == user.id, Call.satisfaction_score < 7, Call.satisfaction_score != None).count()
    neutral_calls = db.query(Call).filter(Call.user_id == user.id, Call.satisfaction_score == None).count()

    return {
        "username": user.username,
        "full_name": user.full_name,
        "email": user.email,
        "business_type": user.business_type,
        "balance": user.balance,
        "agent_count": agent_count,
        "live_calls": live_calls,
        "usage_data": usage_data,
        "satisfaction_data": satisfaction_data,
        "satisfaction_dist": {
            "successful": successful_calls,
            "unsuccessful": unsuccessful_calls,
            "neutral": neutral_calls
        },
        "upgrade_options": [
            {"name": "Starter", "price": "$29/mo", "features": ["5 Agents", "1000 mins"]},
            {"name": "Pro", "price": "$99/mo", "features": ["Unlimited Agents", "5000 mins"]},
            {"name": "Enterprise", "price": "Contact Us", "features": ["Custom Solutions"]}
        ]
    }

# --- Google Sheets Tool Endpoint ---
@app.post("/api/tools/google-sheets/append")
async def google_sheets_append(request: Request, db: Session = Depends(get_db)):
    """Tool endpoint to append data to Google Sheets (via Webhook or Direct API)"""
    body = await request.json()
    headers = request.headers
    # Extract Call ID (Ultravox sends it as X-Call-Id)
    call_id = headers.get("x-call-id") or headers.get("X-Call-Id")
    
    logger.info(f"📥 Received Google Sheets Tool Call. Call ID: {call_id}")
    logger.info(f"📦 Body: {body}")
    
    # Extract data. If 'data' key exists, use it, otherwise use the whole body
    if "data" in body and isinstance(body["data"], dict):
        data = body["data"]
        spreadsheet_id = body.get("spreadsheet_id")
        sheet_name = body.get("sheet_name")
    else:
        data = body
        spreadsheet_id = None
        sheet_name = None

    # 1. Try to find the agent to get configuration
    agent = None
    call_record = None
    
    if call_id:
        call_record = db.query(Call).filter(Call.ultravox_call_id == call_id).first()
        if call_record:
            agent = call_record.agent
            logger.info(f"🔍 Found call record {call_record.id} for agent {agent.name if agent else 'Unknown'}")
    else:
        # Fallback: Try to find the most recent active call
        logger.warning("⚠️ No Call ID found in headers. Trying to find most recent active call...")
        call_record = db.query(Call).filter(Call.status == "started").order_by(Call.created_at.desc()).first()
        if call_record:
            logger.info(f"🔄 Using Agent from recent call: {call_record.id} (Agent: {call_record.agent.name})")
            agent = call_record.agent
            call_id = call_record.ultravox_call_id

    # Save collected data to the call record (for both paths)
    if call_record:
        # Check if this exact data was already saved to prevent duplicates
        existing_data = call_record.collected_data
        new_data_str = json.dumps(data, sort_keys=True)
        
        if existing_data:
            existing_data_str = json.dumps(json.loads(existing_data), sort_keys=True)
            if existing_data_str == new_data_str:
                logger.warning(f"⚠️ Duplicate data detected for Call {call_record.id}, skipping save")
            else:
                # Data is different, update it
                try:
                    call_record.collected_data = new_data_str
                    db.commit()
                    logger.info(f"💾 Updated collected data for Call {call_record.id}")
                    logger.info(f"📋 Data saved: {data}")
                except Exception as e:
                    logger.error(f"❌ Failed to update collected data: {e}")
        else:
            # First time saving data
            try:
                call_record.collected_data = new_data_str
                db.commit()
                logger.info(f"💾 Saved collected data to Call {call_record.id}")
                logger.info(f"📋 Data saved: {data}")
                logger.info(f"🔔 SMS will be sent when call ends to: {call_record.to_number}")
            except Exception as e:
                logger.error(f"❌ Failed to save collected data to DB: {e}")
    else:
        logger.warning("⚠️ No call record found - data cannot be saved for SMS")

    # 2. Priority 1: Use Webhook URL if configured
    webhook_url = agent.google_webhook_url if agent else None
    if webhook_url:
        logger.info(f"🚀 Forwarding data to Google Webhook: {webhook_url}")
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            try:
                # Forward the data dictionary to the Apps Script webhook
                resp = await client.post(webhook_url, json=data, timeout=10.0)
                logger.info(f"✅ Webhook Response ({resp.status_code}): {resp.text}")
                return {"success": True, "message": "Data saved successfully"}
            except Exception as e:
                logger.error(f"❌ Webhook Error: {e}")
                return {"success": True, "message": "Data received"}

    # 3. Priority 2: Use Direct API (requires service-account.json)
    spreadsheet_id = spreadsheet_id or (agent.google_spreadsheet_id if agent else None)
    sheet_name = sheet_name or (agent.google_sheet_name if agent else "Sheet1")
    
    if not spreadsheet_id:
        logger.error("❌ Neither Google Webhook URL nor Spreadsheet ID is configured for this agent")
        raise HTTPException(status_code=400, detail="Neither Google Webhook URL nor Spreadsheet ID is configured for this agent.")

    logger.info(f"📊 Appending to Google Sheets via API: {spreadsheet_id} / {sheet_name}")
    result = google_sheets_service.append_data_dict(spreadsheet_id, sheet_name, data)
    if not result.get("success"):
        logger.error(f"❌ Google Sheets API error: {result.get('error')}")
        raise HTTPException(status_code=500, detail=result.get("error"))
    
    logger.info("✅ Data successfully saved to Google Sheets")
    return result