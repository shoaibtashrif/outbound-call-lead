import os
import asyncio
import json
import logging
import sys
from urllib.parse import unquote
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
from models import Base, Agent, Tool, AgentCreate, AgentResponse, User, UserCreate, UserLogin, UserResponse, Call, TwilioNumber, TwilioNumberCreate, TwilioNumberResponse, SMS, SMSResponse
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from google_sheets_service import google_sheets_service


load_dotenv()

# Configure logging for real-time monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs are flushed immediately
for handler in logger.handlers:
    handler.setLevel(logging.INFO)
    if isinstance(handler, logging.FileHandler):
        handler.flush()

logger.info("🚀 Starting Outbound Call Service with enhanced SMS logging...")

# Database setup
DATABASE_URL = "sqlite:///./outbound_agents_v2.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(title="Outbound Call Service")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

async def monitor_calls_and_balance():
    """Background task to monitor active calls and deduct balance every minute"""
    logger.info("📊 Starting call monitoring and SMS service...")
    while True:
        await asyncio.sleep(60)
        db = SessionLocal()
        try:
            # Find active calls in our DB
            active_calls = db.query(Call).filter(Call.status.in_(["started", "active"])).all()
            api_key = os.getenv("ULTRAVOX_API_KEY")
            
            logger.info(f"🔍 Monitoring {len(active_calls)} active calls...")
            
            for call in active_calls:
                user = db.query(User).filter(User.id == call.user_id).first()
                if not user: 
                    logger.warning(f"⚠️ No user found for call {call.id}")
                    continue
                
                # Check actual status from Ultravox first
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(f"https://api.ultravox.ai/api/calls/{call.ultravox_call_id}", headers={"X-API-Key": api_key})
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("ended"):
                                logger.info(f"📞 Call {call.ultravox_call_id} has ended, processing...")
                                call.status = "ended"
                                if data.get("billedDuration"):
                                    dur = data.get("billedDuration")
                                    if isinstance(dur, str): dur = float(dur.replace("s", ""))
                                    call.duration = int(dur)
                                db.commit()
                                
                                # Check if call is missed and send qualifying SMS
                                await check_and_handle_missed_call(call, data, db, logger)
                                
                                # Send SMS if data collected and not sent yet
                                await send_call_summary_sms(call, db, logger)
                                continue
                except Exception as e:
                    logger.error(f"❌ Error checking call status for {call.ultravox_call_id}: {e}")

                # If still active, deduct cost
                cost_per_min = 0.05
                user.balance -= cost_per_min
                logger.info(f"💰 Deducted ${cost_per_min} from user {user.username}, balance: ${user.balance:.2f}")
                
                if user.balance <= 0:
                    user.balance = 0
                    logger.warning(f"⚠️ User {user.username} balance exhausted, terminating call {call.ultravox_call_id}")
                    # Terminate call
                    try:
                        async with httpx.AsyncClient() as client:
                            await client.post(f"https://api.ultravox.ai/api/calls/{call.ultravox_call_id}/end", headers={"X-API-Key": api_key})
                        call.status = "ended"
                        logger.info(f"✅ Call {call.ultravox_call_id} terminated due to insufficient balance")
                    except Exception as e:
                        logger.error(f"❌ Error terminating call {call.ultravox_call_id}: {e}")
                
                db.commit()
        except Exception as e:
            logger.error(f"❌ Monitor error: {e}")
        finally:
            db.close()

async def check_and_handle_missed_call(call: Call, ultravox_data: dict, db: Session, logger):
    """Check if call is missed and send qualifying SMS"""
    try:
        # Check duration
        duration = call.duration or 0
        is_short_call = duration < 7
        
        # Check if user spoke (has user messages)
        has_user_input = False
        try:
            api_key = os.getenv("ULTRAVOX_API_KEY")
            async with httpx.AsyncClient() as client:
                msg_resp = await client.get(
                    f"https://api.ultravox.ai/api/calls/{call.ultravox_call_id}/messages",
                    headers={"X-API-Key": api_key}
                )
                if msg_resp.status_code == 200:
                    messages = msg_resp.json().get("results", [])
                    # Check if there are any messages from user
                    has_user_input = any(msg.get('role') == 'user' and msg.get('text', '').strip() for msg in messages)
        except Exception as e:
            logger.warning(f"⚠️ Could not check user input for call {call.ultravox_call_id}: {e}")
        
        # Call is missed if duration < 7 seconds OR no user input
        is_missed_call = is_short_call or not has_user_input
        
        if is_missed_call:
            logger.info(f"📞 Call {call.ultravox_call_id} is MISSED (duration: {duration}s, user_input: {has_user_input})")
            
            # Get agent first
            agent = None
            if call.agent_id:
                agent = db.query(Agent).filter(Agent.id == call.agent_id).first()
            
            # For missed calls, use same logic as call summary SMS
            # Get Twilio credentials (same as send_call_summary_sms)
            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            sender_number = call.to_number  # Same as call summary SMS
            
            if not all([account_sid, auth_token, sender_number]):
                logger.error("❌ Missing Twilio credentials in environment variables")
                return
            
            # Recipient is from_number (same as call summary SMS)
            recipient = call.from_number
            
            if not recipient:
                logger.warning(f"⚠️ No recipient number for missed call {call.id}")
                return
            
            # Qualifying questions SMS
            qualifying_questions = (
                "Hi! We tried reaching you but couldn't connect. "
                "Would you be interested in learning more about our services? "
                "Please reply YES if interested, or let us know a better time to call."
            )
            
            logger.info(f"📨 Sending missed call qualifying SMS from {sender_number} to {recipient}")
            logger.info(f"📝 SMS Content: {qualifying_questions}")
            
            # Send SMS via Twilio (same way as call summary SMS)
            try:
                twilio_client = Client(account_sid, auth_token)
                message = twilio_client.messages.create(
                    body=qualifying_questions,
                    from_=sender_number,
                    to=recipient
                )
                
                logger.info(f"✅ Missed call SMS sent successfully!")
                logger.info(f"📨 Message SID: {message.sid}")
                logger.info(f"📱 Status: {message.status}")
                logger.info(f"📞 From: {message.from_}")
                logger.info(f"📞 To: {message.to}")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Error sending missed call SMS: {error_msg}")
                # Check for specific Twilio errors
                if "Permission to send" in error_msg or "not enabled for the region" in error_msg:
                    logger.warning(f"⚠️ SMS not enabled for region {recipient}")
                elif "not SMS-capable" in error_msg:
                    logger.warning(f"⚠️ Sender number {sender_number} is not SMS-capable")
                elif "Invalid 'To' phone number" in error_msg:
                    logger.warning(f"⚠️ Invalid recipient number {recipient}")
                else:
                    logger.error(f"❌ Unhandled SMS error: {error_msg}")
                # Don't save to DB if SMS failed to send
                return
            
            # Save outbound SMS to database (only if SMS was sent successfully)
            sms_record = SMS(
                agent_id=call.agent_id,
                from_number=sender_number,
                to_number=recipient,
                body=qualifying_questions,
                direction="outbound",
                message_sid=message.sid
            )
            db.add(sms_record)
            db.commit()
            db.refresh(sms_record)
            
            logger.info(f"✅ Outbound SMS saved to database with ID: {sms_record.id}")
            
            # Note: Missed call SMS are NOT saved to Google Sheets per user requirement
        else:
            logger.info(f"✅ Call {call.ultravox_call_id} is NOT missed (duration: {duration}s, user_input: {has_user_input})")
            
    except Exception as e:
        logger.error(f"❌ Error handling missed call for {call.ultravox_call_id}: {e}", exc_info=True)

async def send_call_summary_sms(call: Call, db: Session, logger):
    """Send SMS with collected call data"""
    if not call.collected_data or call.sms_sent or not call.to_number:
        if not call.collected_data:
            logger.info(f"📝 No data collected for call {call.id}, skipping SMS")
        elif call.sms_sent:
            logger.info(f"📨 SMS already sent for call {call.id}")
        elif not call.to_number:
            logger.warning(f"⚠️ No recipient number for call {call.id}")
        return

    try:
        # Get Twilio credentials
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        sender_number = call.to_number
        
        if not all([account_sid, auth_token, sender_number]):
            logger.error("❌ Missing Twilio credentials in environment variables")
            return

        # Initialize Twilio client
        twilio_client = Client(account_sid, auth_token)
        
        # Parse collected data
        try:
            collected = json.loads(call.collected_data)
            logger.info(f"📋 Collected data for call {call.id}: {collected}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in collected_data for call {call.id}: {e}")
            return
        
        # Build SMS message
        msg_body = "📞 Call Summary:\n"
        for key, value in collected.items():
            # Clean up the key name for better readability
            clean_key = key.replace('_', ' ').title()
            msg_body += f"• {clean_key}: {value}\n"
        
        # Add call details
        if call.duration:
            minutes = call.duration // 60
            seconds = call.duration % 60
            msg_body += f"• Duration: {minutes}m {seconds}s\n"
        
        msg_body += f"• Call ID: {call.id}\n"
        msg_body += "Thank you for your time!"
        
        # Hardcode recipient for now
        recipient = "+923040610720"
        
        logger.info(f"📨 Sending call summary SMS to {recipient}")
        logger.info(f"📝 SMS Content: {msg_body}")
        
        # Send SMS
        try:
            message = twilio_client.messages.create(
                body=msg_body,
                from_=sender_number,
                to=recipient
            )
            
            # Mark as sent
            call.sms_sent = True
            db.commit()
            
            logger.info(f"✅ Call summary SMS sent successfully!")
            logger.info(f"📨 Message SID: {message.sid}")
            logger.info(f"📱 Status: {message.status}")
            logger.info(f"📞 From: {message.from_}")
            logger.info(f"📞 To: {message.to}")
            
            # Save outbound SMS to database (only if SMS sent successfully)
            agent = None
            if call.agent_id:
                agent = db.query(Agent).filter(Agent.id == call.agent_id).first()
            
            sms_record = SMS(
                agent_id=call.agent_id,
                from_number=sender_number,
                to_number=recipient,  # Use hardcoded number for database too
                body=msg_body,
                direction="outbound",
                message_sid=message.sid
            )
            db.add(sms_record)
            db.commit()
            db.refresh(sms_record)
            
            logger.info(f"✅ Call summary SMS saved to database with ID: {sms_record.id}")
        except Exception as sms_error:
            error_msg = str(sms_error)
            logger.error(f"❌ Error sending call summary SMS: {error_msg}")
            # Re-raise to be caught by outer exception handler
            raise
        
        # Save to Google Sheets if agent is configured
        if agent:
            try:
                sms_data = {
                    "Agent Name": agent.name or "Unknown",
                    "Direction": "outbound",
                    "From Number": sender_number,
                    "To Number": recipient,
                    "Message Body": msg_body,
                    "Date/Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Message SID": message.sid,
                    "Call ID": call.ultravox_call_id,
                    "Type": "Call Summary"
                }
                
                # Priority 1: Use Webhook URL if configured
                if agent.google_webhook_url:
                    logger.info(f"🚀 Forwarding call summary SMS data to Google Webhook")
                    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                        try:
                            resp = await client.post(agent.google_webhook_url, json=sms_data, timeout=10.0)
                            logger.info(f"✅ Webhook Response ({resp.status_code}): {resp.text}")
                        except Exception as e:
                            logger.error(f"❌ Webhook Error: {e}")
                
                # Priority 2: Use Direct API
                elif agent.google_spreadsheet_id:
                    sheet_name = agent.google_sheet_name or "Sheet1"
                    logger.info(f"📊 Appending call summary SMS to Google Sheets")
                    result = google_sheets_service.append_data_dict(agent.google_spreadsheet_id, sheet_name, sms_data)
                    if not result.get("success"):
                        logger.error(f"❌ Google Sheets Error: {result.get('error')}")
                    else:
                        logger.info(f"✅ Call summary SMS saved to Google Sheets")
            except Exception as e:
                logger.error(f"❌ Error saving call summary SMS to Google Sheets: {e}", exc_info=True)
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error sending call summary SMS for call {call.id}: {error_msg}", exc_info=True)
        
        # Check for specific Twilio errors
        if "Permission to send" in error_msg or "not enabled for the region" in error_msg:
            logger.warning(f"⚠️ SMS not enabled for region +923040610720, marking as sent to avoid retries")
            call.sms_sent = True
            db.commit()
        elif "not SMS-capable" in error_msg:
            logger.warning(f"⚠️ Sender number {sender_number} is not SMS-capable")
            call.sms_sent = True
            db.commit()
        elif "Invalid 'To' phone number" in error_msg or "cannot be the same" in error_msg:
            logger.warning(f"⚠️ Invalid recipient number +923040610720 or same as sender, marking as sent")
            call.sms_sent = True
            db.commit()
        else:
            logger.error(f"❌ Unhandled SMS error: {error_msg}")
            # Don't mark as sent for unknown errors to allow retry

@app.on_event("startup")
async def startup_event():
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
                
                print(f"🛠️ Registering Google Sheets Tool at {tool_url}")
                
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
                    print(f"🛠️ Ultravox Tool Registration Response: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Error registering tool: {e}")
    finally:
        db.close()

# --- Auth Setup ---
SECRET_KEY = "super-secret-key-change-this-in-prod"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 hours

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_user_optional(token: Optional[str] = None, db: Session = Depends(get_db)):
    # Helper for pages that might be viewed by guests or logged in users
    if not token:
        return None
    try:
        return await get_current_user(token, db)
    except:
        return None

# Models
# Models
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

# --- Auth API ---

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
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


@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    total_calls = db.query(Call).filter(Call.user_id == user.id).count()
    total_duration = db.query(func.sum(Call.duration)).filter(Call.user_id == user.id).scalar() or 0
    active_agents = db.query(Agent).filter(Agent.user_id == user.id).count()
    
    # Get calls per day for the last 7 days
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    calls_last_7_days = db.query(Call).filter(Call.user_id == user.id, Call.created_at >= seven_days_ago).count()
    
    # Live calls
    live_calls = db.query(Call).filter(
        Call.user_id == user.id, 
        Call.status.in_(["started", "active"])
    ).count()

    return {
        "totalCalls": total_calls,
        "totalDuration": total_duration,
        "activeAgents": active_agents,
        "recentCalls": calls_last_7_days,
        "liveCalls": live_calls
    }

# --- User Profile API ---

@app.get("/api/user/profile", response_model=UserResponse)
async def get_user_profile(user: User = Depends(get_current_user)):
    return user

@app.put("/api/user/profile", response_model=UserResponse)
async def update_user_profile(req: Dict[str, Any], db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if "full_name" in req: user.full_name = req["full_name"]
    if "email" in req: user.email = req["email"]
    if "business_type" in req: user.business_type = req["business_type"]
    
    db.commit()
    db.refresh(user)
    return user

# --- Recent Numbers API ---

@app.get("/api/recent-numbers")
async def get_recent_numbers(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    # Get last 10 unique numbers called by this user
    recent = db.query(Call.to_number).filter(
        Call.user_id == user.id,
        Call.to_number != None
    ).group_by(Call.to_number).order_by(func.max(Call.created_at).desc()).limit(10).all()
    
    return [r[0] for r in recent]

# --- Twilio Numbers API ---

@app.get("/api/numbers")
async def list_numbers(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    from models import TwilioNumber
    numbers = db.query(TwilioNumber).filter(TwilioNumber.user_id == user.id).all()
    return {"results": numbers}

@app.post("/api/numbers")
async def add_number(req: TwilioNumberCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    from models import TwilioNumber
    db_number = TwilioNumber(
        user_id=user.id,
        account_sid=req.account_sid,
        auth_token=req.auth_token,
        phone_number=req.phone_number,
        label=req.label
    )
    db.add(db_number)
    db.commit()
    db.refresh(db_number)
    return db_number

@app.delete("/api/numbers/{number_id}")
async def delete_number(number_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    from models import TwilioNumber
    db_number = db.query(TwilioNumber).filter(TwilioNumber.id == number_id, TwilioNumber.user_id == user.id).first()
    if not db_number:
        raise HTTPException(status_code=404, detail="Number not found")
    db.delete(db_number)
    db.commit()
    return {"success": True}

# --- Agents API ---

async def update_twilio_webhook(db: Session, twilio_number_id: int):
    """Update Twilio number's VoiceUrl to point to our inbound endpoint"""
    num = db.query(TwilioNumber).filter(TwilioNumber.id == twilio_number_id).first()
    if not num:
        return
    
    try:
        client = Client(num.account_sid, num.auth_token)
        # Get server host
        host = os.getenv("SERVER_HOST")
        if not host:
            print("⚠️ SERVER_HOST not set, cannot update Twilio webhook")
            return
            
        if not host.startswith("http"):
            host = f"https://{host}"
            
        # Ensure path prefix
        path_prefix = "/outbound"
        if path_prefix not in host:
            webhook_url = f"{host}{path_prefix}/api/inbound"
        else:
            webhook_url = f"{host}/api/inbound"
            
        # Find the number on Twilio and update it
        incoming_numbers = client.incoming_phone_numbers.list(phone_number=num.phone_number)
        if incoming_numbers:
            incoming_numbers[0].update(voice_url=webhook_url, voice_method='POST')
            print(f"✅ Updated Twilio webhook for {num.phone_number} to {webhook_url}")
        else:
            print(f"⚠️ Could not find number {num.phone_number} on Twilio account")
    except Exception as e:
        print(f"❌ Error updating Twilio webhook: {e}")

@app.get("/api/agents")
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

@app.post("/api/agents")
async def create_agent(req: AgentCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    # Check if number is already assigned to another agent
    if req.twilio_number_id:
        existing = db.query(Agent).filter(Agent.twilio_number_id == req.twilio_number_id).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Phone number is already assigned to agent: {existing.name}")

    # Ensure name is in system prompt
    system_prompt = req.system_prompt
    if req.name.lower() not in system_prompt.lower():
        system_prompt = f"Your name is {req.name}. {system_prompt}"

    # Create agent in database
    db_agent = Agent(
        name=req.name,
        system_prompt=system_prompt,
        voice=req.voice,
        language=req.language,
        user_id=user.id,
        twilio_number_id=req.twilio_number_id,
        google_spreadsheet_id=req.google_spreadsheet_id,
        google_sheet_name=req.google_sheet_name or "Sheet1",
        google_webhook_url=req.google_webhook_url
    )
    
    # Add tools if provided
    if req.tool_names:
        tools = db.query(Tool).filter(Tool.name.in_(req.tool_names)).all()
        db_agent.tools = tools
    
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)
    
    # Update Twilio Webhook if number assigned
    if db_agent.twilio_number_id:
        await update_twilio_webhook(db, db_agent.twilio_number_id)
    
    return {
        "agentId": str(db_agent.id),
        "name": db_agent.name,
        "systemPrompt": db_agent.system_prompt,
        "model": db_agent.model,
        "voice": db_agent.voice,
        "languageHint": db_agent.language,
        "twilio_number_id": db_agent.twilio_number_id,
        "google_spreadsheet_id": db_agent.google_spreadsheet_id,
        "google_sheet_name": db_agent.google_sheet_name,
        "google_webhook_url": db_agent.google_webhook_url,
        "selectedTools": [{"toolName": t.name} for t in db_agent.tools]
    }

@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str):
    api_key = os.getenv("ULTRAVOX_API_KEY")
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.ultravox.ai/api/agents/{agent_id}", headers={"X-API-Key": api_key})
        return resp.json()

@app.put("/api/agents/{agent_id}")
async def update_agent_full(agent_id: int, req: AgentCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    db_agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == user.id).first()
    if not db_agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Check if number is already assigned to another agent
    if req.twilio_number_id and req.twilio_number_id != db_agent.twilio_number_id:
        existing = db.query(Agent).filter(Agent.twilio_number_id == req.twilio_number_id, Agent.id != agent_id).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Phone number is already assigned to agent: {existing.name}")

    # Ensure name is in system prompt
    system_prompt = req.system_prompt
    if req.name.lower() not in system_prompt.lower():
        system_prompt = f"Your name is {req.name}. {system_prompt}"

    db_agent.name = req.name
    db_agent.system_prompt = system_prompt
    db_agent.voice = req.voice
    db_agent.language = req.language
    db_agent.google_spreadsheet_id = req.google_spreadsheet_id
    db_agent.google_sheet_name = req.google_sheet_name or "Sheet1"
    db_agent.google_webhook_url = req.google_webhook_url
    
    old_number_id = db_agent.twilio_number_id
    db_agent.twilio_number_id = req.twilio_number_id
    
    if req.tool_names is not None:
        tools = db.query(Tool).filter(Tool.name.in_(req.tool_names)).all()
        db_agent.tools = tools
        
    db.commit()
    db.refresh(db_agent)

    # Update Twilio Webhook if number changed or assigned
    if db_agent.twilio_number_id and db_agent.twilio_number_id != old_number_id:
        await update_twilio_webhook(db, db_agent.twilio_number_id)

    return db_agent

@app.patch("/api/agents/{agent_id}")
async def update_agent(agent_id: str, req: Dict[str, Any], db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    # Update local DB if it's an integer ID
    try:
        aid = int(agent_id)
        db_agent = db.query(Agent).filter(Agent.id == aid, Agent.user_id == user.id).first()
        if db_agent:
            if "name" in req: db_agent.name = req["name"]
            if "systemPrompt" in req: db_agent.system_prompt = req["systemPrompt"]
            if "voice" in req: db_agent.voice = req["voice"]
            if "language" in req: db_agent.language = req["language"]
            if "twilio_number_id" in req: db_agent.twilio_number_id = req["twilio_number_id"]
            db.commit()
    except:
        pass

    api_key = os.getenv("ULTRAVOX_API_KEY")
    async with httpx.AsyncClient() as client:
        resp = await client.patch(f"https://api.ultravox.ai/api/agents/{agent_id}", headers={"X-API-Key": api_key}, json=req)
        if resp.status_code not in [200, 204]:
             return {"success": True, "local_update": True}
        return resp.json() if resp.status_code == 200 else {"success": True}

@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    # 1. Delete from Ultravox
    api_key = os.getenv("ULTRAVOX_API_KEY")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(f"https://api.ultravox.ai/api/agents/{agent_id}", headers={"X-API-Key": api_key})
            # We don't strictly fail if it's already gone from Ultravox, we still want to clean up local DB
            if resp.status_code not in [200, 204, 404]:
                print(f"Warning: Failed to delete from Ultravox: {resp.text}")
    except Exception as e:
        print(f"Error deleting from Ultravox: {e}")

    # 2. Delete from Local DB
    try:
        # Try as integer ID first
        agent_id_int = int(agent_id)
        db_agent = db.query(Agent).filter(Agent.id == agent_id_int, Agent.user_id == user.id).first()
    except ValueError:
        # Try as Ultravox ID
        db_agent = db.query(Agent).filter(Agent.ultravox_agent_id == agent_id, Agent.user_id == user.id).first()
    
    if db_agent:
        db.delete(db_agent)
        db.commit()
        return {"success": True, "message": "Agent deleted"}
    else:
        # If not found in DB but we tried to delete from Ultravox, return success anyway or 404?
        # If the user passed an ID that doesn't exist, it's 404.
        raise HTTPException(status_code=404, detail="Agent not found")

# --- Tools API ---

@app.get("/api/tools")
async def list_tools(db: Session = Depends(get_db)):
    tools = db.query(Tool).all()
    results = []
    for t in tools:
        results.append({
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "base_url": t.base_url,
            "http_method": t.http_method,
            "created": t.created_at.isoformat() if t.created_at else None
        })
    return {"results": results}

@app.post("/api/tools")
@app.post("/api/tools")
async def create_tool(tool: ToolDefinition, db: Session = Depends(get_db)):
    api_key = os.getenv("ULTRAVOX_API_KEY")
    
    # Construct Ultravox Tool Definition
    definition = {
        "description": tool.description,
        "http": {
            "baseUrlPattern": tool.base_url,
            "httpMethod": tool.http_method
        }
    }

    # Handle Parameters
    if tool.parameters:
        definition["dynamicParameters"] = tool.parameters

    # Handle Authentication
    if tool.authentication:
        auth = tool.authentication
        if auth.get("type") == "api_key":
            definition["http"]["authentication"] = {
                "apiKey": {
                    "headerName": auth.get("header_name", "X-API-Key"),
                    "secret": auth.get("api_key")
                }
            }
        elif auth.get("type") == "bearer":
            definition["http"]["authentication"] = {
                "httpHeader": {
                    "name": "Authorization",
                    "secret": f"Bearer {auth.get('token')}"
                }
            }
        elif auth.get("type") == "basic":
            import base64
            creds = f"{auth.get('username')}:{auth.get('password')}"
            b64_creds = base64.b64encode(creds.encode()).decode()
            definition["http"]["authentication"] = {
                "httpHeader": {
                    "name": "Authorization",
                    "secret": f"Basic {b64_creds}"
                }
            }

    payload = {
        "name": tool.name,
        "definition": definition
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post("https://api.ultravox.ai/api/tools", headers={"X-API-Key": api_key}, json=payload)
        if resp.status_code != 201:
            # If it already exists in Ultravox, we might want to still save it locally if missing
            if resp.status_code != 409:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
        
        # Save to local DB
        db_tool = db.query(Tool).filter(Tool.name == tool.name).first()
        if not db_tool:
            db_tool = Tool(
                name=tool.name,
                description=tool.description,
                base_url=tool.base_url,
                http_method=tool.http_method
            )
            db.add(db_tool)
            db.commit()
            
        return resp.json() if resp.status_code == 201 else {"success": True, "message": "Tool already exists in Ultravox, saved locally"}

@app.delete("/api/tools/{tool_name}")
async def delete_tool(tool_name: str):
    api_key = os.getenv("ULTRAVOX_API_KEY")
    async with httpx.AsyncClient() as client:
        resp = await client.delete(f"https://api.ultravox.ai/api/tools/{tool_name}", headers={"X-API-Key": api_key})
        if resp.status_code not in [200, 204]:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return {"success": True}

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
    else:
        # Fallback: Try to find the most recent active call
        logger.info("⚠️ No Call ID found in headers. Trying to find most recent active call...")
        call_record = db.query(Call).filter(Call.status == "started").order_by(Call.created_at.desc()).first()
        if call_record:
            logger.info(f"🔄 Using Agent from recent call: {call_record.id}")
            if call_record.agent:
                agent = call_record.agent
                logger.info(f"✅ Found Agent: {agent.name}")
            else:
                logger.warning(f"⚠️ Call {call_record.id} has no agent assigned")
            call_id = call_record.ultravox_call_id

    # Save collected data to the call record (for both paths)
    if call_record:
        # Check if this exact data was already saved to prevent duplicates
        existing_data = call_record.collected_data
        new_data_str = json.dumps(data, sort_keys=True)
        
        if existing_data:
            try:
                existing_data_str = json.dumps(json.loads(existing_data), sort_keys=True)
                if existing_data_str == new_data_str:
                    logger.info(f"⚠️ Duplicate data detected for Call {call_record.id}, skipping save")
                else:
                    # Data is different, update it
                    try:
                        call_record.collected_data = new_data_str
                        db.commit()
                        logger.info(f"💾 Updated collected data for Call {call_record.id}")
                    except Exception as e:
                        logger.error(f"⚠️ Failed to update collected data: {e}")
            except json.JSONDecodeError:
                # Invalid existing data, overwrite it
                call_record.collected_data = new_data_str
                db.commit()
                logger.info(f"💾 Saved collected data to Call {call_record.id} (overwrote invalid data)")
        else:
            # First time saving data
            try:
                call_record.collected_data = new_data_str
                db.commit()
                logger.info(f"💾 Saved collected data to Call {call_record.id}")
            except Exception as e:
                logger.error(f"⚠️ Failed to save collected data to DB: {e}")

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
                # Still return success to Ultravox so tool doesn't retry
                return {"success": True, "message": "Data received, webhook failed"}

    # 3. Priority 2: Use Direct API (requires service-account.json)
    spreadsheet_id = spreadsheet_id or (agent.google_spreadsheet_id if agent else None)
    sheet_name = sheet_name or (agent.google_sheet_name if agent else "Sheet1")
    
    if not spreadsheet_id:
        if not agent:
            logger.warning("⚠️ No agent found and no spreadsheet_id provided in request")
        else:
            logger.warning(f"⚠️ Agent '{agent.name}' has no Google Sheets configuration")
        # Return success anyway so Ultravox doesn't retry, but log the issue
        return {"success": True, "message": "No Google Sheets configuration found. Data saved to call record only."}

    logger.info(f"📊 Appending to Google Sheets via API: {spreadsheet_id} / {sheet_name}")
    try:
        result = google_sheets_service.append_data_dict(spreadsheet_id, sheet_name, data)
        if not result.get("success"):
            logger.error(f"❌ Google Sheets API Error: {result.get('error')}")
            # Return success anyway so Ultravox doesn't retry, but log the error
            return {"success": True, "message": f"Data received but Google Sheets save failed: {result.get('error')}"}
        logger.info(f"✅ Successfully appended data to Google Sheets")
        return result
    except Exception as e:
        logger.error(f"❌ Exception while saving to Google Sheets: {e}", exc_info=True)
        # Return success anyway so Ultravox doesn't retry
        return {"success": True, "message": f"Data received but Google Sheets save failed: {str(e)}"}

@app.post("/api/calls/{call_id}/end")
async def end_call(call_id: str, db: Session = Depends(get_db)):
    """End an active call manually"""
    api_key = os.getenv("ULTRAVOX_API_KEY")
    twilio_sid_env = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_token_env = os.getenv("TWILIO_AUTH_TOKEN")
    
    # Try to find the call in our DB to get Twilio SID
    db_call = db.query(Call).filter(Call.ultravox_call_id == call_id).first()
    
    try:
        # 1. Try to end via Twilio if we have the SID
        if db_call and db_call.twilio_sid and twilio_sid_env and twilio_token_env:
            try:
                client = Client(twilio_sid_env, twilio_token_env)
                client.calls(db_call.twilio_sid).update(status='completed')
                print(f"✅ Call {call_id} ended via Twilio SID {db_call.twilio_sid}")
                db_call.status = "ended"
                db.commit()
                return {"success": True, "message": "Call ended via Twilio"}
            except Exception as e:
                print(f"⚠️ Twilio end error: {e}")

        # 2. Try to end via Ultravox
        async with httpx.AsyncClient() as client:
            # Try DELETE first
            try:
                resp = await client.delete(f"https://api.ultravox.ai/api/calls/{call_id}", headers={"X-API-Key": api_key})
                if resp.status_code in [200, 204]:
                    if db_call:
                        db_call.status = "ended"
                        db.commit()
                    return {"success": True, "message": "Call ended via Ultravox"}
            except Exception as e:
                print(f"DELETE endpoint error: {e}")

            # Try other endpoints
            endpoints_to_try = [
                f"https://api.ultravox.ai/api/calls/{call_id}/end",
                f"https://api.ultravox.ai/api/calls/{call_id}/hangup"
            ]
            
            for endpoint in endpoints_to_try:
                try:
                    resp = await client.post(endpoint, headers={"X-API-Key": api_key}, json={"status": "ended"})
                    if resp.status_code in [200, 204]:
                        if db_call:
                            db_call.status = "ended"
                            db.commit()
                        return {"success": True, "message": "Call ended successfully"}
                except Exception as e:
                    print(f"POST endpoint error: {e}")
            
            # If we reached here and it's a Twilio call, maybe it's already ended or we can't find it
            if db_call:
                db_call.status = "ended"
                db.commit()
                return {"success": True, "message": "Call marked as ended in database"}
                
            raise HTTPException(status_code=500, detail="Unable to end call")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending call: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending call: {str(e)}")

# --- CSV Upload ---

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...), user: User = Depends(get_current_user)):
    """Upload CSV or Excel file with phone numbers"""
    try:
        contents = await file.read()
        
        # Detect file type and parse
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="File must be CSV or Excel format")
        
        # Extract phone numbers (look for common column names)
        phone_column = None
        for col in df.columns:
            if col.lower() in ['phone', 'phone_number', 'number', 'mobile', 'tel', 'telephone']:
                phone_column = col
                break
        
        if phone_column is None:
            # If no standard column found, use first column
            phone_column = df.columns[0]
        
        # Extract and clean phone numbers
        numbers = df[phone_column].astype(str).tolist()
        numbers = [n.strip() for n in numbers if n and n.strip() and n != 'nan']
        
        return {
            "success": True,
            "count": len(numbers),
            "numbers": numbers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# --- Satisfaction Analysis ---

@app.post("/api/analyze-satisfaction/{call_id}")
async def analyze_satisfaction(call_id: str):
    """Analyze call transcript and generate satisfaction score"""
    api_key = os.getenv("ULTRAVOX_API_KEY")
    
    try:
        # Fetch call details with messages
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"https://api.ultravox.ai/api/calls/{call_id}", headers={"X-API-Key": api_key})
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail="Failed to fetch call details")
            
            call_data = resp.json()
            
            # Fetch messages
            msg_resp = await client.get(f"https://api.ultravox.ai/api/calls/{call_id}/messages", headers={"X-API-Key": api_key})
            messages = msg_resp.json().get("results", []) if msg_resp.status_code == 200 else []
        
        # Build transcript
        transcript = "\n".join([f"{msg['role']}: {msg.get('text', '')}" for msg in messages if msg.get('text')])
        
        if not transcript:
            return {"score": 0, "analysis": "No transcript available"}
        
        # Simple satisfaction analysis (you can enhance this with AI)
        score = calculate_satisfaction_score(transcript, call_data)
        
        return {
            "score": score,
            "call_id": call_id,
            "analysis": generate_satisfaction_analysis(transcript, score)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing satisfaction: {str(e)}")

    # Ensure score is between 1-10
    return max(1, min(10, score))

async def analyze_satisfaction_ai(transcript: str, duration_seconds: float) -> int:
    """Analyze satisfaction using AI logic (Groq preferred, then OpenAI, then heuristic)"""
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    prompt = f"""
    Analyze the following call transcript and provide a satisfaction score from 1 to 10.
    Criteria:
    - Human answered (not voicemail): +3 marks
    - Call longer than a minute: +3 marks
    - Sale/Agreement (customer agrees to offer): +3 marks
    - Customer happy at the end: +1 mark
    
    Transcript:
    {transcript}
    
    Duration: {duration_seconds} seconds
    
    Return ONLY the integer score.
    """

    if groq_key:
        try:
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0
            )
            score_text = response.choices[0].message.content.strip()
            return int(''.join(filter(str.isdigit, score_text)) or 5)
        except Exception as e:
            print(f"⚠️ Groq analysis error: {e}")

    if openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0
            )
            score_text = response.choices[0].message.content.strip()
            return int(''.join(filter(str.isdigit, score_text)) or 5)
        except Exception as e:
            print(f"⚠️ OpenAI analysis error: {e}")

    # Fallback to heuristic
    score = 1 # Base score
    transcript_lower = transcript.lower()
    
    # 1. Human answered (not voicemail)
    voicemail_keywords = ["voicemail", "leave a message", "after the tone", "not available"]
    is_voicemail = any(kw in transcript_lower for kw in voicemail_keywords)
    if not is_voicemail and len(transcript) > 20:
        score += 3
        
    # 2. Call longer than a minute
    if duration_seconds > 60:
        score += 3
        
    # 3. Sale/Agreement
    sale_keywords = ["agree", "yes", "sign me up", "interested", "deal", "accept", "book", "order"]
    if any(kw in transcript_lower for kw in sale_keywords):
        score += 3
        
    # 4. Customer happy at the end
    happy_keywords = ["thank", "great", "perfect", "excellent", "good", "appreciate", "bye"]
    last_part = transcript_lower[-200:] if len(transcript_lower) > 200 else transcript_lower
    if any(kw in last_part for kw in happy_keywords):
        score += 1
        
    return max(1, min(10, score))

def generate_satisfaction_analysis(transcript: str, score: int) -> str:
    """Generate human-readable analysis"""
    if score >= 8:
        return "Customer appears highly satisfied. Positive engagement throughout the call."
    elif score >= 6:
        return "Customer seems moderately satisfied. Some positive indicators present."
    elif score >= 4:
        return "Customer satisfaction is neutral. Mixed signals in the conversation."
    else:
        return "Customer appears dissatisfied. Negative indicators detected."

# --- Calls ---


@app.post("/api/call-agent")
async def call_agent(req: AgentCallRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    # Balance Check
    if user.balance <= 0:
        raise HTTPException(status_code=402, detail="Insufficient balance. Please top up.")
    ultravox_api_key = os.getenv("ULTRAVOX_API_KEY")
    if not ultravox_api_key:
        raise HTTPException(status_code=500, detail="ULTRAVOX_API_KEY not set")

    # Get agent from database
    try:
        agent_id_int = int(req.agent_id)
        agent = db.query(Agent).filter(Agent.id == agent_id_int).first()
    except ValueError:
        # If it's not an integer, try as ultravox agent ID
        agent = db.query(Agent).filter(Agent.ultravox_agent_id == req.agent_id).first()
    
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {req.agent_id}")

    # Use agent's assigned Twilio number if available
    if agent.twilio_number:
        twilio_sid = agent.twilio_number.account_sid
        twilio_token = agent.twilio_number.auth_token
        from_number = agent.twilio_number.phone_number
    else:
        twilio_sid = req.twilio_account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        twilio_token = req.twilio_auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        from_number = req.from_number or os.getenv("TWILIO_PHONE_NUMBER")
    
    if not all([twilio_sid, twilio_token, from_number]):
        raise HTTPException(status_code=400, detail="Twilio credentials/number missing")

    # 1. Create Ultravox Call with agent configuration
    url = "https://api.ultravox.ai/api/calls"
    
    # Build payload with agent's configuration
    payload = {
        "systemPrompt": agent.system_prompt,
        "model": agent.model,
        "voice": agent.voice,
        "languageHint": agent.language,
        "temperature": 0.3,
        "medium": {"twilio": {}},
        "firstSpeakerSettings": {"user": {}},
        "recordingEnabled": True
    }
    
    # Add tools if agent has them
    if agent.tools:
        payload["selectedTools"] = [{"toolName": tool.name} for tool in agent.tools]
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers={"X-API-Key": ultravox_api_key}, json=payload)
            if resp.status_code != 201:
                raise HTTPException(status_code=resp.status_code, detail=f"Ultravox Error: {resp.text}")
            
            data = resp.json()
            join_url = data["joinUrl"]
            call_id = data["callId"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create Ultravox call: {str(e)}")

    # 2. Initiate Twilio Call (Bridging)
    try:
        client = Client(twilio_sid, twilio_token)
        host = req.server_host or os.getenv("SERVER_HOST")
        if not host:
             raise HTTPException(status_code=400, detail="Server Host required")
        
        if not host.startswith("http"):
            host = f"https://{host}"
            
        path_prefix = "/outbound"
        if path_prefix not in host:
             twiml_url = f"{host}{path_prefix}/api/twiml?joinUrl={join_url}"
        else:
             twiml_url = f"{host}/api/twiml?joinUrl={join_url}"
        
        call = client.calls.create(
            to=req.to_number,
            from_=from_number,
            url=twiml_url
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Twilio Error: {str(e)}")

    # Save call to DB
    try:
        new_call = Call(
            ultravox_call_id=call_id,
            user_id=user.id,
            agent_id=agent.id,
            to_number=req.to_number,
            from_number=from_number,
            status="started",
            twilio_sid=call.sid,
            direction="outbound"
        )
        db.add(new_call)
        db.commit()
    except Exception as e:
        print(f"Error saving call to DB: {e}")

    return {"status": "success", "call_id": call_id, "twilio_sid": call.sid}


@app.post("/api/call")
async def make_call(call_request: CallRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    # Balance Check
    if user.balance <= 0:
        raise HTTPException(status_code=402, detail="Insufficient balance. Please top up.")
    # Config
    ultravox_api_key = os.getenv("ULTRAVOX_API_KEY")
    if not ultravox_api_key:
        raise HTTPException(status_code=500, detail="ULTRAVOX_API_KEY not set")

    twilio_sid = call_request.twilio_account_sid or os.getenv("TWILIO_ACCOUNT_SID")
    twilio_token = call_request.twilio_auth_token or os.getenv("TWILIO_AUTH_TOKEN")
    from_number = call_request.from_number or os.getenv("TWILIO_PHONE_NUMBER")
    
    if not all([twilio_sid, twilio_token, from_number]):
        raise HTTPException(status_code=400, detail="Twilio credentials/number missing")

    # 1. Create Ultravox Call
    ultravox_url = "https://api.ultravox.ai/api/calls"
    headers = {"X-API-Key": ultravox_api_key}
    
    payload = {
        "systemPrompt": call_request.system_prompt,
        "model": "fixie-ai/ultravox",
        "voice": call_request.voice or "a656a751-b754-4621-b571-e1298cb7e5bb",
        "languageHint": "en",
        "temperature": 0.3,
        "medium": {"twilio": {}}, 
        "firstSpeakerSettings": {"user": {}}, # User speaks first (answers phone)
        "recordingEnabled": True
    }
    
    if call_request.tools:
        payload["selectedTools"] = call_request.tools

    print(f"🚀 Initiating Instant Call with prompt: {call_request.system_prompt[:100]}...")
    print(f"📦 Payload: {json.dumps(payload)}")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(ultravox_url, headers=headers, json=payload, timeout=30.0)
            if resp.status_code != 201:
                print(f"❌ Ultravox Error: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=resp.status_code, detail=f"Ultravox Error: {resp.text}")
            
            data = resp.json()
            join_url = data["joinUrl"]
            call_id = data["callId"]
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        print(f"❌ Failed to create Ultravox call: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create Ultravox call: {str(e)}")

    # 2. Initiate Twilio Call
    try:
        client = Client(twilio_sid, twilio_token)
        
        # Construct callback URL
        host = call_request.server_host or os.getenv("SERVER_HOST")
        if not host:
             # Fallback to request host if not provided, but this might be localhost
             raise HTTPException(status_code=400, detail="Server Host required for Twilio callback")
        
        # Ensure protocol
        if not host.startswith("http"):
            host = f"https://{host}"
            
        # IMPORTANT: Ensure we point to the /outbound path if using the main domain
        # If the host is just the domain, append /outbound
        # We assume if the user provides a host, it's the root domain.
        # We need to route to THIS service's /api/twiml endpoint.
        # Since Nginx routes /outbound/ -> localhost:8002/, the external URL is /outbound/api/twiml
        
        # Check if we are already including /outbound in the host (unlikely)
        path_prefix = "/outbound"
        if path_prefix not in host:
             twiml_url = f"{host}{path_prefix}/api/twiml?joinUrl={join_url}"
        else:
             twiml_url = f"{host}/api/twiml?joinUrl={join_url}"
        
        print(f"Initiating call to {call_request.to_number} from {from_number} with URL {twiml_url}")
        
        call = client.calls.create(
            to=call_request.to_number,
            from_=from_number,
            url=twiml_url
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Twilio Error: {str(e)}")

    # Save call to DB
    try:
        new_call = Call(
            ultravox_call_id=call_id,
            user_id=user.id,
            to_number=call_request.to_number,
            from_number=from_number,
            status="started",
            twilio_sid=call.sid,
            direction="outbound"
        )
        db.add(new_call)
        db.commit()
    except Exception as e:
        print(f"Error saving call to DB: {e}")

    return {"status": "success", "call_id": call_id, "twilio_sid": call.sid}

@app.post("/api/twiml")
@app.get("/api/twiml")
async def get_twiml(joinUrl: str):
    # Return TwiML to connect to Ultravox
    # Ultravox joinUrl is wss://...
    # Twilio <Stream> connects to wss
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{joinUrl}" />
    </Connect>
</Response>"""
    return Response(content=xml, media_type="application/xml")

@app.post("/api/inbound")
@app.get("/api/inbound")
async def handle_inbound(request: Request, db: Session = Depends(get_db)):
    """Handle inbound calls from Twilio"""
    # Twilio sends data as form parameters
    form_data = await request.form()
    # Twilio uses 'To' for the number dialed (our Twilio number)
    # and 'From' for the caller's number
    to_number = form_data.get("To")
    from_number = form_data.get("From")
    call_sid = form_data.get("CallSid")
    
    print(f"📞 Incoming call to {to_number} from {from_number} (SID: {call_sid})")
    
    if not to_number:
        return Response(content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Error: No destination number.</Say></Response>', media_type="application/xml")

    # Find agent associated with this number
    # We need to match the phone number. Twilio usually sends it with '+'.
    # We check both with and without '+' to be safe.
    agent = db.query(Agent).join(TwilioNumber).filter(
        (TwilioNumber.phone_number == to_number) | 
        (TwilioNumber.phone_number == to_number.replace("+", ""))
    ).first()
    
    if not agent:
        print(f"⚠️ No agent found for number {to_number}")
        # Try a more flexible search if needed, but for now exact match
        return Response(content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Sorry, no agent is assigned to this number.</Say></Response>', media_type="application/xml")

    # Check user balance
    user = agent.user
    if not user or user.balance <= 0:
        print(f"⚠️ User {user.username if user else 'unknown'} has insufficient balance for inbound call")
        return Response(content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Sorry, this service is currently unavailable.</Say></Response>', media_type="application/xml")

    # Create Ultravox Call
    ultravox_api_key = os.getenv("ULTRAVOX_API_KEY")
    if not ultravox_api_key:
         return Response(content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Configuration error.</Say></Response>', media_type="application/xml")

    url = "https://api.ultravox.ai/api/calls"
    payload = {
        "systemPrompt": agent.system_prompt,
        "model": agent.model,
        "voice": agent.voice,
        "languageHint": agent.language,
        "temperature": 0.3,
        "medium": {"twilio": {}},
        "firstSpeakerSettings": {"agent": {}}, # Agent speaks first for inbound
        "recordingEnabled": True
    }
    
    # Add tools if agent has them
    if agent.tools:
        payload["selectedTools"] = [{"toolName": tool.name} for tool in agent.tools]
        
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers={"X-API-Key": ultravox_api_key}, json=payload)
            if resp.status_code != 201:
                print(f"❌ Ultravox Error: {resp.text}")
                return Response(content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Error connecting to AI agent.</Say></Response>', media_type="application/xml")
            
            data = resp.json()
            join_url = data["joinUrl"]
            ultravox_call_id = data["callId"]
            
            # Save call to DB
            new_call = Call(
                ultravox_call_id=ultravox_call_id,
                user_id=user.id,
                agent_id=agent.id,
                to_number=to_number,
                from_number=from_number,
                status="started",
                twilio_sid=call_sid,
                direction="inbound"
            )
            db.add(new_call)
            db.commit()
            
            print(f"✅ Inbound call connected. Ultravox Call ID: {ultravox_call_id}")
            
            # Return TwiML
            xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{join_url}" />
    </Connect>
</Response>"""
            return Response(content=xml, media_type="application/xml")
            
    except Exception as e:
        print(f"❌ Inbound error: {str(e)}")
        return Response(content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Internal server error.</Say></Response>', media_type="application/xml")

@app.post("/sms")
@app.get("/sms")
async def receive_sms(request: Request, db: Session = Depends(get_db)):
    """Receive inbound SMS from Twilio"""
    try:
        # Twilio sends data as form parameters
        form_data = await request.form()
        
        # Get and normalize phone numbers (handle URL encoding)
        sender_raw = form_data.get('From', 'Unknown')
        your_number_raw = form_data.get('To', 'Unknown')
        
        # Decode URL encoding if present
        sender = unquote(sender_raw) if sender_raw else 'Unknown'
        your_number = unquote(your_number_raw) if your_number_raw else 'Unknown'
        
        message_body = form_data.get('Body', '')
        message_sid = form_data.get('MessageSid', '')
        direction = form_data.get('MessageStatus', 'inbound')  # Usually 'inbound' for received messages
        
        logger.info(f"📱 SMS received - From: {sender}, To: {your_number}, Body: {message_body[:50]}...")
        
        # Normalize phone numbers for matching (remove +, spaces, dashes, etc.)
        def normalize_phone(phone):
            if not phone:
                return ""
            # Remove +, spaces, dashes, parentheses
            normalized = phone.replace("+", "").replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace(".", "")
            return normalized

        your_number_normalized = normalize_phone(your_number)
        logger.info(f"🔍 Looking for agent with number: {your_number} (normalized: {your_number_normalized})")

        # Find agent associated with this number - try multiple matching strategies
        agent = None

        # Strategy 1: Exact match with stored numbers
        agent = db.query(Agent).join(TwilioNumber).filter(
            (TwilioNumber.phone_number == your_number)
        ).first()

        # Strategy 2: If no exact match, try normalized matching
        if not agent:
            logger.info("🔄 No exact match found, trying normalized matching...")
            all_twilio_numbers = db.query(TwilioNumber).all()
            logger.info(f"📋 Checking against {len(all_twilio_numbers)} Twilio numbers in database")

            for tn in all_twilio_numbers:
                stored_normalized = normalize_phone(tn.phone_number)
                logger.info(f"  Comparing: stored '{tn.phone_number}' -> '{stored_normalized}' vs incoming '{your_number}' -> '{your_number_normalized}'")

                if stored_normalized == your_number_normalized:
                    agent = db.query(Agent).filter(Agent.twilio_number_id == tn.id).first()
                    if agent:
                        logger.info(f"✅ Found agent '{agent.name}' via normalized number matching")
                        break

        # Strategy 3: Last resort - try partial matching for common formatting issues
        if not agent:
            logger.info("🔄 Still no match, trying partial matching...")
            for tn in all_twilio_numbers:
                stored_normalized = normalize_phone(tn.phone_number)

                # Try multiple partial matching strategies
                # Strategy 3a: Last 9 digits (skip area code differences)
                if len(stored_normalized) >= 9 and len(your_number_normalized) >= 9:
                    stored_last9 = stored_normalized[-9:]
                    incoming_last9 = your_number_normalized[-9:]
                    logger.info(f"  Partial match (last 9): stored '{stored_last9}', incoming '{incoming_last9}'")
                    if stored_last9 == incoming_last9:
                        agent = db.query(Agent).filter(Agent.twilio_number_id == tn.id).first()
                        if agent:
                            logger.info(f"✅ Found agent '{agent.name}' via last 9 digits matching")
                            break

                # Strategy 3b: Last 8 digits
                if not agent and len(stored_normalized) >= 8 and len(your_number_normalized) >= 8:
                    stored_last8 = stored_normalized[-8:]
                    incoming_last8 = your_number_normalized[-8:]
                    logger.info(f"  Partial match (last 8): stored '{stored_last8}', incoming '{incoming_last8}'")
                    if stored_last8 == incoming_last8:
                        agent = db.query(Agent).filter(Agent.twilio_number_id == tn.id).first()
                        if agent:
                            logger.info(f"✅ Found agent '{agent.name}' via last 8 digits matching")
                            break

                # Strategy 3c: Last 7 digits (very lenient matching)
                if not agent and len(stored_normalized) >= 7 and len(your_number_normalized) >= 7:
                    stored_last7 = stored_normalized[-7:]
                    incoming_last7 = your_number_normalized[-7:]
                    logger.info(f"  Partial match (last 7): stored '{stored_last7}', incoming '{incoming_last7}'")
                    if stored_last7 == incoming_last7:
                        agent = db.query(Agent).filter(Agent.twilio_number_id == tn.id).first()
                        if agent:
                            logger.info(f"✅ Found agent '{agent.name}' via last 7 digits matching")
                            break

                # Strategy 3d: Fuzzy matching - check if one number contains the other
                if not agent:
                    # Check if the shorter number is contained in the longer one
                    shorter = min(stored_normalized, your_number_normalized, key=len)
                    longer = max(stored_normalized, your_number_normalized, key=len)
                    logger.info(f"  Fuzzy match: checking if '{shorter}' is in '{longer}'")
                    if shorter in longer:
                        agent = db.query(Agent).filter(Agent.twilio_number_id == tn.id).first()
                        if agent:
                            logger.info(f"✅ Found agent '{agent.name}' via fuzzy substring matching")
                            break

        
        agent_id = agent.id if agent else None
        agent_name = agent.name if agent else "Unknown"

        if agent:
            logger.info(f"✅ Agent found: {agent.name} (ID: {agent.id})")
            logger.info(f"📊 Agent Google Sheets config: Webhook={agent.google_webhook_url}, SheetID={agent.google_spreadsheet_id}")
        else:
            logger.warning(f"⚠️ No agent found for number {your_number}. SMS will be saved without agent association.")
            # Log all available Twilio numbers for debugging
            all_numbers = db.query(TwilioNumber).all()
            logger.info(f"📋 Available Twilio numbers in database: {[f'{tn.phone_number} (ID:{tn.id})' for tn in all_numbers]}")

            # Strategy 4: Fallback - if there's only one agent with a Twilio number, use it
            agents_with_numbers = db.query(Agent).filter(Agent.twilio_number_id != None).all()
            logger.info(f"  Fallback check: Found {len(agents_with_numbers)} agents with Twilio numbers")
            if len(agents_with_numbers) == 1:
                agent = agents_with_numbers[0]
                agent_id = agent.id
                agent_name = agent.name
                logger.info(f"✅ Fallback: Using the only agent with Twilio number: '{agent.name}' (ID: {agent.id})")
                logger.info(f"📊 Fallback agent Google Sheets config: Webhook={agent.google_webhook_url}, SheetID={agent.google_spreadsheet_id}")
            elif len(agents_with_numbers) > 1:
                logger.warning(f"⚠️ Multiple agents with Twilio numbers found ({len(agents_with_numbers)}), cannot determine which one to use")
            else:
                logger.warning(f"⚠️ No agents with Twilio numbers found in database")
        
        # Save SMS to database
        sms_record = SMS(
            agent_id=agent_id,
            from_number=sender,
            to_number=your_number,
            body=message_body,
            direction="inbound",
            message_sid=message_sid
        )
        db.add(sms_record)
        db.commit()
        db.refresh(sms_record)
        
        logger.info(f"✅ SMS saved to database with ID: {sms_record.id}, Agent: {agent_name}")
        
        # Save to Google Sheets if agent is configured
        if agent:
            try:
                # Prepare data for Google Sheets
                sms_data = {
                    "Agent Name": agent_name or "Unknown",
                    "Direction": "inbound",
                    "From Number": sender,
                    "To Number": your_number,
                    "Message Body": message_body,
                    "Date/Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Message SID": message_sid
                }
                
                logger.info(f"📊 Agent Google Sheets config - Webhook: {agent.google_webhook_url}, Spreadsheet ID: {agent.google_spreadsheet_id}, Sheet: {agent.google_sheet_name}")
                
                # Priority 1: Use Webhook URL if configured
                if agent.google_webhook_url:
                    logger.info(f"🚀 Forwarding SMS data to Google Webhook: {agent.google_webhook_url}")
                    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                        try:
                            resp = await client.post(agent.google_webhook_url, json=sms_data, timeout=10.0)
                            logger.info(f"✅ Webhook Response ({resp.status_code}): {resp.text}")
                        except Exception as e:
                            logger.error(f"❌ Webhook Error: {e}")
                
                # Priority 2: Use Direct API (requires service-account.json)
                elif agent.google_spreadsheet_id:
                    sheet_name = agent.google_sheet_name or "Sheet1"
                    logger.info(f"📊 Appending SMS to Google Sheets via API: {agent.google_spreadsheet_id} / {sheet_name}")
                    try:
                        result = google_sheets_service.append_data_dict(agent.google_spreadsheet_id, sheet_name, sms_data)
                        if not result.get("success"):
                            logger.error(f"❌ Google Sheets Error: {result.get('error')}")
                        else:
                            logger.info(f"✅ SMS saved to Google Sheets successfully")
                    except Exception as gs_error:
                        logger.error(f"❌ Exception saving SMS to Google Sheets: {gs_error}", exc_info=True)
                else:
                    logger.warning(f"⚠️ Agent '{agent.name}' has no Google Sheets configuration (no webhook URL or spreadsheet ID)")
            except Exception as e:
                logger.error(f"❌ Error saving SMS to Google Sheets: {e}", exc_info=True)
        else:
            logger.warning(f"⚠️ No agent found, skipping Google Sheets save")
        
        # Return TwiML response to Twilio
        from twilio.twiml.messaging_response import MessagingResponse
        resp = MessagingResponse()
        return Response(content=str(resp), media_type="text/xml", status_code=200)
        
    except Exception as e:
        logger.error(f"❌ Error processing SMS: {e}")
        # Still return valid TwiML
        from twilio.twiml.messaging_response import MessagingResponse
        resp = MessagingResponse()
        return Response(content=str(resp), media_type="text/xml", status_code=500)

@app.get("/api/sms")
async def get_sms_messages(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Get SMS messages for the current user"""
    try:
        # Get all agent IDs for this user
        user_agent_ids = [a.id for a in db.query(Agent.id).filter(Agent.user_id == user.id).all()]
        
        # Get all SMS messages - use LEFT JOIN to include SMS without agents
        from sqlalchemy import or_
        
        # Build filter condition
        if user_agent_ids:
            filter_condition = or_(
                SMS.agent_id.in_(user_agent_ids),
                SMS.agent_id == None
            )
        else:
            # If user has no agents, only show unassigned SMS
            filter_condition = SMS.agent_id == None
        
        sms_list = db.query(SMS).outerjoin(Agent).filter(filter_condition).order_by(SMS.created_at.desc()).offset(skip).limit(limit).all()
        
        results = []
        for sms in sms_list:
            # Double-check that SMS belongs to user's agent or has no agent
            if sms.agent_id is None or (sms.agent and sms.agent.user_id == user.id):
                results.append({
                    "id": sms.id,
                    "agent_id": sms.agent_id,
                    "agent_name": sms.agent.name if sms.agent else "Unassigned",
                    "from_number": sms.from_number,
                    "to_number": sms.to_number,
                    "body": sms.body,
                    "direction": sms.direction,
                    "message_sid": sms.message_sid,
                    "created_at": sms.created_at.isoformat() if sms.created_at else None
                })
        
        # Count total
        total = db.query(SMS).filter(filter_condition).count()
        
        logger.info(f"📱 Returning {len(results)} SMS messages for user {user.username} (total: {total}, user has {len(user_agent_ids)} agents)")
        
        return {
            "results": results,
            "total": total,
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"❌ Error fetching SMS messages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Models
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

def create_ultravox_agent(api_key: str, system_prompt: str, voice: str, tools: Optional[List] = None):
    url = "https://api.ultravox.ai/api/agents"
    headers = {"X-API-Key": api_key}
    payload = {
        "name": f"Outbound-Agent-{os.urandom(4).hex()}",
        "systemPrompt": system_prompt,
        "model": "fixie-ai/ultravox",
        "voice": voice,
        "firstSpeakerSettings": {"user": {}} # User speaks first for outbound
    }
    if tools:
        payload["tools"] = tools
        
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code != 201:
        raise Exception(f"Failed to create agent: {resp.text}")
    return resp.json()["agentId"]

@app.post("/api/schedule")
async def schedule_calls(req: ScheduleRequest):
    # Config
    ultravox_api_key = os.getenv("ULTRAVOX_API_KEY")
    if not ultravox_api_key:
        raise HTTPException(status_code=500, detail="ULTRAVOX_API_KEY not set")

    twilio_sid = req.twilio_account_sid or os.getenv("TWILIO_ACCOUNT_SID")
    twilio_token = req.twilio_auth_token or os.getenv("TWILIO_AUTH_TOKEN")
    from_number = req.from_number or os.getenv("TWILIO_PHONE_NUMBER")
    
    if not all([twilio_sid, twilio_token, from_number]):
        raise HTTPException(status_code=400, detail="Twilio credentials/number missing")

    try:
        # 1. Create an Agent for this batch
        agent_id = create_ultravox_agent(ultravox_api_key, req.system_prompt, req.voice, req.tools)
        
        # 2. Prepare Calls
        calls = []
        for num in req.to_numbers:
            calls.append({
                "medium": {
                    "twilio": {
                        "outgoing": {
                            "to": num,
                            "from": from_number
                        }
                    }
                }
            })

        # 3. Create Batch
        batch_url = f"https://api.ultravox.ai/api/agents/{agent_id}/scheduled_batches"
        batch_payload = {
            "calls": calls
        }
        if req.window_start:
            batch_payload["windowStart"] = req.window_start
        if req.window_end:
            batch_payload["windowEnd"] = req.window_end
            
        async with httpx.AsyncClient() as client:
            resp = await client.post(batch_url, headers={"X-API-Key": ultravox_api_key}, json=batch_payload)
            if resp.status_code != 201:
                raise Exception(f"Failed to create batch: {resp.text}")
            
            return {"status": "success", "batch_id": resp.json()["batchId"], "agent_id": agent_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ... (previous code)

@app.get("/api/history")
async def get_call_history(
    limit: int = 20, 
    page: int = 1, 
    db: Session = Depends(get_db), 
    user: User = Depends(get_current_user)
):
    # Calculate offset
    offset = (page - 1) * limit
    
    # Query local DB
    total_calls = db.query(Call).filter(Call.user_id == user.id).count()
    calls = db.query(Call).filter(Call.user_id == user.id).order_by(Call.created_at.desc()).offset(offset).limit(limit).all()
    
    # Sync status/duration for calls that look incomplete
    # We do this in background or just await it here? Await for now to be accurate.
    # To avoid too many requests, we only check calls < 24h old that are 'started' or missing duration
    
    async def sync_call(call):
        if call.status == "started" or call.duration is None:
            try:
                api_key = os.getenv("ULTRAVOX_API_KEY")
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"https://api.ultravox.ai/api/calls/{call.ultravox_call_id}", headers={"X-API-Key": api_key})
                    if resp.status_code == 200:
                        data = resp.json()
                        # Update DB
                        was_ended = call.status == "ended"
                        call.status = "ended" if data.get("ended") else "active"
                        if data.get("billedDuration"):
                            dur = data.get("billedDuration")
                            if isinstance(dur, str):
                                dur = float(dur.replace("s", ""))
                            call.duration = int(dur)
                        
                        # If call just ended, check for missed call
                        if data.get("ended") and not was_ended:
                            await check_and_handle_missed_call(call, data, db, logger)
                        
                        return True
            except Exception as e:
                print(f"Sync error for {call.ultravox_call_id}: {e}")
        return False

    # Run sync for needed calls
    tasks = [sync_call(call) for call in calls if call.status == "started" or call.duration is None]
    if tasks:
        await asyncio.gather(*tasks)
        db.commit() # Commit updates

    results = []
    for call in calls:
        # Get Agent Name
        agent_name = "Instant Call"
        if call.agent_id:
            agent = db.query(Agent).filter(Agent.id == call.agent_id).first()
            if agent:
                agent_name = agent.name
        
        results.append({
            "callId": call.ultravox_call_id,
            "created": call.created_at.isoformat(),
            "status": call.status,
            "duration": f"{call.duration}s" if call.duration is not None else "0s",
            "toNumber": call.to_number,
            "fromNumber": call.from_number,
            "agentName": agent_name,
            "direction": call.direction,
            "satisfactionScore": call.satisfaction_score,
            "medium": {"twilio": {"to": call.to_number, "from": call.from_number}} if call.to_number else {},
        })
        
    has_next = (offset + limit) < total_calls
    has_prev = page > 1
    
    return {
        "results": results,
        "pagination": {
            "current_page": page,
            "has_next": has_next,
            "has_prev": has_prev,
            "total_pages": (total_calls + limit - 1) // limit if limit > 0 else 0,
            "total_items": total_calls,
            "next_cursor": None 
        }
    }

@app.get("/api/calls/{call_id}")
async def get_call_details(call_id: str, db: Session = Depends(get_db)):
    api_key = os.getenv("ULTRAVOX_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ULTRAVOX_API_KEY not set")
    
    url = f"https://api.ultravox.ai/api/calls/{call_id}"
    
    try:
        async with httpx.AsyncClient() as client:
            # Fetch basic details
            resp = await client.get(url, headers={"X-API-Key": api_key})
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"Failed to fetch call details: {resp.text}")
            data = resp.json()
        
            # Fetch recording URL using the correct Ultravox API endpoint
            recording_url = None
            try:
                # Try the recording endpoint with follow_redirects=True
                recording_resp = await client.get(f"{url}/recording", headers={"X-API-Key": api_key}, follow_redirects=True)
                print(f"🎵 Recording API response: {recording_resp.status_code}")
                
                if recording_resp.status_code == 200:
                    content_type = recording_resp.headers.get("content-type", "").lower()
                    print(f"🎵 Content-Type: {content_type}")
                    
                    if "application/json" in content_type:
                        try:
                            recording_data = recording_resp.json()
                            print(f"🎵 Recording JSON data: {recording_data}")
                            recording_url = (recording_data.get("url") or 
                                           recording_data.get("recordingUrl") or
                                           recording_data.get("recording_url") or
                                           recording_data.get("downloadUrl") or
                                           recording_data.get("download_url"))
                        except Exception as json_error:
                            print(f"⚠️ JSON parse error: {json_error}")
                            recording_url = None
                    elif "audio" in content_type or "video" in content_type:
                        # Direct audio/video file - use the final URL after redirects
                        recording_url = str(recording_resp.url)
                        print(f"🎵 Direct audio URL: {recording_url}")
                    else:
                        # Try to get URL from response
                        recording_url = str(recording_resp.url)
                        print(f"🎵 Fallback URL: {recording_url}")
                elif recording_resp.status_code == 302:
                    # Handle redirect manually
                    location = recording_resp.headers.get("Location")
                    if location:
                        recording_url = location
                        print(f"🎵 Redirect URL: {recording_url}")
                else:
                    print(f"⚠️ Recording API returned {recording_resp.status_code}: {recording_resp.text}")
                    
            except Exception as e:
                print(f"⚠️ Could not fetch recording: {e}")
            
            if recording_url:
                # Skip validation to avoid potential issues, just set the URL
                data["recordingUrl"] = recording_url
                print(f"✅ Recording URL set: {recording_url[:100]}...")
            else:
                print("❌ No recording URL found")
                # Clear recordingUrl if it exists in data but we couldn't verify/fetch it
                if "recordingUrl" in data:
                    del data["recordingUrl"]
        
            # Fetch messages/transcript
            try:
                msg_resp = await client.get(f"{url}/messages", headers={"X-API-Key": api_key})
                if msg_resp.status_code == 200:
                    messages = msg_resp.json().get("results", [])
                    data["messages"] = messages
                    
                    # Calculate satisfaction score if call is completed
                    if data.get("ended"):
                        transcript = "\n".join([f"{msg['role']}: {msg.get('text', '')}" for msg in messages if msg.get('text')])
                        if transcript:
                            try:
                                duration = data.get("billedDuration", 0)
                                if isinstance(duration, str):
                                    duration = float(duration.replace("s", ""))
                                score = await analyze_satisfaction_ai(transcript, float(duration))
                                data["satisfactionScore"] = score
                                data["satisfactionAnalysis"] = generate_satisfaction_analysis(transcript, score)
                                
                                # Save to DB
                                db_call = db.query(Call).filter(Call.ultravox_call_id == call_id).first()
                                if db_call:
                                    db_call.satisfaction_score = score
                                    was_ended = db_call.status == "ended"
                                    db_call.status = "ended"
                                    db.commit()
                                    
                                    # Check for missed call if call just ended
                                    if data.get("ended") and not was_ended:
                                        await check_and_handle_missed_call(db_call, data, db, logger)
                            except Exception as e:
                                print(f"⚠️ Error calculating satisfaction score: {e}")
                else:
                    print(f"⚠️ Messages API returned {msg_resp.status_code}")
            except Exception as e:
                print(f"⚠️ Could not fetch messages: {e}")
        
            # Add cost information if available
            if "cost" not in data and "billedDuration" in data:
                try:
                    duration = data.get("billedDuration", 0)
                    if isinstance(duration, str):
                        duration = float(duration.replace("s", ""))
                    duration_minutes = float(duration) / 60
                    data["cost"] = round(duration_minutes * 0.05, 4)
                except Exception as e:
                    print(f"⚠️ Error calculating cost: {e}")
                    data["cost"] = 0.0

            # Add local DB info
            db_call = db.query(Call).filter(Call.ultravox_call_id == call_id).first()
            if db_call:
                data["direction"] = db_call.direction
                if db_call.agent:
                    data["agentName"] = db_call.agent.name
                else:
                    data["agentName"] = "Instant Call"
            
            return data
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in get_call_details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/calls/{call_id}/recording-proxy")
async def get_recording_proxy(call_id: str):
    """Proxy endpoint to serve call recordings and avoid CORS issues"""
    api_key = os.getenv("ULTRAVOX_API_KEY")
    
    try:
        async with httpx.AsyncClient() as client:
            # Try to get recording directly
            recording_resp = await client.get(f"https://api.ultravox.ai/api/calls/{call_id}/recording", headers={"X-API-Key": api_key}, follow_redirects=True)
            
            if recording_resp.status_code == 200:
                content_type = recording_resp.headers.get("content-type", "audio/mpeg")
                
                if "audio" in content_type or "video" in content_type:
                    # Stream the audio content directly
                    return Response(
                        content=recording_resp.content,
                        media_type=content_type,
                        headers={
                            "Access-Control-Allow-Origin": "*",
                            "Access-Control-Allow-Methods": "GET",
                            "Access-Control-Allow-Headers": "*",
                        }
                    )
                else:
                    # Try to parse JSON and get URL
                    try:
                        recording_data = recording_resp.json()
                        recording_url = (recording_data.get("url") or 
                                       recording_data.get("recordingUrl") or
                                       recording_data.get("recording_url"))
                        if recording_url:
                            # Fetch the actual audio file
                            audio_resp = await client.get(recording_url)
                            if audio_resp.status_code == 200:
                                return Response(
                                    content=audio_resp.content,
                                    media_type="audio/mpeg",
                                    headers={
                                        "Access-Control-Allow-Origin": "*",
                                        "Access-Control-Allow-Methods": "GET",
                                        "Access-Control-Allow-Headers": "*",
                                    }
                                )
                    except Exception:
                        pass
            
            raise HTTPException(status_code=404, detail="Recording not available")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching recording: {str(e)}")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
