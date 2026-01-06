import os
import asyncio
import json
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


load_dotenv()

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
    while True:
        await asyncio.sleep(60)
        db = SessionLocal()
        try:
            # Find active calls in our DB
            active_calls = db.query(Call).filter(Call.status.in_(["started", "active"])).all()
            api_key = os.getenv("ULTRAVOX_API_KEY")
            
            for call in active_calls:
                user = db.query(User).filter(User.id == call.user_id).first()
                if not user: continue
                
                # Check actual status from Ultravox first
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(f"https://api.ultravox.ai/api/calls/{call.ultravox_call_id}", headers={"X-API-Key": api_key})
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("ended"):
                                call.status = "ended"
                                if data.get("billedDuration"):
                                    dur = data.get("billedDuration")
                                    if isinstance(dur, str): dur = float(dur.replace("s", ""))
                                    call.duration = int(dur)
                                db.commit()
                                continue
                except: pass

                # If still active, deduct cost
                cost_per_min = 0.05
                user.balance -= cost_per_min
                
                if user.balance <= 0:
                    user.balance = 0
                    # Terminate call
                    try:
                        async with httpx.AsyncClient() as client:
                            await client.post(f"https://api.ultravox.ai/api/calls/{call.ultravox_call_id}/end", headers={"X-API-Key": api_key})
                        call.status = "ended"
                    except Exception as e:
                        print(f"Error terminating call {call.ultravox_call_id}: {e}")
                
                db.commit()
        except Exception as e:
            print(f"Monitor error: {e}")
        finally:
            db.close()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(monitor_calls_and_balance())

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
            "selectedTools": [{"toolName": t.name} for t in agent.tools],
            "created": agent.created_at.isoformat() if agent.created_at else None
        })
    return {"results": results}

@app.post("/api/agents")
async def create_agent(req: AgentCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):

    # Create agent in database
    db_agent = Agent(
        name=req.name,
        system_prompt=req.system_prompt,
        voice=req.voice,
        language=req.language,
        user_id=user.id,
        twilio_number_id=req.twilio_number_id
    )
    
    # Add tools if provided
    if req.tool_names:
        tools = db.query(Tool).filter(Tool.name.in_(req.tool_names)).all()
        db_agent.tools = tools
    
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)
    
    return {
        "agentId": str(db_agent.id),
        "name": db_agent.name,
        "systemPrompt": db_agent.system_prompt,
        "model": db_agent.model,
        "voice": db_agent.voice,
        "languageHint": db_agent.language,
        "twilio_number_id": db_agent.twilio_number_id,
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
    
    db_agent.name = req.name
    db_agent.system_prompt = req.system_prompt
    db_agent.voice = req.voice
    db_agent.language = req.language
    db_agent.twilio_number_id = req.twilio_number_id
    
    if req.tool_names is not None:
        tools = db.query(Tool).filter(Tool.name.in_(req.tool_names)).all()
        db_agent.tools = tools
        
    db.commit()
    db.refresh(db_agent)
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
async def list_tools():
    api_key = os.getenv("ULTRAVOX_API_KEY")
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.ultravox.ai/api/tools", headers={"X-API-Key": api_key})
        data = resp.json()
        if isinstance(data, list):
            return {"results": data}
        return data

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
            twilio_sid=call.sid
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
            twilio_sid=call.sid
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
                        call.status = "ended" if data.get("ended") else "active"
                        if data.get("billedDuration"):
                            dur = data.get("billedDuration")
                            if isinstance(dur, str):
                                dur = float(dur.replace("s", ""))
                            call.duration = int(dur)
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
                                    db_call.status = "ended"
                                    db.commit()
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
                        # Handle "12s" format if applicable, or just parse string
                        duration = float(duration.replace("s", ""))
                    duration_minutes = float(duration) / 60
                    data["cost"] = round(duration_minutes * 0.05, 4)
                except Exception as e:
                    print(f"⚠️ Error calculating cost: {e}")
                    data["cost"] = 0.0
            
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
