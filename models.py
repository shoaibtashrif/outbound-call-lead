from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Table, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List

Base = declarative_base()

# Association table for agent-tool many-to-many relationship
agent_tools = Table('agent_tools', Base.metadata,
    Column('agent_id', Integer, ForeignKey('agents.id')),
    Column('tool_name', String, ForeignKey('tools.name'))
)

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    full_name = Column(String, nullable=True)
    business_type = Column(String, nullable=True)
    subscription_type = Column(String, default="starter")
    hashed_password = Column(String, nullable=False)
    balance = Column(Float, default=10.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    ultravox_api_key = Column(String, nullable=True) # For future use
    
    agents = relationship("Agent", back_populates="user")
    calls = relationship("Call", back_populates="user")

class Call(Base):
    __tablename__ = 'calls'
    
    id = Column(Integer, primary_key=True, index=True)
    ultravox_call_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    agent_id = Column(Integer, ForeignKey('agents.id'), nullable=True)
    to_number = Column(String, nullable=True)
    from_number = Column(String, nullable=True)
    status = Column(String, default="started")
    duration = Column(Integer, nullable=True) # Duration in seconds
    twilio_sid = Column(String, nullable=True)
    direction = Column(String, default="outbound") # "inbound" or "outbound"
    direction = Column(String, default="outbound") # "inbound" or "outbound"
    satisfaction_score = Column(Integer, nullable=True)
    collected_data = Column(Text, nullable=True) # JSON string of data collected by tools
    sms_sent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="calls")
    agent = relationship("Agent")

class TwilioNumber(Base):
    __tablename__ = 'twilio_numbers'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    account_sid = Column(String, nullable=False)
    auth_token = Column(String, nullable=False)
    phone_number = Column(String, nullable=False)
    label = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", backref="twilio_numbers")

class Agent(Base):
    __tablename__ = 'agents'
    
    id = Column(Integer, primary_key=True, index=True)
    ultravox_agent_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    system_prompt = Column(Text)
    voice = Column(String, default="d5594111-ddca-442a-8796-f0fced479a03")
    language = Column(String, default="en")
    model = Column(String, default="fixie-ai/ultravox")
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True) # Nullable for backward compatibility/existing agents
    twilio_number_id = Column(Integer, ForeignKey('twilio_numbers.id'), nullable=True)
    google_spreadsheet_id = Column(String, nullable=True)
    google_sheet_name = Column(String, default="Sheet1")
    google_webhook_url = Column(String, nullable=True)
    transfer_number = Column(String, nullable=True)  # Phone number for call transfer (coldTransfer/warmTransfer)
    
    user = relationship("User", back_populates="agents")
    tools = relationship("Tool", secondary=agent_tools, back_populates="agents")
    twilio_number = relationship("TwilioNumber")

class Tool(Base):
    __tablename__ = 'tools'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    ultravox_tool_name = Column(String)
    description = Column(Text)
    base_url = Column(String)
    http_method = Column(String, default="POST")
    is_builtin = Column(Boolean, default=False)  # Flag for Ultravox built-in tools
    parameter_overrides = Column(Text, nullable=True)  # JSON string for default parameter overrides
    created_at = Column(DateTime, default=datetime.utcnow)
    
    agents = relationship("Agent", secondary=agent_tools, back_populates="tools")

class SMS(Base):
    __tablename__ = 'sms'
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey('agents.id'), nullable=True)
    from_number = Column(String, nullable=False)
    to_number = Column(String, nullable=False)
    body = Column(Text, nullable=False)
    direction = Column(String, default="inbound")  # "inbound" or "outbound"
    message_sid = Column(String, unique=True, index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    agent = relationship("Agent")

# Pydantic models
class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    business_type: Optional[str] = None
    subscription_type: Optional[str] = "starter"

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str]
    full_name: Optional[str]
    business_type: Optional[str]
    subscription_type: Optional[str]
    balance: float
    created_at: datetime
    
    class Config:
        from_attributes = True

class TwilioNumberCreate(BaseModel):
    account_sid: str
    auth_token: str
    phone_number: str
    label: Optional[str] = None

class TwilioNumberResponse(BaseModel):
    id: int
    account_sid: str
    phone_number: str
    label: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class AgentCreate(BaseModel):
    name: str
    system_prompt: str
    voice: Optional[str] = "d5594111-ddca-442a-8796-f0fced479a03"
    language: Optional[str] = "en"
    tool_names: Optional[List[str]] = None
    twilio_number_id: Optional[int] = None
    google_spreadsheet_id: Optional[str] = None
    google_sheet_name: Optional[str] = "Sheet1"
    google_webhook_url: Optional[str] = None
    transfer_number: Optional[str] = None

class AgentResponse(BaseModel):
    id: int
    ultravox_agent_id: Optional[str]
    name: str
    system_prompt: str
    voice: str
    language: str
    model: str
    tools: List[str] = []
    user_id: Optional[int]
    twilio_number_id: Optional[int]
    google_spreadsheet_id: Optional[str]
    google_sheet_name: Optional[str]
    google_webhook_url: Optional[str]
    transfer_number: Optional[str]
    
    class Config:
        from_attributes = True

class SMSResponse(BaseModel):
    id: int
    agent_id: Optional[int]
    agent_name: Optional[str]
    from_number: str
    to_number: str
    body: str
    direction: str
    message_sid: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True
