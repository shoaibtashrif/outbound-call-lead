"""
User profile and stats routes
"""
from datetime import datetime, timedelta
from typing import Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session
from models import User, Call, UserResponse, Agent
from auth import get_current_user
from config import get_db

router = APIRouter()

@router.get("/api/stats")
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

@router.get("/api/user/profile", response_model=UserResponse)
async def get_user_profile(user: User = Depends(get_current_user)):
    return user

@router.put("/api/user/profile", response_model=UserResponse)
async def update_user_profile(req: Dict[str, Any], db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if "full_name" in req: user.full_name = req["full_name"]
    if "email" in req: user.email = req["email"]
    if "business_type" in req: user.business_type = req["business_type"]
    
    db.commit()
    db.refresh(user)
    return user

@router.get("/api/recent-numbers")
async def get_recent_numbers(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    # Get last 10 unique numbers called by this user
    recent = db.query(Call.to_number).filter(
        Call.user_id == user.id,
        Call.to_number != None
    ).group_by(Call.to_number).order_by(func.max(Call.created_at).desc()).limit(10).all()
    
    return [r[0] for r in recent]