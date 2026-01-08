"""
Dashboard and user profile routes
"""
from datetime import datetime, timedelta
from typing import Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session
from models import User, Agent, Call
from auth import get_current_user
from config import get_db

router = APIRouter()

@router.get("/api/dashboard")
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