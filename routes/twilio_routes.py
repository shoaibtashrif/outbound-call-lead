"""
Twilio numbers management routes
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models import User, TwilioNumber, TwilioNumberCreate
from auth import get_current_user
from config import get_db

router = APIRouter()

@router.get("/api/numbers")
async def list_numbers(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    numbers = db.query(TwilioNumber).filter(TwilioNumber.user_id == user.id).all()
    return {"results": numbers}

@router.post("/api/numbers")
async def add_number(req: TwilioNumberCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
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

@router.delete("/api/numbers/{number_id}")
async def delete_number(number_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    db_number = db.query(TwilioNumber).filter(TwilioNumber.id == number_id, TwilioNumber.user_id == user.id).first()
    if not db_number:
        raise HTTPException(status_code=404, detail="Number not found")
    db.delete(db_number)
    db.commit()
    return {"success": True}