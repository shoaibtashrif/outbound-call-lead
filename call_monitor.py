"""
Background task to monitor active calls and handle SMS sending
"""
import os
import asyncio
import logging
import httpx
from sqlalchemy.orm import Session
from models import Call, User
from config import SessionLocal
from sms_service import send_call_summary_sms

logger = logging.getLogger(__name__)

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
                                
                                # Send SMS if data collected and not sent yet
                                result = await send_call_summary_sms(call, db, logger)
                                if result["success"]:
                                    logger.info(f"✅ SMS sent for call {call.id} - SID: {result.get('message_sid')}")
                                else:
                                    logger.warning(f"⚠️ SMS failed for call {call.id} - Reason: {result.get('reason')}")
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