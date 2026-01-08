#!/usr/bin/env python3
"""
Test script for SMS functionality
"""
import os
import sys
import json
import asyncio
import logging
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from config import SessionLocal, logger
from models import Call, User
from sms_service import send_call_summary_sms

load_dotenv()

async def test_sms_functionality():
    """Test SMS sending with sample data"""
    logger.info("🧪 Starting SMS functionality test...")
    
    # Check Twilio credentials
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    sender_number = os.getenv("TWILIO_PHONE_NUMBER")
    
    logger.info(f"📋 Twilio Config Check:")
    logger.info(f"  Account SID: {account_sid[:10]}..." if account_sid else "  Account SID: ❌ Missing")
    logger.info(f"  Auth Token: {auth_token[:10]}..." if auth_token else "  Auth Token: ❌ Missing")
    logger.info(f"  Phone Number: {sender_number}" if sender_number else "  Phone Number: ❌ Missing")
    
    if not all([account_sid, auth_token, sender_number]):
        logger.error("❌ Missing Twilio credentials!")
        return False
    
    # Test with a sample call record
    db = SessionLocal()
    try:
        # Find a recent call with collected data
        test_call = db.query(Call).filter(
            Call.collected_data.isnot(None),
            Call.to_number.isnot(None)
        ).order_by(Call.created_at.desc()).first()
        
        if not test_call:
            logger.warning("⚠️ No calls with collected data found. Creating test data...")
            
            # Create a test call record
            test_data = {
                "name": "John Doe",
                "email": "john@example.com", 
                "phone": "+1234567890",
                "interest": "Product Demo"
            }
            
            test_call = Call(
                ultravox_call_id="test-call-123",
                user_id=1,  # Assuming user ID 1 exists
                to_number=sender_number,  # Send to our own number for testing
                from_number=sender_number,
                status="ended",
                duration=120,
                collected_data=json.dumps(test_data),
                sms_sent=False
            )
            db.add(test_call)
            db.commit()
            db.refresh(test_call)
            logger.info(f"✅ Created test call record: {test_call.id}")
        
        logger.info(f"🧪 Testing SMS with call ID: {test_call.id}")
        logger.info(f"📞 Recipient: {test_call.to_number}")
        logger.info(f"📋 Data: {test_call.collected_data}")
        
        # Reset SMS sent flag for testing
        test_call.sms_sent = False
        db.commit()
        
        # Send SMS
        await send_call_summary_sms(test_call, db, logger)
        
        # Check if SMS was marked as sent
        db.refresh(test_call)
        if test_call.sms_sent:
            logger.info("✅ SMS test completed successfully!")
            return True
        else:
            logger.error("❌ SMS was not marked as sent")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    finally:
        db.close()

async def test_direct_twilio():
    """Test Twilio directly without database"""
    logger.info("🧪 Testing direct Twilio SMS...")
    
    try:
        from twilio.rest import Client
        
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        sender_number = os.getenv("TWILIO_PHONE_NUMBER")
        
        client = Client(account_sid, auth_token)
        
        message = client.messages.create(
            body="🧪 Test SMS from Outbound Call Service - SMS functionality is working!",
            from_=sender_number,
            to=sender_number  # Send to self for testing
        )
        
        logger.info(f"✅ Direct Twilio test successful!")
        logger.info(f"📨 Message SID: {message.sid}")
        logger.info(f"📱 Status: {message.status}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Direct Twilio test failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--direct":
        # Test Twilio directly
        result = asyncio.run(test_direct_twilio())
    else:
        # Test full SMS functionality
        result = asyncio.run(test_sms_functionality())
    
    if result:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("💥 Tests failed!")
        sys.exit(1)