"""
SMS service for sending call summaries
"""
import os
import json
import logging
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from sqlalchemy.orm import Session
from models import Call

logger = logging.getLogger(__name__)

def format_sms_message(collected_data: dict, call: Call) -> str:
    """Format collected data into a readable SMS message"""
    msg_body = "📞 Call Summary:\n"
    
    # Process collected data
    for key, value in collected_data.items():
        # Clean up the key name for better readability
        clean_key = key.replace('_', ' ').replace('-', ' ').title()
        
        # Truncate long values
        if isinstance(value, str) and len(value) > 50:
            value = value[:47] + "..."
        
        msg_body += f"• {clean_key}: {value}\n"
    
    # Add call details
    if call.duration:
        minutes = call.duration // 60
        seconds = call.duration % 60
        msg_body += f"• Duration: {minutes}m {seconds}s\n"
    
    msg_body += f"• Call ID: {call.id}\n"
    msg_body += "Thank you for your time!"
    
    # Ensure message is within SMS limits (160 chars for single SMS, 1600 for concatenated)
    if len(msg_body) > 1500:
        msg_body = msg_body[:1497] + "..."
    
    return msg_body

async def send_call_summary_sms(call: Call, db: Session, logger_instance=None):
    """Send SMS with collected call data"""
    if logger_instance is None:
        logger_instance = logger
    
    # Pre-flight checks
    if not call.collected_data:
        logger_instance.info(f"📝 No data collected for call {call.id}, skipping SMS")
        return {"success": False, "reason": "no_data"}
    
    if call.sms_sent:
        logger_instance.info(f"📨 SMS already sent for call {call.id}")
        return {"success": False, "reason": "already_sent"}
    
    if not call.to_number:
        logger_instance.warning(f"⚠️ No recipient number for call {call.id}")
        return {"success": False, "reason": "no_recipient"}

    # Get Twilio credentials
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    sender_number = os.getenv("TWILIO_PHONE_NUMBER")
    
    if not all([account_sid, auth_token, sender_number]):
        logger_instance.error("❌ Missing Twilio credentials in environment variables")
        logger_instance.error(f"   Account SID: {'✅' if account_sid else '❌'}")
        logger_instance.error(f"   Auth Token: {'✅' if auth_token else '❌'}")
        logger_instance.error(f"   Phone Number: {'✅' if sender_number else '❌'}")
        return {"success": False, "reason": "missing_credentials"}

    try:
        # Parse collected data
        try:
            collected = json.loads(call.collected_data)
            logger_instance.info(f"📋 Collected data for call {call.id}: {collected}")
        except json.JSONDecodeError as e:
            logger_instance.error(f"❌ Invalid JSON in collected_data for call {call.id}: {e}")
            return {"success": False, "reason": "invalid_json", "error": str(e)}
        
        # Build SMS message
        msg_body = format_sms_message(collected, call)
        recipient = call.to_number
        
        logger_instance.info(f"📨 Preparing SMS for call {call.id}")
        logger_instance.info(f"📞 From: {sender_number}")
        logger_instance.info(f"📞 To: {recipient}")
        logger_instance.info(f"📝 Message length: {len(msg_body)} characters")
        logger_instance.info(f"📝 SMS Content:\n{msg_body}")
        
        # Initialize Twilio client
        twilio_client = Client(account_sid, auth_token)
        
        # Send SMS
        logger_instance.info("🚀 Sending SMS via Twilio...")
        message = twilio_client.messages.create(
            body=msg_body,
            from_=sender_number,
            to=recipient
        )
        
        # Mark as sent in database
        call.sms_sent = True
        db.commit()
        
        logger_instance.info(f"✅ SMS sent successfully!")
        logger_instance.info(f"📨 Message SID: {message.sid}")
        logger_instance.info(f"📱 Status: {message.status}")
        logger_instance.info(f"📞 From: {message.from_}")
        logger_instance.info(f"📞 To: {message.to}")
        logger_instance.info(f"💰 Price: {message.price} {message.price_unit}" if message.price else "💰 Price: Pending")
        
        return {
            "success": True, 
            "message_sid": message.sid,
            "status": message.status,
            "message_length": len(msg_body)
        }
        
    except TwilioRestException as e:
        error_msg = str(e)
        logger_instance.error(f"❌ Twilio API error for call {call.id}: {error_msg}")
        logger_instance.error(f"   Error Code: {e.code}")
        logger_instance.error(f"   Error Status: {e.status}")
        
        # Handle specific Twilio errors
        should_mark_sent = False
        
        if e.code == 21614:  # 'To' number is not a valid mobile number
            logger_instance.warning(f"⚠️ Invalid mobile number {recipient}, marking as sent")
            should_mark_sent = True
        elif e.code == 21408:  # Permission to send an SMS has not been enabled
            logger_instance.warning(f"⚠️ SMS not enabled for region {recipient}, marking as sent")
            should_mark_sent = True
        elif e.code == 21606:  # The From phone number is not SMS-capable
            logger_instance.warning(f"⚠️ Sender number {sender_number} is not SMS-capable")
            should_mark_sent = True
        elif "not enabled for the region" in error_msg.lower():
            logger_instance.warning(f"⚠️ SMS not enabled for region {recipient}, marking as sent")
            should_mark_sent = True
        
        if should_mark_sent:
            call.sms_sent = True
            db.commit()
            logger_instance.info("✅ Marked as sent to avoid future retry attempts")
        
        return {
            "success": False, 
            "reason": "twilio_error",
            "error": error_msg,
            "code": e.code,
            "marked_sent": should_mark_sent
        }
        
    except Exception as e:
        error_msg = str(e)
        logger_instance.error(f"❌ Unexpected error sending SMS for call {call.id}: {error_msg}")
        logger_instance.error(f"   Error type: {type(e).__name__}")
        
        # Don't mark as sent for unknown errors to allow retry
        return {
            "success": False,
            "reason": "unknown_error", 
            "error": error_msg
        }

def test_twilio_credentials():
    """Test Twilio credentials and SMS capability"""
    logger.info("🧪 Testing Twilio credentials...")
    
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    sender_number = os.getenv("TWILIO_PHONE_NUMBER")
    
    if not all([account_sid, auth_token, sender_number]):
        logger.error("❌ Missing Twilio credentials")
        return False
    
    try:
        client = Client(account_sid, auth_token)
        
        # Test account access
        account = client.api.accounts(account_sid).fetch()
        logger.info(f"✅ Account access successful: {account.friendly_name}")
        
        # Test phone number capabilities
        number = client.incoming_phone_numbers.list(phone_number=sender_number)
        if number:
            capabilities = number[0].capabilities
            logger.info(f"📞 Number capabilities: SMS={capabilities.get('sms', False)}, Voice={capabilities.get('voice', False)}")
            return capabilities.get('sms', False)
        else:
            logger.warning(f"⚠️ Phone number {sender_number} not found in account")
            return False
            
    except Exception as e:
        logger.error(f"❌ Credential test failed: {e}")
        return False