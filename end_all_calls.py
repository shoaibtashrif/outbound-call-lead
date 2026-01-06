#!/usr/bin/env python3
import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

# Get Twilio credentials
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')

if not account_sid or not auth_token:
    print("❌ Twilio credentials not found in .env file")
    exit(1)

client = Client(account_sid, auth_token)

print("🔍 Finding all active calls...")

# Get all in-progress calls
calls = client.calls.list(status='in-progress')

if not calls:
    print("✅ No active calls found")
else:
    print(f"📞 Found {len(calls)} active call(s)")
    for call in calls:
        print(f"\n  Call SID: {call.sid}")
        print(f"  From: {call.from_formatted}")
        print(f"  To: {call.to_formatted}")
        print(f"  Status: {call.status}")
        print(f"  Duration: {call.duration} seconds")
        
        # End the call
        try:
            call.update(status='completed')
            print(f"  ✅ Call ended successfully!")
        except Exception as e:
            print(f"  ❌ Error ending call: {e}")

print("\n✅ All active calls have been ended!")
