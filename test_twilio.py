import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

sid = os.getenv("TWILIO_ACCOUNT_SID")
token = os.getenv("TWILIO_AUTH_TOKEN")
from_number = os.getenv("TWILIO_PHONE_NUMBER")
to_number = os.getenv("DESTINATION_PHONE_NUMBER")

print(f"Testing Twilio Credentials:")
print(f"SID: {sid}")
print(f"Token: {token[:4]}...{token[-4:]} (Length: {len(token)})")
print(f"From: {from_number}")
print(f"To: {to_number}")

try:
    client = Client(sid, token)
    # Try to fetch account details to verify credentials
    account = client.api.accounts(sid).fetch()
    print(f"✅ Credentials Valid! Account Name: {account.friendly_name}")
except Exception as e:
    print(f"❌ Credentials Invalid: {e}")
