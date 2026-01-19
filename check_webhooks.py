import os
from twilio.rest import Client
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import TwilioNumber

DATABASE_URL = "sqlite:///./outbound_agents_v2.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

def check_webhooks():
    tns = db.query(TwilioNumber).all()
    for tn in tns:
        print(f"Checking number: {tn.phone_number}")
        try:
            client = Client(tn.account_sid, tn.auth_token)
            incoming_phone_numbers = client.incoming_phone_numbers.list(phone_number=tn.phone_number)
            if incoming_phone_numbers:
                ipn = incoming_phone_numbers[0]
                print(f"  Voice URL: {ipn.voice_url}")
                print(f"  Voice Method: {ipn.voice_method}")
            else:
                print("  Number not found in Twilio account.")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    check_webhooks()
    db.close()
