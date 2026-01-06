import os
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Agent, TwilioNumber
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = "sqlite:///./outbound_agents_v2.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

async def sync_all_webhooks():
    db = SessionLocal()
    agents = db.query(Agent).filter(Agent.twilio_number_id != None).all()
    
    host = os.getenv("SERVER_HOST")
    if not host:
        print("❌ SERVER_HOST not set")
        return
        
    if not host.startswith("http"):
        host = f"https://{host}"
        
    path_prefix = "/outbound"
    webhook_url = f"{host}{path_prefix}/api/inbound" if path_prefix not in host else f"{host}/api/inbound"
    
    print(f"🔄 Syncing {len(agents)} agents to webhook: {webhook_url}")
    
    for agent in agents:
        num = agent.twilio_number
        if not num: continue
        
        try:
            client = Client(num.account_sid, num.auth_token)
            incoming_numbers = client.incoming_phone_numbers.list(phone_number=num.phone_number)
            if incoming_numbers:
                incoming_numbers[0].update(voice_url=webhook_url, voice_method='POST')
                print(f"✅ Updated {agent.name} ({num.phone_number})")
            else:
                print(f"⚠️ Number {num.phone_number} not found for {agent.name}")
        except Exception as e:
            print(f"❌ Error for {agent.name}: {e}")
            
    db.close()

if __name__ == "__main__":
    asyncio.run(sync_all_webhooks())
