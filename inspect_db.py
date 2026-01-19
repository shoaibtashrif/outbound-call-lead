from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Agent, TwilioNumber, User

DATABASE_URL = "sqlite:///./outbound_agents_v2.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

print("--- Agents ---")
agents = db.query(Agent).all()
for agent in agents:
    tn = agent.twilio_number
    print(f"Agent ID: {agent.id}, Name: {agent.name}, Voice: {agent.voice}")
    if tn:
        print(f"  Twilio Number: {tn.phone_number}, Account SID: {tn.account_sid}")
    else:
        print("  Twilio Number: NONE (using defaults)")

print("\n--- Twilio Numbers ---")
tns = db.query(TwilioNumber).all()
for tn in tns:
    print(f"ID: {tn.id}, Number: {tn.phone_number}, Account SID: {tn.account_sid}")

db.close()
