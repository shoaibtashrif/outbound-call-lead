from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Agent
import os

DATABASE_URL = "sqlite:///./outbound_agents_v2.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

bad_voice_id = "a656a751-b754-4621-b571-e1298cb7e5bb"
good_voice_id = "d5594111-ddca-442a-8796-f0fced479a03"

print("Checking for agents with bad or empty voice ID...")
agents = db.query(Agent).filter(
    (Agent.voice == bad_voice_id) | 
    (Agent.voice == "") | 
    (Agent.voice == None)
).all()

if agents:
    print(f"Found {len(agents)} agents with bad voice ID.")
    for agent in agents:
        print(f"Updating agent {agent.name} (ID: {agent.id})")
        agent.voice = good_voice_id
    db.commit()
    print("Updated all agents.")
else:
    print("No agents found with bad voice ID.")

db.close()
