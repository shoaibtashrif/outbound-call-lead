from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import User, Base

engine = create_engine("sqlite:///outbound_agents_v2.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def update_balances():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        for user in users:
            user.balance = 10.0
            print(f"Updated balance for user: {user.username}")
        db.commit()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    update_balances()
