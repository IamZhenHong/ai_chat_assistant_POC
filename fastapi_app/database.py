from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import sqlalchemy


# Load environment variables from .env file
load_dotenv()

# Get Supabase PostgreSQL URL
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL and Key are required.")

# Construct the PostgreSQL connection URL (Supabase)
DATABASE_URL = f"postgresql://postgres:{SUPABASE_KEY}@db.{SUPABASE_URL.split('//')[1]}:5432/postgres"

# ✅ Create the SQLAlchemy Engine (Connect to Supabase)
engine = create_engine(DATABASE_URL)

# ✅ Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ✅ Define the Base model class
Base = sqlalchemy.orm.declarative_base()

# ✅ Function to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
