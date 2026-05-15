# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Use SQLite file (change to postgresql://user:pass@host/db if needed)
SQLALCHEMY_DATABASE_URL = "sqlite:///./scheduler.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}  # needed for SQLite
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session (for FastAPI or direct use)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()