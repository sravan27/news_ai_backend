"""
Database session management for the News AI application.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
import os
from pathlib import Path

# Get database URL from environment or use SQLite default
DATABASE_URL = os.environ.get(
    "DATABASE_URL", 
    f"sqlite:///{Path(__file__).resolve().parent.parent}/data/news_ai.db"
)

# Create engine
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_session = scoped_session(SessionLocal)

@contextmanager
def get_db():
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

# For FastAPI dependency
def get_db_session():
    """FastAPI dependency for database sessions."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()