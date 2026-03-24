from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import datetime

# Standard SQLite database for our registered frontend users
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    threads = relationship("ChatThread", back_populates="owner")

class ChatThread(Base):
    __tablename__ = "chat_threads"
    id = Column(String, primary_key=True, index=True) # Will store LangGraph thread JSON uuid
    title = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    history_json = Column(String, default="[]") # Will physically store the chat memory across restarts
    user_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="threads")

# Automatically create the users table
Base.metadata.create_all(bind=engine)