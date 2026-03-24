from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from pydantic import BaseModel
import os

from langchain_core.messages import HumanMessage, AIMessage

# Import our custom files
import db
from agent import agent_app

# Security configuration
SECRET_KEY = "super-secret-key-for-interview-demo"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="ReAct Agent API")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup HTML templates
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

# DB Dependency
def get_db():
    db_session = db.SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

# API Schemas
class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str
    password: str

import uuid

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ThreadCreate(BaseModel):
    title: str

# Auth Utilities
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), session: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = session.query(db.User).filter(db.User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# =======================
# API ENDPOINTS
# =======================

@app.get("/")
def serve_frontend(request: Request):
    """Serves the beautiful full-stack UI"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/register")
def register(user: UserCreate, session: Session = Depends(get_db)):
    """Registers a new user in the SQLite db"""
    db_user = session.query(db.User).filter(db.User.username == user.username).first()
    db_email = session.query(db.User).filter(db.User.email == user.email).first()
    if db_user or db_email:
        raise HTTPException(status_code=400, detail="Username or Email already registered")
    
    hashed_pw = get_password_hash(user.password)
    new_user = db.User(
        username=user.username, 
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_pw
    )
    session.add(new_user)
    session.commit()
    return {"message": "User created successfully"}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_db)):
    """Authenticates the user and gives them a JWT Token"""
    user = session.query(db.User).filter(db.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/threads")
def create_thread(thread: ThreadCreate, session: Session = Depends(get_db), current_user: db.User = Depends(get_current_user)):
    """Creates a new chat session for the user"""
    new_id = str(uuid.uuid4())
    new_thread = db.ChatThread(id=new_id, title=thread.title, user_id=current_user.id)
    session.add(new_thread)
    session.commit()
    return {"id": new_id, "title": new_thread.title}

@app.get("/threads")
def get_threads(session: Session = Depends(get_db), current_user: db.User = Depends(get_current_user)):
    """Lists all chat sessions belonging to the user"""
    threads = session.query(db.ChatThread).filter(db.ChatThread.user_id == current_user.id).all()
    return [{"id": t.id, "title": t.title} for t in threads]

@app.post("/chat")
def chat(request: ChatRequest, session: Session = Depends(get_db), current_user: db.User = Depends(get_current_user)):
    """Handles chat logic and securely routes it to LangGraph ReAct Agent"""
    # Verify ownership
    user_thread = session.query(db.ChatThread).filter(db.ChatThread.id == request.thread_id, db.ChatThread.user_id == current_user.id).first()
    if not user_thread:
        raise HTTPException(status_code=403, detail="Thread not found or unauthorized")
        
    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = {"messages": [HumanMessage(content=request.message)]}
    
    try:
        new_title = None
        if user_thread.title == "New Conversation":
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Start up a fast model just for the title
            title_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
            title_res = title_llm.invoke(f"Write a maximum 2 word title for this prompt. NO quotes, NO extra words. Prompt: '{request.message}'")
            new_title = str(title_res.content).strip().replace('"', '')
            user_thread.title = new_title

        # Run the graph
        result_state = agent_app.invoke(inputs, config=config)
        final_message = str(result_state["messages"][-1].content)
        
        # Manually save the graph state into SQLite JSON to avoid serialization errors
        state = agent_app.get_state(config)
        messages_payload = []
        if hasattr(state, "values") and isinstance(state.values, dict) and "messages" in state.values:
            for msg in state.values["messages"]:
                m_type = getattr(msg, "type", "unknown")
                m_content = getattr(msg, "content", "")
                messages_payload.append({"type": m_type, "content": str(m_content)})
        
        import json
        user_thread.history_json = json.dumps(messages_payload)
        session.commit()
        
        return {"response": final_message, "new_title": new_title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history(thread_id: str, session: Session = Depends(get_db), current_user: db.User = Depends(get_current_user)):
    """Retrieves the full chat history strictly isolated to the logged-in user and specific thread"""
    user_thread = session.query(db.ChatThread).filter(db.ChatThread.id == thread_id, db.ChatThread.user_id == current_user.id).first()
    if not user_thread:
        raise HTTPException(status_code=403, detail="Thread not found")
        
    config = {"configurable": {"thread_id": thread_id}}
    import json
    
    # Fetch memory directly from LangGraph internal memory
    state = agent_app.get_state(config)
    
    # If the memory server is empty (e.g., server just restarted), load the JSON from our custom SQLite database!
    if not hasattr(state, "values") or not isinstance(state.values, dict) or "messages" not in state.values or not state.values["messages"]:
        if user_thread.history_json and user_thread.history_json != "[]":
            saved_messages = json.loads(user_thread.history_json)
            restored_msgs = []
            for m in saved_messages:
                if m["type"] == "human":
                    restored_msgs.append(HumanMessage(content=m["content"]))
                elif m["type"] == "ai":
                    restored_msgs.append(AIMessage(content=m["content"]))
            if restored_msgs:
                # Instantly seed the LangGraph MemorySaver so it remembers!
                agent_app.update_state(config, {"messages": restored_msgs})
                state = agent_app.get_state(config)
    
    chat_history = []
    if hasattr(state, "values") and isinstance(state.values, dict) and "messages" in state.values:
        for msg in state.values["messages"]:
            if msg.type == "human":
                chat_history.append({"role": "user", "content": str(msg.content)})
            elif msg.type == "ai" and msg.content:
                chat_history.append({"role": "assistant", "content": str(msg.content)})
                
    return {"history": chat_history}