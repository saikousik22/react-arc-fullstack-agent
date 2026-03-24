# Full-Stack LangGraph ReAct Agent

A production-ready, full-stack application demonstrating a complex LangGraph **ReAct (Reasoning and Acting)** architecture powered by **Gemini 2.5 Flash**.

This application acts like a ChatGPT clone on the frontend but utilizes an intelligent LangGraph backend capable of reasoning through multi-part questions, utilizing tools natively, and securely isolating conversation memory per user!

## ✨ Features
- **ChatGPT-Style Frontend**: A modern, responsive "Glassmorphism" UI built from scratch in vanilla HTML/CSS/JS.
- **Dynamic Multi-Thread Memory**: Users can create dozens of separate chats. FastAPI dynamically spins up independent `thread_id` states.
- **Persistent AI Memory**: LangGraph's native `MemorySaver` checkpointer is manually flushed to SQLite so your entire chat history perfectly survives server restarts!
- **JWT Authentication**: Full user registration, login routing, and memory isolation.
- **Intelligent Tool Use**: Gemini defaults to utilizing its massive native knowledge for math, trivia, and coding, but selectively triggers tools (like Tavily web live search) when information runs out.
- **LLM Auto-Titling**: Gemini immediately reads your first message in a new thread and automatically names your chat in the sidebar.

<br>

## 🚀 Setup & Installation Instructions

Follow these steps to run the agent locally on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/saikousik22/react-arc-fullstack-agent.git
cd react-arc-fullstack-agent
```

### 2. Configure Environment Variables
The application strictly depends on Google Gemini and Tavily Search APIs. 
Create a file specifically named `.env` in the root folder of the project, and add your API keys:

```env
GOOGLE_API_KEY="your-google-gemini-key-here"
TAVILY_API_KEY="your-tavily-search-key-here"
```

### 3. Install Dependencies
Make sure you have Python installed. Install all the necessary packages (FastAPI, SQLAlchemy, LangGraph, etc.) using `pip`:
```bash
pip install -r requirements.txt
```

### 4. Run the Full-Stack Agent
Start the FastAPI server utilizing `uvicorn`. The API automatically routes both your frontend HTML elements and LangGraph execution securely.

```bash
uvicorn server:app --reload
```

### 5. Start Chatting!
Open your browser and navigate to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

Register a quick user account, click **"+ New Chat"** on the left sidebar, and ask Gemini a deeply complex question! 

> **Tip:** Keep an eye on the terminal where `uvicorn` is running while you type. I have explicitly configured `agent.py` to stream its direct underlying reasoning and tool execution blocks directly to the terminal so you can trace the agent's "thoughts" in real-time!