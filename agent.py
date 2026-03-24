import os
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# ==========================================
# 1. Define Tools
# ==========================================
tavily_tool = TavilySearchResults(max_results=2)

class AddInput(BaseModel):
    a: int = Field(description="First integer")
    b: int = Field(description="Second integer")

@tool("add", args_schema=AddInput)
def add(a: int, b: int) -> int:
    """Add two integer numbers."""
    print(f"\n[SYSTEM LOG] 🛠️ Executing tool 'add' with arguments: a={a}, b={b}")
    return a + b

class SubtractInput(BaseModel):
    a: int = Field(description="Base integer")
    b: int = Field(description="Integer to subtract")

@tool("subtract", args_schema=SubtractInput)
def subtract(a: int, b: int) -> int:
    """Subtract the second integer number from the first integer number."""
    print(f"\n[SYSTEM LOG] 🛠️ Executing tool 'subtract' with arguments: a={a}, b={b}")
    return a - b

tools = [tavily_tool, add, subtract]
tool_node = ToolNode(tools)

# ==========================================
# 2. Define Model
# ==========================================
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)

from langchain_core.messages import SystemMessage

def call_model(state: MessagesState):
    messages = state['messages']
    
    sys_prompt = SystemMessage(
        content=(
            "You are a highly capable AI assistant doing your best to answer every part of the user's prompt.\n"
            "Follow these rules to maximize your helpfulness:\n"
            "1. Try to answer as much of the question as possible using your massive built-in knowledge native reasoning (math, coding, logic, trivia) WITHOUT tools.\n"
            "2. If a specific part of the user's prompt requires up-to-date web information or explicit calculation (e.g. testing the add/subtract function explicitly), DO NOT give up. Immediately call the relevant tools to gather that specific missing information.\n"
            "3. If a question has multiple parts, answer the parts you know natively, use tools for the parts you don't know, and combine them into a single, cohesive final answer.\n"
            "4. NEVER say 'I only have tools for X' or complain about missing tools. Just solve whatever you can intelligently and fallback politely if absolutely impossible."
        )
    )
    
    print("\n[LLM NODE] Gemini is reasoning...")
    full_messages = [sys_prompt] + messages
    response = model.invoke(full_messages)
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"[LLM DECISION] Gemini decided to call tools: {response.tool_calls}")
    else:
        print("[LLM DECISION] Gemini decided to return a final answer directly.")
        
    return {"messages": [response]}

def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

# ==========================================
# 3. Build Graph
# ==========================================
workflow = StateGraph(MessagesState)
workflow.add_node("tool_calling_llm", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge("__start__", "tool_calling_llm")
workflow.add_conditional_edges("tool_calling_llm", should_continue)
workflow.add_edge("tools", "tool_calling_llm")

# Extremely fast internal memory server that holds active sessions
memory = MemorySaver()

# Compile and export the agent for FastAPI to use
agent_app = workflow.compile(checkpointer=memory)