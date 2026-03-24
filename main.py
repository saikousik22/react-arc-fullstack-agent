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
# 1. Define Tools (The "Acting" part)
# ==========================================

# 1. Tavily Search Tool for web searching
tavily_tool = TavilySearchResults(max_results=2)

# 2. Basic Addition Tool
class AddInput(BaseModel):
    a: int = Field(description="The first integer number to add")
    b: int = Field(description="The second integer number to add")

@tool("add", args_schema=AddInput)
def add(a: int, b: int) -> int:
    """Add two integer numbers."""
    print(f"\n[SYSTEM LOG] 🛠️  Executing tool 'add' with arguments: a={a}, b={b}")
    return a + b

# 3. Basic Subtraction Tool
class SubtractInput(BaseModel):
    a: int = Field(description="The base integer number")
    b: int = Field(description="The integer number to subtract from the base number (a - b)")

@tool("subtract", args_schema=SubtractInput)
def subtract(a: int, b: int) -> int:
    """Subtract the second integer number from the first integer number (a - b)."""
    print(f"\n[SYSTEM LOG] 🛠️  Executing tool 'subtract' with arguments: a={a}, b={b}")
    return a - b

# Combine tools into a list
tools = [tavily_tool, add, subtract]

# Create the node that LangGraph uses to execute our tools programmatically
tool_node = ToolNode(tools)

# ==========================================
# 2. Define the Model (The "Reasoning" part)
# ==========================================
# We use ChatGoogleGenerativeAI (Gemini) and bind the tools to it.
try:
    # "gemini-1.5-flash" is fast, cost-effective, and fully supports tool-calling
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)
except Exception as e:
    print(f"Error initializing model: {e}")
    print("Please make sure you have set your GOOGLE_API_KEY in the .env file.")
    exit(1)

# ==========================================
# 3. Define Graph Nodes and Edges
# ==========================================

# Node: The Agent
def call_model(state: MessagesState):
    """This node invokes the LLM to decide what to do next."""
    print("\n[LLM NODE] Gemini is reasoning...")
    messages = state['messages']
    response = model.invoke(messages)
    
    # Let's explicitly log the LLM's raw decision!
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"[LLM DECISION] Gemini decided to call tools automatically: {response.tool_calls}")
    else:
        print(f"[LLM DECISION] Gemini decided to return a final answer.")
        
    # Return the new message to be appended to the state
    return {"messages": [response]}

# Conditional Edge: Router
def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """
    This function acts as a router. It checks the last message from the model.
    If the model decided to call a tool, we route to the "tools" node.
    Otherwise, we route to the __end__ node to finish the execution.
    """
    messages = state['messages']
    last_message = messages[-1]
    
    # Check if the model made a tool call
    if last_message.tool_calls:
        return "tools"
    
    # If no tool calls, the agent has finished reasoning and formed a final answer
    return "__end__"

# ==========================================
# 4. Build the LangGraph
# ==========================================
workflow = StateGraph(MessagesState)

# Add the nodes (Notice the explicit "tool_calling_llm" node name you requested!)
workflow.add_node("tool_calling_llm", call_model)
workflow.add_node("tools", tool_node)

# Add the edges to connect the nodes in a loop (ReAct loop)
workflow.add_edge("__start__", "tool_calling_llm")

# The agent either goes to tools or finishes
workflow.add_conditional_edges("tool_calling_llm", should_continue)

# The tools always go back to the agent for the next step of reasoning
workflow.add_edge("tools", "tool_calling_llm")

# Initialize memory to persist state between graph runs
memory = MemorySaver()

# Compile the graph into an executable app with memory
app = workflow.compile(checkpointer=memory)

# ==========================================
# 5. Run the demonstration
# ==========================================
def ask_question(question: str, thread_id: str):
    """Helper function to run a query with a specific memory thread."""
    print(f"\nUser Query: {question}")
    
    inputs = {"messages": [HumanMessage(content=question)]}
    # Pass the thread ID inside the config! This retrieves the previous memory for this thread.
    config = {"configurable": {"thread_id": thread_id}}
    
    # Stream the steps
    for event in app.stream(inputs, config=config, stream_mode="updates"):
        for node_name, node_state in event.items():
            print(f"\n=========================================")
            print(f"🔄 GRAPH ROUTED TO NODE: '{node_name}'")
            print(f"=========================================")
            
            # Print the latest message generated by the node
            latest_message = node_state["messages"][-1]
            
            if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                print(f"🤖 LLM requested the following actions to be run:")
                for tool_call in latest_message.tool_calls:
                    print(f"  -> CALL TOOL: '{tool_call['name']}' | ARGS: {tool_call['args']}")
            
            elif hasattr(latest_message, 'name') and node_name == 'tools':
                print(f"🛠️ TOOL EXECUTION COMPLETED: '{latest_message.name}'")
                print(f"  -> RAW RESULT RETURNED TO LLM: {latest_message.content}")
                
            else:
                print(f"✅ FINAL CONVERSATIONAL ANSWER FROM LLM:\n{latest_message.content}")

def main():
    print("🤖 LangGraph ReAct Architecture Demonstration [Powered by Gemini]")
    print("================================================================")
    
    # Thread ID represents a unique user session.
    # Everything run with this ID will share the same memory list.
    thread_id = "user_session_1"
    
    print("\n--- [CONVERSATION TURN 1: Building Context] ---")
    question_1 = "Search for the year Google was founded. Then, add 20 to that year. Finally, subtract 5 from the result."
    ask_question(question_1, thread_id)
    
    print("\n\n--- [CONVERSATION TURN 2: Testing Memory Layer] ---")
    question_2 = "What was the final math result we just got, and what did we subtract from?"
    ask_question(question_2, thread_id)

if __name__ == "__main__":
    # Ensure API keys are set before running
    missing_keys = []
    if not os.environ.get("GOOGLE_API_KEY"):
        missing_keys.append("GOOGLE_API_KEY")
    if not os.environ.get("TAVILY_API_KEY"):
        missing_keys.append("TAVILY_API_KEY")
        
    if missing_keys:
        print(f"⚠️  Warning: Missing API Keys: {', '.join(missing_keys)}")
        print("Please configure them in your .env file or environment variables.")
    else:
        main()