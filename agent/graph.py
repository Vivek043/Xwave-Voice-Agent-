"""
LangGraph Agent — Stateful AI Agent with Conditional Routing

What you learn here:
- LangGraph state machines (nodes + edges = agent graph)
- Conditional routing based on sentiment (key skill for Xwave-style agents)
- Tool calling via LangChain
- Automatic tracing via LangSmith (just set the env vars — it's automatic!)

Architecture:
    START
      │
      ▼
  [analyze]  ←── Reads sentiment + conversation history
      │
      ▼
  [route] ─── if escalate → [escalate] → END
      │
      ▼ else
  [respond]  ←── Calls Ollama LLM (or Groq fallback)
      │
      ▼
    END
"""

import os
import logging
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, START

from agent.prompts import get_prompt_for_sentiment

load_dotenv()
logger = logging.getLogger(__name__)

# ── LLM Configuration ─────────────────────────────────────────────────────────

def get_local_llm():
    """Primary: Ollama running locally. Unlimited, free."""
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.7,
    )

def get_groq_llm():
    """Fallback: Groq cloud (free tier, very fast)."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",  # Free tier model
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=512,
    )


# ── Agent State Schema ─────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The shared state that flows through every node in the graph.
    Think of it as the agent's working memory for one conversation turn.
    """
    # Inputs
    session_id: str
    user_message: str
    sentiment: dict          # From sentiment_service
    turn_number: int
    conversation_history: list  # Previous turns from DB

    # Internal
    should_escalate: bool
    negative_turn_count: int
    system_prompt: str

    # Outputs
    agent_response: str
    model_used: str
    action: str              # "respond" | "escalate"


# ── Graph Nodes ────────────────────────────────────────────────────────────────

def analyze_node(state: AgentState) -> AgentState:
    """
    Node 1: Analyze the current state.
    Decides whether to escalate based on sentiment + history.
    """
    sentiment = state["sentiment"]
    negative_turns = state.get("negative_turn_count", 0)
    threshold = float(os.getenv("SENTIMENT_ESCALATION_THRESHOLD", "-0.5"))
    escalation_turn_count = int(os.getenv("ESCALATION_TURN_COUNT", "3"))

    # Escalation logic: sentiment too negative AND pattern persists
    hard_escalate = (
        sentiment.get("escalate_flag", False)
        and negative_turns >= escalation_turn_count
    )

    # Build the right system prompt based on state
    system_prompt = get_prompt_for_sentiment(
        sentiment_label=sentiment.get("label", "NEUTRAL"),
        sentiment_score=sentiment.get("score", 0.0),
        negative_turns=negative_turns,
        turn_number=state.get("turn_number", 1),
        session_id=state.get("session_id", ""),
        should_escalate=hard_escalate,
    )

    logger.info(f"[analyze] escalate={hard_escalate}, neg_turns={negative_turns}")

    return {
        **state,
        "should_escalate": hard_escalate,
        "system_prompt": system_prompt,
    }


def respond_node(state: AgentState) -> AgentState:
    """
    Node 2: Generate a response using the LLM.
    Tries Ollama first (local, free), falls back to Groq if Ollama is down.
    """
    model_used = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    # Build message list for the LLM
    messages = [SystemMessage(content=state["system_prompt"])]

    # Add conversation history (last 6 turns to avoid token explosion)
    for turn in state.get("conversation_history", [])[-6:]:
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    # Add current user message
    messages.append(HumanMessage(content=state["user_message"]))

    # Try Ollama first
    try:
        llm = get_local_llm()
        response = llm.invoke(messages)
        agent_text = response.content
        logger.info(f"[respond] Ollama responded: '{agent_text[:80]}...'")
    except Exception as e:
        logger.warning(f"Ollama failed ({e}), falling back to Groq...")
        try:
            llm = get_groq_llm()
            response = llm.invoke(messages)
            agent_text = response.content
            model_used = "groq/llama-3.1-70b-versatile"
            logger.info(f"[respond] Groq fallback responded: '{agent_text[:80]}...'")
        except Exception as e2:
            agent_text = "I'm sorry, I'm having trouble processing your request right now. A specialist will follow up shortly."
            model_used = "fallback_message"
            logger.error(f"Both LLMs failed: {e2}")

    return {
        **state,
        "agent_response": agent_text,
        "model_used": model_used,
        "action": "respond",
    }


def escalate_node(state: AgentState) -> AgentState:
    """
    Node 3: Generate an escalation response.
    In a real system, this would also page on-call staff via SMS/email.
    """
    logger.info(f"[escalate] Triggering escalation for session {state['session_id']}")

    # Still use LLM to generate a warm, human-sounding escalation message
    model_used = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    messages = [
        SystemMessage(content=state["system_prompt"]),  # Already the escalation prompt
        HumanMessage(content=state["user_message"]),
    ]

    try:
        llm = get_local_llm()
        response = llm.invoke(messages)
        agent_text = response.content
    except Exception:
        agent_text = (
            "I completely understand your frustration, and I sincerely apologize. "
            "I'm connecting you with a specialist right now who will follow up within 15 minutes. "
            "Would you prefer a phone call, text message, or email?"
        )

    return {
        **state,
        "agent_response": agent_text,
        "model_used": model_used,
        "action": "escalate",
    }


# ── Conditional Router ─────────────────────────────────────────────────────────

def route_after_analyze(state: AgentState) -> str:
    """
    This is where LangGraph shines — conditional edges based on state.
    After analyzing, route to either 'respond' or 'escalate'.
    """
    if state.get("should_escalate", False):
        return "escalate"
    return "respond"


# ── Build the Graph ────────────────────────────────────────────────────────────

def build_agent_graph():
    """
    Assemble the LangGraph state machine.
    LangSmith automatically traces this graph if LANGCHAIN_TRACING_V2=true.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("analyze", analyze_node)
    graph.add_node("respond", respond_node)
    graph.add_node("escalate", escalate_node)

    # Add edges
    graph.add_edge(START, "analyze")
    graph.add_conditional_edges("analyze", route_after_analyze, {
        "respond": "respond",
        "escalate": "escalate",
    })
    graph.add_edge("respond", END)
    graph.add_edge("escalate", END)

    return graph.compile()


# Singleton — compile once, reuse
agent_graph = build_agent_graph()
