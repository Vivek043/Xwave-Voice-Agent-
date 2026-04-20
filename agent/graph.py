"""
LangGraph Agent — Stateful AI Agent with RAG + Tool Calling + Conditional Routing

This is the upgraded Xwave agent graph. The original was:
    analyze → route → respond/escalate

Now it's:
    START
      │
      ▼
  [analyze]  ←── Reads sentiment + conversation history
      │
      ▼
  [retrieve_knowledge]  ←── RAG: fetch relevant docs from ChromaDB
      │
      ▼
  [route] ─── if escalate → [escalate] → END
      │
      ▼ else
  [respond]  ←── Calls LLM with knowledge context + tool awareness
      │
      ▼
  [parse_tool_call]  ←── Did the LLM request a tool?
      │
      ├── yes → [execute_tool] → [respond_with_tool_result] → END
      │
      └── no → END

What this demonstrates:
- RAG (Retrieval-Augmented Generation) — grounding responses in company data
- Agentic tool calling — LLM decides when/which tool to use
- Multi-step reasoning — retrieve → think → act → respond
- LangSmith traces the entire flow automatically
"""

import os
import re
import json
import logging
from typing import TypedDict
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, START

from agent.prompts import get_prompt_for_sentiment
from services.rag_service import retrieve, format_context_for_llm
from services.tools_service import execute_tool, get_tools_description, TOOL_MAP

load_dotenv()
logger = logging.getLogger(__name__)

# ── LLM Configuration ─────────────────────────────────────────────────────────

def get_local_llm():
    """Primary: Ollama running locally. Unlimited, free."""
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.7,
        timeout=8,
    )

def get_groq_llm():
    """Fallback: Groq cloud (free tier, very fast)."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=512,
    )


# ── Agent State Schema ─────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The shared state that flows through every node in the graph.
    Now includes RAG context and tool call tracking.
    """
    # Inputs
    session_id: str
    user_message: str
    sentiment: dict
    turn_number: int
    conversation_history: list

    # Internal
    should_escalate: bool
    negative_turn_count: int
    system_prompt: str

    # RAG
    retrieved_context: str      # Formatted knowledge base context
    retrieval_sources: list     # Source documents used

    # Tool calling
    tool_call_raw: str          # Raw tool call string from LLM
    tool_name: str              # Parsed tool name
    tool_params: dict           # Parsed tool parameters
    tool_result: dict           # Result from tool execution
    needs_tool: bool            # Whether LLM requested a tool

    # Outputs
    agent_response: str
    model_used: str
    action: str


# ── Graph Nodes ────────────────────────────────────────────────────────────────

def analyze_node(state: AgentState) -> AgentState:
    """
    Node 1: Analyze the current state.
    Decides whether to escalate based on sentiment + history.
    (Same as before — this logic stays clean.)
    """
    sentiment = state["sentiment"]
    negative_turns = state.get("negative_turn_count", 0)
    escalation_turn_count = int(os.getenv("ESCALATION_TURN_COUNT", "3"))

    hard_escalate = (
        sentiment.get("escalate_flag", False)
        and negative_turns >= escalation_turn_count
    )

    logger.info(f"[analyze] escalate={hard_escalate}, neg_turns={negative_turns}")

    return {
        **state,
        "should_escalate": hard_escalate,
    }


async def retrieve_knowledge_node(state: AgentState) -> AgentState:
    """
    Node 2 (NEW): RAG Retrieval.
    Searches the knowledge base for chunks relevant to the user's question.
    The retrieved context gets injected into the system prompt.
    """
    query = state["user_message"]
    chunks = await retrieve(query, top_k=4)

    context_str = format_context_for_llm(chunks)
    sources = [c["source"] for c in chunks]

    # Now build the system prompt WITH knowledge context and tool descriptions
    sentiment = state["sentiment"]
    tools_desc = get_tools_description()

    system_prompt = get_prompt_for_sentiment(
        sentiment_label=sentiment.get("label", "NEUTRAL"),
        sentiment_score=sentiment.get("score", 0.0),
        negative_turns=state.get("negative_turn_count", 0),
        turn_number=state.get("turn_number", 1),
        session_id=state.get("session_id", ""),
        should_escalate=state.get("should_escalate", False),
        knowledge_context=context_str,
        tools_description=tools_desc,
    )

    logger.info(f"[retrieve] Found {len(chunks)} relevant chunks from: {sources}")

    return {
        **state,
        "retrieved_context": context_str,
        "retrieval_sources": sources,
        "system_prompt": system_prompt,
    }


def _call_llm(messages: list) -> tuple[str, str]:
    """
    Try Ollama first, fall back to Groq. Returns (response_text, model_used).
    """
    model_used = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    try:
        llm = get_local_llm()
        response = llm.invoke(messages)
        return response.content, model_used
    except Exception as e:
        logger.warning(f"Ollama failed ({e}), falling back to Groq...")
        try:
            llm = get_groq_llm()
            response = llm.invoke(messages)
            return response.content, "groq/llama-3.3-70b-versatile"
        except Exception as e2:
            logger.error(f"Both LLMs failed: {e2}")
            return (
                "I'm sorry, I'm having trouble processing your request right now. "
                "A specialist will follow up shortly.",
                "fallback_message",
            )


def respond_node(state: AgentState) -> AgentState:
    """
    Node 3: Generate a response using the LLM.
    Now includes RAG context and tool awareness in the prompt.
    """
    messages = [SystemMessage(content=state["system_prompt"])]

    for turn in state.get("conversation_history", [])[-6:]:
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=state["user_message"]))

    agent_text, model_used = _call_llm(messages)

    logger.info(f"[respond] {model_used} → '{agent_text[:80]}...'")

    return {
        **state,
        "agent_response": agent_text,
        "model_used": model_used,
        "action": "respond",
    }


def parse_tool_call_node(state: AgentState) -> AgentState:
    """
    Node 4 (NEW): Check if the LLM's response contains a tool call.
    Parses [TOOL_CALL: tool_name | param1=value1 | param2=value2] format.
    """
    response = state.get("agent_response", "")

    # Regex to match tool call pattern
    pattern = r'\[TOOL_CALL:\s*(\w+)\s*(?:\|([^\]]*))?\]'
    match = re.search(pattern, response)

    if not match:
        return {**state, "needs_tool": False, "tool_name": "", "tool_params": {}}

    tool_name = match.group(1).strip()
    params_str = match.group(2) or ""

    # Parse parameters
    params = {}
    if params_str.strip():
        for part in params_str.split("|"):
            part = part.strip()
            if "=" in part:
                key, value = part.split("=", 1)
                params[key.strip()] = value.strip()

    # Validate tool exists
    if tool_name not in TOOL_MAP:
        logger.warning(f"[parse_tool] LLM called unknown tool: {tool_name}")
        return {**state, "needs_tool": False, "tool_name": "", "tool_params": {}}

    logger.info(f"[parse_tool] Tool detected: {tool_name}({params})")

    return {
        **state,
        "needs_tool": True,
        "tool_name": tool_name,
        "tool_params": params,
        "tool_call_raw": match.group(0),
    }


async def execute_tool_node(state: AgentState) -> AgentState:
    """
    Node 5 (NEW): Execute the tool the LLM requested.
    Gets the result and stores it for the final response.
    """
    tool_name = state["tool_name"]
    tool_params = state["tool_params"]

    logger.info(f"[execute_tool] Running {tool_name} with {tool_params}")
    result = await execute_tool(tool_name, tool_params)

    logger.info(f"[execute_tool] Result: {json.dumps(result, indent=2)[:200]}")

    return {
        **state,
        "tool_result": result,
    }


def respond_with_tool_result_node(state: AgentState) -> AgentState:
    """
    Node 6 (NEW): Generate a natural language response using the tool result.
    The LLM takes the raw tool output and converts it into a conversational response.
    """
    tool_result = state.get("tool_result", {})
    original_response = state.get("agent_response", "")

    # Build a follow-up prompt with the tool result
    messages = [
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(content=state["user_message"]),
        AIMessage(content=original_response),
        HumanMessage(content=(
            f"Tool '{state['tool_name']}' returned this result:\n"
            f"{json.dumps(tool_result, indent=2)}\n\n"
            "Now respond to the user naturally using this information. "
            "Be concise and conversational (this will be spoken aloud). "
            "Do NOT include any [TOOL_CALL:...] tags in this response."
        )),
    ]

    agent_text, model_used = _call_llm(messages)

    logger.info(f"[respond_with_tool] Final: '{agent_text[:80]}...'")

    return {
        **state,
        "agent_response": agent_text,
        "model_used": model_used,
        "action": f"tool:{state['tool_name']}",
    }


def escalate_node(state: AgentState) -> AgentState:
    """
    Escalation node — now with tool awareness (can offer to schedule callback).
    """
    logger.info(f"[escalate] Triggering escalation for session {state['session_id']}")

    messages = [
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(content=state["user_message"]),
    ]

    agent_text, model_used = _call_llm(messages)

    return {
        **state,
        "agent_response": agent_text,
        "model_used": model_used,
        "action": "escalate",
    }


# ── Conditional Routers ──────────────────────────────────────────────────────

def route_after_analyze(state: AgentState) -> str:
    """Route to escalation or knowledge retrieval."""
    if state.get("should_escalate", False):
        return "escalate"
    return "retrieve_knowledge"


def route_after_tool_parse(state: AgentState) -> str:
    """Route based on whether the LLM wants to call a tool."""
    if state.get("needs_tool", False):
        return "execute_tool"
    return END


# ── Build the Graph ──────────────────────────────────────────────────────────

def build_agent_graph():
    """
    Assemble the upgraded LangGraph state machine.

    Flow:
        START → analyze → route →
            ├── escalate path: escalate → END
            └── normal path:   retrieve_knowledge → respond → parse_tool_call →
                                  ├── needs tool: execute_tool → respond_with_tool_result → END
                                  └── no tool:    END
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("analyze", analyze_node)
    graph.add_node("retrieve_knowledge", retrieve_knowledge_node)
    graph.add_node("respond", respond_node)
    graph.add_node("parse_tool_call", parse_tool_call_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("respond_with_tool_result", respond_with_tool_result_node)
    graph.add_node("escalate", escalate_node)

    # Wire up edges
    graph.add_edge(START, "analyze")

    # After analyze: escalate or retrieve knowledge
    graph.add_conditional_edges("analyze", route_after_analyze, {
        "escalate": "escalate",
        "retrieve_knowledge": "retrieve_knowledge",
    })

    # After retrieval: always respond
    graph.add_edge("retrieve_knowledge", "respond")

    # After respond: check for tool calls
    graph.add_edge("respond", "parse_tool_call")

    # After parsing: execute tool or finish
    graph.add_conditional_edges("parse_tool_call", route_after_tool_parse, {
        "execute_tool": "execute_tool",
        END: END,
    })

    # After tool execution: generate final response
    graph.add_edge("execute_tool", "respond_with_tool_result")
    graph.add_edge("respond_with_tool_result", END)

    # Escalation always ends
    graph.add_edge("escalate", END)

    return graph.compile()


# Singleton — compile once, reuse
agent_graph = build_agent_graph()
