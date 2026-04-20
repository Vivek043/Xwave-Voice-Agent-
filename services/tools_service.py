"""
Agent Tools — Mock CRM/Helpdesk Actions for Agentic Capabilities

These tools simulate real business actions that a production AI agent would perform.
In production (like Mira), these would connect to real APIs: Salesforce, Zendesk, Calendly, etc.
Here they return realistic mock data to demonstrate the ARCHITECTURE.

What this proves in an interview:
- You understand tool-calling patterns (the core of agentic AI)
- You know how to structure tool inputs/outputs for LLM consumption
- You can design the boundary between AI reasoning and deterministic actions

Each tool follows the same pattern:
    Input: structured parameters → Process: business logic → Output: structured result
    The LLM decides WHICH tool to call and with WHAT parameters.
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ── Mock Data Store (simulates CRM/helpdesk database) ───────────────────────

MOCK_TICKETS = {
    "TK-1001": {
        "id": "TK-1001",
        "subject": "Cannot send emails from NovaCRM",
        "status": "in_progress",
        "priority": "P2",
        "created": "2026-04-15T10:30:00",
        "assigned_to": "Sarah Chen",
        "last_update": "2026-04-16T14:00:00",
        "description": "User reports email integration returning 403 errors since yesterday. Gmail OAuth token may have expired.",
        "resolution_eta": "2026-04-17T18:00:00",
    },
    "TK-1042": {
        "id": "TK-1042",
        "subject": "Dashboard loading slowly",
        "status": "open",
        "priority": "P3",
        "created": "2026-04-16T09:15:00",
        "assigned_to": "Unassigned",
        "last_update": "2026-04-16T09:15:00",
        "description": "Dashboard takes 15+ seconds to load. User has 18,000 contacts.",
        "resolution_eta": None,
    },
    "TK-1087": {
        "id": "TK-1087",
        "subject": "Request to upgrade from Starter to Professional",
        "status": "resolved",
        "priority": "P4",
        "created": "2026-04-10T11:00:00",
        "assigned_to": "Mike Torres",
        "last_update": "2026-04-11T16:30:00",
        "description": "User wants to upgrade plan. Processed upgrade and confirmed new features.",
        "resolution_eta": None,
    },
}

MOCK_ACCOUNTS = {
    "ACC-2001": {
        "id": "ACC-2001",
        "name": "Vivek Sharma",
        "email": "vivek@example.com",
        "company": "TechStart Inc.",
        "plan": "Professional",
        "status": "active",
        "created": "2025-08-15",
        "contacts_count": 4200,
        "open_tickets": ["TK-1001"],
        "account_manager": "Jessica Park",
    },
    "ACC-2002": {
        "id": "ACC-2002",
        "name": "Priya Patel",
        "email": "priya@example.com",
        "company": "DataFlow Solutions",
        "plan": "Enterprise",
        "status": "active",
        "created": "2024-03-22",
        "contacts_count": 45000,
        "open_tickets": ["TK-1042"],
        "account_manager": "Jessica Park",
    },
}


# ── Tool Definitions ─────────────────────────────────────────────────────────
# Each tool returns a dict that the LLM can use to formulate a natural response

TOOL_DEFINITIONS = [
    {
        "name": "check_ticket_status",
        "description": "Look up the current status of a support ticket by ticket ID. Use when the user asks about their ticket, support request, or issue status.",
        "parameters": {
            "ticket_id": "The ticket ID (e.g., TK-1001). If user doesn't provide one, ask for it.",
        },
    },
    {
        "name": "create_ticket",
        "description": "Create a new support ticket for the user's issue. Use when the user reports a problem that needs tracking.",
        "parameters": {
            "subject": "Brief description of the issue",
            "description": "Detailed description of the problem",
            "priority": "P1 (critical), P2 (high), P3 (medium), or P4 (low)",
        },
    },
    {
        "name": "reset_password",
        "description": "Initiate a password reset for the user's account. Use when user says they forgot their password or can't log in.",
        "parameters": {
            "email": "The user's registered email address",
        },
    },
    {
        "name": "lookup_account",
        "description": "Look up account details by email or account ID. Use when you need to verify the user's identity or check their plan details.",
        "parameters": {
            "identifier": "Email address or account ID (e.g., ACC-2001)",
        },
    },
    {
        "name": "schedule_callback",
        "description": "Schedule a callback from a human support specialist. Use when the issue is complex or the user explicitly requests to speak with a person.",
        "parameters": {
            "preferred_time": "When the user would like the callback (e.g., 'tomorrow morning', '3pm today')",
            "reason": "Brief reason for the callback",
        },
    },
]


# ── Tool Implementations ─────────────────────────────────────────────────────

async def check_ticket_status(ticket_id: str) -> dict:
    """Look up a support ticket's current status."""
    ticket_id = ticket_id.strip().upper()

    ticket = MOCK_TICKETS.get(ticket_id)
    if not ticket:
        return {
            "tool": "check_ticket_status",
            "success": False,
            "message": f"No ticket found with ID {ticket_id}. Please verify the ticket number.",
            "suggestion": "Valid format is TK-XXXX (e.g., TK-1001)",
        }

    status_friendly = {
        "open": "Open — waiting to be assigned to an agent",
        "in_progress": "In Progress — an agent is actively working on this",
        "resolved": "Resolved — this issue has been fixed",
        "closed": "Closed",
    }

    result = {
        "tool": "check_ticket_status",
        "success": True,
        "ticket_id": ticket["id"],
        "subject": ticket["subject"],
        "status": status_friendly.get(ticket["status"], ticket["status"]),
        "priority": ticket["priority"],
        "assigned_to": ticket["assigned_to"],
        "last_update": ticket["last_update"],
    }

    if ticket.get("resolution_eta"):
        result["estimated_resolution"] = ticket["resolution_eta"]

    logger.info(f"🎫 Ticket lookup: {ticket_id} → {ticket['status']}")
    return result


async def create_ticket(subject: str, description: str, priority: str = "P3") -> dict:
    """Create a new support ticket."""
    ticket_id = f"TK-{1100 + len(MOCK_TICKETS)}"

    new_ticket = {
        "id": ticket_id,
        "subject": subject,
        "status": "open",
        "priority": priority.upper() if priority else "P3",
        "created": datetime.utcnow().isoformat(),
        "assigned_to": "Auto-assignment pending",
        "last_update": datetime.utcnow().isoformat(),
        "description": description,
        "resolution_eta": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
    }
    MOCK_TICKETS[ticket_id] = new_ticket

    logger.info(f"🎫 Created ticket: {ticket_id} — {subject}")
    return {
        "tool": "create_ticket",
        "success": True,
        "ticket_id": ticket_id,
        "message": f"Ticket {ticket_id} has been created and will be assigned to an agent shortly.",
        "priority": new_ticket["priority"],
        "estimated_response": "Within 12 hours for P2/P3 priority",
    }


async def reset_password(email: str) -> dict:
    """Initiate a password reset."""
    email = email.strip().lower()

    # Check if account exists
    account = None
    for acc in MOCK_ACCOUNTS.values():
        if acc["email"] == email:
            account = acc
            break

    if not account:
        return {
            "tool": "reset_password",
            "success": False,
            "message": f"No account found with email {email}. Please check the email address and try again.",
        }

    logger.info(f"🔑 Password reset initiated for {email}")
    return {
        "tool": "reset_password",
        "success": True,
        "message": f"A password reset link has been sent to {email}. It will arrive within 2 minutes and expires after 1 hour.",
        "account_name": account["name"],
        "next_steps": "Check your email (including spam folder) for the reset link.",
    }


async def lookup_account(identifier: str) -> dict:
    """Look up account information."""
    identifier = identifier.strip()

    account = None
    # Try by account ID first
    if identifier.upper().startswith("ACC-"):
        account = MOCK_ACCOUNTS.get(identifier.upper())
    else:
        # Try by email
        for acc in MOCK_ACCOUNTS.values():
            if acc["email"] == identifier.lower():
                account = acc
                break

    if not account:
        return {
            "tool": "lookup_account",
            "success": False,
            "message": f"No account found for '{identifier}'. Please verify and try again.",
        }

    logger.info(f"👤 Account lookup: {account['id']} — {account['name']}")
    return {
        "tool": "lookup_account",
        "success": True,
        "account_id": account["id"],
        "name": account["name"],
        "company": account["company"],
        "plan": account["plan"],
        "status": account["status"],
        "contacts_count": account["contacts_count"],
        "open_tickets": account["open_tickets"],
        "account_manager": account["account_manager"],
    }


async def schedule_callback(preferred_time: str, reason: str) -> dict:
    """Schedule a callback from a human specialist."""
    callback_id = f"CB-{uuid.uuid4().hex[:6].upper()}"

    logger.info(f"📞 Callback scheduled: {callback_id} — {reason}")
    return {
        "tool": "schedule_callback",
        "success": True,
        "callback_id": callback_id,
        "message": f"Callback {callback_id} has been scheduled. A specialist will call you {preferred_time}.",
        "preferred_time": preferred_time,
        "reason": reason,
        "note": "You'll receive a confirmation email with the specialist's name and direct number.",
    }


# ── Tool Router ──────────────────────────────────────────────────────────────

TOOL_MAP = {
    "check_ticket_status": check_ticket_status,
    "create_ticket": create_ticket,
    "reset_password": reset_password,
    "lookup_account": lookup_account,
    "schedule_callback": schedule_callback,
}


async def execute_tool(tool_name: str, parameters: dict) -> dict:
    """Execute a tool by name with given parameters."""
    tool_fn = TOOL_MAP.get(tool_name)
    if not tool_fn:
        return {"tool": tool_name, "success": False, "message": f"Unknown tool: {tool_name}"}

    try:
        return await tool_fn(**parameters)
    except Exception as e:
        logger.error(f"Tool execution error: {tool_name} — {e}")
        return {"tool": tool_name, "success": False, "message": f"Tool error: {str(e)}"}


def get_tools_description() -> str:
    """
    Format tool definitions into a string the LLM can understand.
    This gets injected into the system prompt so the LLM knows what tools are available.
    """
    lines = ["AVAILABLE TOOLS (you can use these to help the user):"]
    for tool in TOOL_DEFINITIONS:
        params = ", ".join(f"{k}: {v}" for k, v in tool["parameters"].items())
        lines.append(f"\n• {tool['name']}: {tool['description']}")
        lines.append(f"  Parameters: {params}")

    lines.append(
        "\n\nTo use a tool, include EXACTLY this format in your response:\n"
        "[TOOL_CALL: tool_name | param1=value1 | param2=value2]\n"
        "Example: [TOOL_CALL: check_ticket_status | ticket_id=TK-1001]\n"
        "Example: [TOOL_CALL: reset_password | email=user@example.com]\n"
        "Only call ONE tool per response. After the tool result comes back, "
        "you'll respond to the user with the information."
    )
    return "\n".join(lines)
