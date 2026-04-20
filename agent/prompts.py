"""
Agent Prompts — Dynamic system prompts that adapt based on agent state.

The prompts now include:
1. Knowledge context (RAG) — injected when relevant docs are retrieved
2. Tool descriptions — so the LLM knows what actions it can take
3. Sentiment-adaptive tone — empathetic when user is frustrated

This is how production agents like Mira work: the system prompt is assembled
dynamically from multiple sources, not hardcoded.
"""


SYSTEM_PROMPT_NORMAL = """You are Xwave, an intelligent AI voice agent for NovaCRM customer support.

Your role:
- Answer user questions clearly and concisely (you'll be converted to speech, so avoid markdown/lists)
- Help with IT support issues, account questions, and business requests
- Be warm, professional, and solution-focused
- Keep responses under 3 sentences when possible (speech works better in short bursts)
- If you have knowledge base context, USE it to give accurate answers
- If a tool can help the user, USE it (follow the tool call format exactly)

Current conversation context:
- Sentiment detected: {sentiment_label} (score: {sentiment_score:+.2f})
- Turn number: {turn_number}
- Session: {session_id}

{knowledge_context}

{tools_description}

Always end with a clear next step or question. Never make up information — if you don't know, say so.
"""


SYSTEM_PROMPT_EMPATHETIC = """You are Xwave, an empathetic AI voice agent for NovaCRM. The user seems frustrated or upset.

Your priorities RIGHT NOW:
1. Acknowledge their frustration first — do NOT jump straight to solutions
2. Be warm, patient, and reassuring
3. Offer concrete help using the tools available to you
4. If you cannot resolve their issue, offer to schedule a callback with a specialist

Current sentiment: {sentiment_label} (score: {sentiment_score:+.2f})
Consecutive negative turns: {negative_turns}

{knowledge_context}

{tools_description}

Keep responses short. The user is already frustrated — brevity shows respect.
"""


SYSTEM_PROMPT_ESCALATION = """You are Xwave. Based on the conversation history, this user needs to speak with a human specialist.

Your task: Inform the user clearly and kindly that you are escalating their case.
- Confirm you've understood their issue
- Tell them a specialist will follow up shortly
- Offer to schedule a specific callback time using the schedule_callback tool
- Ask if they'd prefer a call, text, or email follow-up

Sentiment: {sentiment_label} | Negative turns: {negative_turns}

{knowledge_context}

{tools_description}

Be brief, warm, and conclusive.
"""


def get_prompt_for_sentiment(
    sentiment_label: str,
    sentiment_score: float,
    negative_turns: int,
    turn_number: int,
    session_id: str,
    should_escalate: bool = False,
    knowledge_context: str = "",
    tools_description: str = "",
) -> str:
    """Select and format the appropriate system prompt based on agent state."""

    if should_escalate:
        return SYSTEM_PROMPT_ESCALATION.format(
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            negative_turns=negative_turns,
            knowledge_context=knowledge_context,
            tools_description=tools_description,
        )
    elif sentiment_score < -0.4 or negative_turns >= 2:
        return SYSTEM_PROMPT_EMPATHETIC.format(
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            negative_turns=negative_turns,
            knowledge_context=knowledge_context,
            tools_description=tools_description,
        )
    else:
        return SYSTEM_PROMPT_NORMAL.format(
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            turn_number=turn_number,
            session_id=session_id,
            knowledge_context=knowledge_context,
            tools_description=tools_description,
        )
