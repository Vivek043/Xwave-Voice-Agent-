# Xwave — Live Demo Script for Recruiter Call

## Pre-Demo Checklist (do this 10 min before the call)

```bash
# 1. Start Ollama (if using local LLM)
ollama serve

# 2. Activate environment + start backend
cd ~/Desktop/Xwave
source venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# 3. Load the knowledge base (if not already loaded)
curl -X POST http://localhost:8000/knowledge/load-defaults

# 4. Verify everything works
curl http://localhost:8000/health

# 5. Open these in browser tabs:
#    Tab 1: Frontend  → file:///Users/saivivek/Desktop/Xwave/Frontend/index.html
#    Tab 2: Swagger   → http://localhost:8000/docs
#    Tab 3: LangSmith → https://smith.langchain.com (your project)
```

## Quick Test (30 seconds, make sure nothing is broken)
Type in the text input: "What pricing plans do you offer?"
— Should get a RAG-grounded answer about NovaCRM plans.
— You should see the purple "Knowledge Base" tag on the response.

---

## The Demo Flow (5-7 minutes)

### Opening (30 sec)
> "I built Xwave as a full-stack AI voice agent that mirrors the architecture of production platforms like your Mira product. Let me show you what it does."

### Demo 1: RAG Knowledge Retrieval (1 min)

Type: **"What are your pricing plans?"**

**What to point out:**
- The response cites actual pricing from the uploaded NovaCRM FAQ — not hallucinated
- The purple "Knowledge Base" tag shows RAG was used
- In the activity feed, you'll see "Retrieved from company_faq.md"
- Click the Debug button (top right) to show the JSON trace with rag_sources

**What to say:**
> "The agent doesn't hallucinate — it retrieves relevant chunks from the company's uploaded knowledge base using ChromaDB vector search, then grounds its response in that data. This is the same RAG pattern that powers Mira Voice when a customer uploads their company docs."

### Demo 2: Tool Calling (1 min)

Type: **"What's the status of ticket TK-1001?"**

**What to point out:**
- The blue tool tag shows "check_ticket_status" was called
- The activity feed shows the tool execution
- The response includes real ticket data (assigned to Sarah Chen, P2 priority, in progress)
- The debug panel shows tool_result with structured data

**What to say:**
> "The LLM decided autonomously that it needed to call a tool. It parsed the ticket ID from my message, called the check_ticket_status function, got structured data back, then converted that into a natural conversational response. In production this would hit your Zendesk or Salesforce API — the architecture is identical."

### Demo 3: Another Tool — Password Reset (30 sec)

Type: **"I forgot my password, my email is vivek@example.com"**

**What to point out:**
- Agent calls reset_password tool
- Responds with confirmation and next steps
- This mimics Mira IT's password reset capability

### Demo 4: Sentiment Analysis (1 min)

Type: **"I'm really frustrated, this is the third time I'm calling about this issue"**

**What to point out:**
- The sentiment strip at the top turns RED and slides to the negative side
- Shows FRUSTRATED with a negative score
- The agent's tone changes — it acknowledges frustration first before offering solutions
- Explain: "The system prompt dynamically switches between normal, empathetic, and escalation modes based on real-time sentiment"

Then type: **"Nothing ever gets fixed! I want to talk to someone!"**

**What to point out:**
- Score drops further
- If it escalates, the red banner appears
- Agent offers to schedule a callback (tool calling + escalation combined)

### Demo 5: Voice (if mic works) (30 sec)

Hold the mic button and say something like: "Can you help me with my account?"

**What to point out:**
- Speech-to-text via Groq Whisper
- Full pipeline: voice → transcript → sentiment → RAG → LLM → tools → voice response
- Agent speaks back using Edge-TTS with sentiment-adaptive voice selection

### Demo 6: LangSmith Traces (1 min)

Switch to your LangSmith tab and show the latest trace.

**What to point out:**
- The full graph execution: analyze → retrieve_knowledge → respond → parse_tool_call → execute_tool → respond_with_tool_result
- Each node shows input/output, latency, and token counts
- "Every decision the agent makes is fully observable. In production, this is how you debug and optimize agent behavior."

### Demo 7: Swagger API Docs (30 sec)

Switch to http://localhost:8000/docs

**What to point out:**
- Clean REST API with all endpoints documented
- Knowledge Base section (upload, search, stats)
- Agent section (chat, voice, conversation history)
- "This is a complete API that could be integrated into any frontend or telephony system"

---

## Key Talking Points (weave these in naturally)

1. **"Zero cost"** — Every component is free-tier or local. Ollama for local LLM, Groq for cloud fallback, Edge-TTS, VADER for sentiment, ChromaDB for vectors. No API spend.

2. **"Fail-graceful architecture"** — Ollama down? Auto-fallback to Groq. Both down? Static message. The system never crashes on the user.

3. **"This maps directly to Mira"** — RAG = Mira's "upload your knowledge base". Tool calling = Mira's CRM/ticket integrations. Sentiment routing = Mira's intelligent call handling. Escalation = Mira's handoff to humans.

4. **"LangGraph, not hardcoded"** — The agent graph is a real state machine with conditional edges. Adding a new capability means adding a node, not rewriting if/else chains.

5. **"Observable by default"** — LangSmith traces every turn automatically. In production this is critical for debugging why an agent gave a wrong answer.

---

## Questions They Might Ask (and your answers)

**"How would you scale this to handle real phone calls?"**
> "Swap the browser WebRTC frontend for a Twilio or Vonage SIP integration. The backend API is already designed for it — the /agent/chat/voice endpoint takes audio in and returns audio out. You'd just need a telephony bridge in front of it, which is what Mira uses with Deepgram and ElevenLabs."

**"Why not use OpenAI / Claude directly?"**
> "For the portfolio I prioritized zero cost to demonstrate that the architecture works regardless of which LLM sits behind it. In production, swapping Ollama for OpenAI is a one-line config change — the LangGraph orchestration layer is model-agnostic."

**"How does the RAG handle large knowledge bases?"**
> "ChromaDB scales to millions of vectors. The chunking strategy (500 chars with 80-char overlap) is tuned for embedding model context windows. In production you'd add metadata filtering (by tenant, document type) and a re-ranking step before injecting into the LLM context."

**"What would you improve next?"**
> "Three things: real-time WebSocket streaming for live transcription, multi-tenant support so each customer gets isolated knowledge bases, and a smarter LLM router that picks the right model based on query complexity — fast model for simple lookups, large model for nuanced reasoning."
