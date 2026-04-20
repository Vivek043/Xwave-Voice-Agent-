<p align="center">
  <img src="https://img.shields.io/badge/Xwave-AI%20Voice%20Agent-F5C518?style=for-the-badge&logo=soundcharts&logoColor=white" alt="Xwave"/>
</p>

<h1 align="center">Xwave</h1>
<h3 align="center">Intelligent Voice Agent with RAG, Agentic Tool Calling & Real-Time Sentiment Routing</h3>

<p align="center">
  <img src="https://img.shields.io/badge/version-2.0.0-F5C518?style=flat-square" alt="Version"/>
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/LangGraph-Agentic-1C3C3C?style=flat-square&logo=langchain&logoColor=white" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/ChromaDB-RAG-FF6F00?style=flat-square" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/cost-%240%2Fmonth-22C55E?style=flat-square" alt="Zero Cost"/>
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"/>
</p>

<p align="center">
  A production-grade AI voice agent that <strong>listens</strong>, <strong>understands</strong>, <strong>retrieves knowledge</strong>, <strong>takes actions</strong>, and <strong>speaks back</strong> — all running locally at zero cost.
</p>

---

## The Problem

> *"Our customers call at 3 AM. Our support team doesn't work at 3 AM."*

Businesses lose revenue and trust every time a customer call goes unanswered. Traditional IVR systems frustrate users with rigid menus. Generic chatbots hallucinate answers because they don't know your business.

**Xwave solves this.** It's an AI voice agent that knows your company's knowledge base, detects customer frustration in real-time, takes autonomous actions (ticket lookups, password resets, callback scheduling), and escalates to humans only when it should.

---

## How It Works

```
                                    XWAVE AGENT ARCHITECTURE
  ╔══════════════════════════════════════════════════════════════════════════════╗
  ║                                                                            ║
  ║   User speaks                                                              ║
  ║      │                                                                     ║
  ║      ▼                                                                     ║
  ║   ┌──────────┐    ┌──────────────┐    ┌────────────────────┐              ║
  ║   │ Groq     │───▶│  VADER       │───▶│  LangGraph Agent   │              ║
  ║   │ Whisper  │    │  Sentiment   │    │  ┌──────────────┐  │              ║
  ║   │ (STT)    │    │  Analysis    │    │  │  Analyze     │  │              ║
  ║   └──────────┘    └──────────────┘    │  └──────┬───────┘  │              ║
  ║                                        │         │          │              ║
  ║                                        │         ▼          │              ║
  ║                  ┌─────────────┐       │  ┌──────────────┐  │              ║
  ║                  │  ChromaDB   │◀──────│  │  Retrieve    │  │              ║
  ║                  │  Vector     │──────▶│  │  Knowledge   │  │              ║
  ║                  │  Store      │       │  └──────┬───────┘  │              ║
  ║                  └─────────────┘       │         │          │              ║
  ║                                        │         ▼          │              ║
  ║                                        │  ┌──────────────┐  │              ║
  ║                  ┌─────────────┐       │  │  Respond     │  │              ║
  ║                  │  Ollama /   │◀──────│  │  (LLM)       │  │              ║
  ║                  │  Groq LLM   │──────▶│  └──────┬───────┘  │              ║
  ║                  └─────────────┘       │         │          │              ║
  ║                                        │         ▼          │              ║
  ║                  ┌─────────────┐       │  ┌──────────────┐  │              ║
  ║                  │  Mock CRM   │◀──────│  │  Tool Call   │  │              ║
  ║                  │  Ticket Sys │──────▶│  │  Executor    │  │              ║
  ║                  │  Auth Sys   │       │  └──────┬───────┘  │              ║
  ║                  └─────────────┘       │         │          │              ║
  ║                                        └─────────┼──────────┘              ║
  ║                                                  │                         ║
  ║      ┌──────────┐                                ▼                         ║
  ║      │ Edge-TTS │◀──────────────────── Agent Response                      ║
  ║      │ (Speech) │                                                          ║
  ║      └────┬─────┘                     ┌──────────────────┐                ║
  ║           │                           │  LangSmith       │                ║
  ║           ▼                           │  (Full Trace)    │                ║
  ║      User hears                       └──────────────────┘                ║
  ║      the response                                                          ║
  ║                                                                            ║
  ╚══════════════════════════════════════════════════════════════════════════════╝
```

### The Agent Graph (LangGraph State Machine)

Every conversation turn flows through a **7-node directed graph** with conditional routing:

```
START
  │
  ▼
[analyze] ─── Reads sentiment score + conversation history
  │
  ├── sentiment critical + 3 consecutive negative turns
  │   └──▶ [escalate] ──▶ END
  │
  ▼
[retrieve_knowledge] ─── RAG: queries ChromaDB for relevant company docs
  │
  ▼
[respond] ─── LLM generates response with knowledge context + tool awareness
  │
  ▼
[parse_tool_call] ─── Did the LLM request a tool?
  │
  ├── yes ──▶ [execute_tool] ──▶ [respond_with_tool_result] ──▶ END
  │
  └── no ──▶ END
```

---

## Features

### Voice Pipeline
| Stage | Technology | What It Does |
|-------|-----------|--------------|
| **Listen** | Groq Whisper Large v3 | Real-time speech-to-text (2,000 req/day free) |
| **Understand** | VADER Sentiment | 7-level emotion detection: Happy → Upset |
| **Think** | LangGraph + Ollama/Groq | Multi-node stateful reasoning with conditional routing |
| **Remember** | ChromaDB (RAG) | Retrieves relevant knowledge from uploaded company docs |
| **Act** | Tool Calling Engine | Autonomous actions: ticket lookup, password reset, CRM queries |
| **Speak** | Edge-TTS | Neural voice synthesis with sentiment-adaptive tone selection |
| **Trace** | LangSmith | Full observability — every node, every decision, every token |

### RAG Knowledge Base
Upload your company docs and Xwave becomes an expert on your business:
- **Ingest**: PDF, Markdown, plain text — chunked and embedded automatically
- **Retrieve**: Cosine similarity search via ChromaDB (all-MiniLM-L6-v2 embeddings)
- **Ground**: Retrieved context injected into the LLM prompt — no hallucination
- **Manage**: Upload, search, stats, and clear endpoints via REST API

### Agentic Tool Calling
The LLM doesn't just talk — it **does things**:

| Tool | What It Does | Example Trigger |
|------|-------------|----------------|
| `check_ticket_status` | Looks up support ticket by ID | *"What's the status of TK-1001?"* |
| `create_ticket` | Creates a new support ticket | *"I have a bug to report"* |
| `reset_password` | Initiates password reset flow | *"I forgot my password"* |
| `lookup_account` | Retrieves account/CRM details | *"Look up my account"* |
| `schedule_callback` | Books a human specialist callback | *"Can I talk to someone?"* |

### Sentiment-Driven Escalation
```
Score ≥  0.7  →  HAPPY    😄  ──┐
Score ≥  0.4  →  POSITIVE 😊   │
Score ≥  0.1  →  CALM     🙂   ├── Normal prompt
Score ≥ -0.1  →  NEUTRAL  😐  ──┘
Score ≥ -0.4  →  CONCERNED 😕 ──┐── Empathetic prompt
Score ≥ -0.65 →  FRUSTRATED 😟 ─┘
Score <  -0.65 → UPSET    😤  ──── Escalation (after 3 turns)
```

The agent dynamically switches between **normal**, **empathetic**, and **escalation** system prompts based on real-time sentiment analysis. Three consecutive negative turns trigger automatic escalation to a human specialist.

---

## Tech Stack

```
┌─────────────────────────────────────────────────────────┐
│                     ZERO COST STACK                      │
├──────────────────┬──────────────────────────────────────┤
│ API Framework    │ FastAPI + Uvicorn                     │
│ Agent Framework  │ LangGraph (stateful, conditional)     │
│ Local LLM        │ Ollama (llama3.1:8b)                  │
│ Cloud LLM        │ Groq (llama-3.3-70b — free tier)      │
│ Speech-to-Text   │ Groq Whisper Large v3 (free tier)     │
│ Text-to-Speech   │ Edge-TTS (Microsoft neural voices)    │
│ Sentiment        │ VADER (local lexicon NLP)             │
│ Vector Store     │ ChromaDB (local, persistent)          │
│ Embeddings       │ all-MiniLM-L6-v2 (built into Chroma) │
│ Database         │ SQLite via SQLAlchemy                 │
│ Observability    │ LangSmith (free tier)                 │
│ Frontend         │ Vanilla HTML/CSS/JS + WebRTC          │
│ Containerization │ Docker + Docker Compose               │
└──────────────────┴──────────────────────────────────────┘
```

Every component is **free-tier or fully local**. No credit card. No vendor lock-in.

---

## Project Structure

```
Xwave/
├── agent/
│   ├── graph.py              # LangGraph state machine (7 nodes, 2 conditional edges)
│   └── prompts.py            # Dynamic system prompts (normal/empathetic/escalation)
│
├── services/
│   ├── rag_service.py        # ChromaDB vector store, chunking, retrieval
│   ├── tools_service.py      # 5 mock tools (CRM, tickets, auth, scheduling)
│   ├── sentiment_service.py  # VADER sentiment analysis (7 emotion levels)
│   ├── stt_service.py        # Groq Whisper speech-to-text
│   ├── tts_service.py        # Edge-TTS with adaptive voice selection
│   └── db_service.py         # SQLAlchemy models + conversation persistence
│
├── routers/
│   ├── agent.py              # Full pipeline: sentiment → RAG → agent → tools
│   ├── knowledge.py          # Knowledge base CRUD (upload, search, stats)
│   ├── voice.py              # Standalone STT/TTS endpoints
│   ├── sentiment.py          # Standalone sentiment endpoint
│   └── health.py             # Health check
│
├── knowledge_base/
│   ├── company_faq.md        # Sample: NovaCRM FAQ (pricing, account, integrations)
│   ├── support_policies.md   # Sample: SLA, escalation paths, refund policies
│   └── product_updates.md    # Sample: Release notes and feature announcements
│
├── Frontend/
│   ├── index.html            # Aether-inspired UI with floating particles
│   ├── style.css             # White + butter yellow glassmorphism theme
│   └── app.js                # WebRTC mic, sentiment chips, canvas animations
│
├── backend/
│   ├── main.py               # FastAPI app entry point
│   ├── .env                  # Environment variables (gitignored)
│   └── .env.example          # Template for environment setup
│
├── docker-compose.yml        # One-command deployment
├── Dockerfile                # Backend container
├── Dockerfile.frontend       # Frontend container
└── requirements.txt          # Python dependencies
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed (for local LLM)
- [Groq API key](https://console.groq.com) (free — for STT + LLM fallback)
- [LangSmith API key](https://smith.langchain.com) (free — for tracing)

### 1. Clone & Setup
```bash
git clone https://github.com/Vivek043/Xwave-Voice-Agent-.git
cd Xwave-Voice-Agent-
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp backend/.env.example backend/.env
# Edit backend/.env with your API keys
```

### 3. Pull the Local LLM
```bash
ollama pull llama3.1:8b
```

### 4. Start the Server
```bash
uvicorn backend.main:app --reload --port 8000
```

### 5. Load the Knowledge Base
```bash
curl -X POST http://localhost:8000/knowledge/load-defaults
```

### 6. Open the UI
```
open Frontend/index.html    # macOS
# or just open the file in your browser
```

### Docker (Alternative)
```bash
docker-compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

---

## API Reference

### Agent Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/agent/chat` | Text-based agent turn (sentiment → RAG → LLM → tools) |
| `POST` | `/agent/chat/voice` | Full voice pipeline (audio in → audio out) |
| `GET` | `/agent/conversation/{id}` | Retrieve full conversation history |

### Knowledge Base Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/knowledge/load-defaults` | Ingest sample NovaCRM documents |
| `POST` | `/knowledge/upload` | Upload a file (PDF, MD, TXT) |
| `POST` | `/knowledge/ingest` | Ingest raw text |
| `POST` | `/knowledge/search` | Test retrieval without agent |
| `GET` | `/knowledge/stats` | View knowledge base statistics |
| `DELETE` | `/knowledge/clear` | Wipe knowledge base |

### Utility Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/voice/transcribe` | Standalone speech-to-text |
| `GET` | `/voice/speak?text=...` | Standalone text-to-speech |
| `POST` | `/sentiment/analyze` | Standalone sentiment analysis |
| `GET` | `/health` | Health check |

---

## Demo Scenarios

### Scenario 1: Knowledge-Grounded Q&A
```
User:  "What pricing plans do you offer?"
Xwave: "We offer three plans — Starter at $29/month per user with up to 1,000
        contacts, Professional at $79/month with advanced automation and API
        access, and Enterprise at $149/month with unlimited contacts and a
        dedicated account manager. All plans include a 14-day free trial."
                                          [RAG: retrieved from company_faq.md]
```

### Scenario 2: Autonomous Tool Calling
```
User:  "What's the status of ticket TK-1001?"
Xwave: [TOOL_CALL: check_ticket_status] → executes lookup
Xwave: "Ticket TK-1001 regarding email sending issues is currently in progress.
        It's been assigned to Sarah Chen and the estimated resolution is
        tomorrow by 6 PM. Is there anything else I can help with?"
```

### Scenario 3: Sentiment Escalation
```
Turn 1: "This is broken again!" ──────── Sentiment: FRUSTRATED (-0.72)
         → Empathetic prompt activated
Turn 2: "I've called three times!!" ──── Sentiment: UPSET (-0.85)
         → Negative turn count: 2
Turn 3: "Nothing ever gets fixed!!!" ─── Sentiment: UPSET (-0.91)
         → 3 consecutive negatives → AUTO-ESCALATION
Xwave: "I completely understand your frustration, and I'm sorry for the
        repeated issues. I'm connecting you with a specialist right now.
        Would you prefer a callback by phone, text, or email?"
```

---

## LangSmith Observability

Every agent turn is automatically traced in LangSmith:

```
Trace: agent_graph.invoke
├── analyze_node          (2ms)   → escalate=false, neg_turns=0
├── retrieve_knowledge    (45ms)  → 4 chunks from company_faq.md
├── respond_node          (1.2s)  → groq/llama-3.3-70b-versatile
├── parse_tool_call_node  (1ms)   → tool=check_ticket_status
├── execute_tool_node     (3ms)   → success=true, TK-1001 in_progress
└── respond_with_result   (0.9s)  → final response with tool data
```

Visit [smith.langchain.com](https://smith.langchain.com) to see full traces with token counts, latencies, and decision paths.

---

## Design Philosophy

| Principle | Implementation |
|-----------|---------------|
| **Free-first** | Every component is free-tier or local. Zero spend. |
| **Fail-gracefully** | Ollama down? Auto-fallback to Groq. Both down? Static message. |
| **Ground truth** | RAG ensures responses come from your docs, not hallucination. |
| **Agentic, not scripted** | LLM decides when to use tools — not hardcoded `if/else`. |
| **Observable** | LangSmith traces every decision the agent makes. |
| **Empathy-aware** | Sentiment drives tone — frustrated users get care, not scripts. |

---

## Roadmap

- [x] Voice pipeline (STT → LLM → TTS)
- [x] Sentiment analysis with 7 emotion levels
- [x] LangGraph stateful agent with conditional routing
- [x] RAG knowledge base with ChromaDB
- [x] Agentic tool calling (5 tools)
- [x] LangSmith full-trace observability
- [x] Sentiment-adaptive voice selection
- [x] Auto-escalation after persistent negative sentiment
- [ ] Real-time WebSocket streaming (live transcription)
- [ ] Multi-LLM intelligent router (complexity-based model selection)
- [ ] Long-term memory (cross-session user context)
- [ ] Outbound agent (proactive follow-up calls)
- [ ] Webhook integrations (Slack, email notifications on escalation)
- [ ] Multi-tenant support with tenant-isolated knowledge bases

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for STT + LLM fallback |
| `LANGCHAIN_API_KEY` | Yes | LangSmith key for tracing |
| `LANGCHAIN_TRACING_V2` | Yes | Set to `true` to enable traces |
| `LANGCHAIN_PROJECT` | No | LangSmith project name (default: `voice-agent`) |
| `OLLAMA_BASE_URL` | No | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | No | Local model name (default: `llama3.1:8b`) |
| `SENTIMENT_ESCALATION_THRESHOLD` | No | Score threshold for escalation (default: `-0.5`) |
| `ESCALATION_TURN_COUNT` | No | Consecutive negative turns before escalation (default: `3`) |

---

## Built With

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain"/>
  <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama"/>
  <img src="https://img.shields.io/badge/Groq-F55036?style=for-the-badge" alt="Groq"/>
  <img src="https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
</p>

---

<p align="center">
  <strong>Xwave</strong> — Because your customers deserve an agent that listens, understands, and acts.<br/>
  <sub>Built with the conviction that production-grade AI shouldn't cost a dime to prototype.</sub>
</p>
