# Multi-Agent Data Analyst System [Streamlit Live Demo](https://dhananjaya-paliwal-data-analyst-agent.streamlit.app)

> **Production-grade AI system that analyzes CSV data using natural language queries with 8 specialized LLM agents orchestrated by LangGraph**

Transform your CSV data into actionable insights through conversational AI. Features semantic caching (80% similarity) for instant responses, military-grade PII encryption, intelligent answerability checks, multi-step planning, and self-healing query refinement.

---

## Table of Contents

- [Why This System?](#why-this-system)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Security & Privacy](#security--privacy)
- [Observability](#observability)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)

---

## Why This System?

### The Problem
- SQL/Python expertise required
- Manual data cleaning
- Time-consuming query writing
- No PII protection
- Expensive repeated queries

### The Solution
- ✅ Natural language queries (no SQL)
- ✅ Intelligent caching (99.7% faster)
- ✅ Automatic PII protection
- ✅ Self-healing refinement
- ✅ Transparent cost tracking

---

## Key Features

### 1. Multi-Agent Intelligence (8 Specialized Agents)

| Agent | Model | Key Feature |
|-------|-------|-------------|
| **Safety Guard** | llama-3.1-8b-instant | Validates query safety |
| **File Router** | llama-3.1-8b-instant | File selection + Answerability check |
| **Planner** | llama-3.3-70b-versatile | Multi-step strategy generation |
| **Data Analyst** | openai/gpt-oss-120b | Pandas code generation |
| **Insights Generator** | llama-3.3-70b-versatile | Creates insights first |
| **Logic Critic** | openai/gpt-oss-120b | Validates code logic |
| **Data Critic** | openai/gpt-oss-120b | Validates data quality |
| **Insights Critic** | llama-3.3-70b-versatile | Validates insights |

**Agent Flow:** Safety → Router (Answerability) → Planner (Multi-step) → Analyst → Insights First → Critics (3×) → Refinement (if needed)

#### File Router - Answerability Check
Before selecting files, the router asks:
- "Can we answer this question with available data?"
- "Do we have the necessary columns?"
- If NO → Returns honest "Cannot answer" instead of hallucinating

#### Planner - Multi-Step Strategy
Creates detailed, hierarchical plans:
```
1. Load customers.csv and orders.csv
2. Join on customer_id
   └─ Substep 2.1: Check for nulls
   └─ Substep 2.2: Inner join
3. Group by customer_id
4. Sum amount column
5. Sort descending
```

#### Self-Healing Refinement Process
When confidence < 80%:
1. **Analyze feedback** - "Missing null check in join"
2. **Regenerate code** - Data Analyst fixes the issue
3. **Re-validate** - Critics re-evaluate solution
4. **Repeat if needed** - Max 2 attempts total
5. **Example:** Wrong join → Add null handling → Critics approve ✅

### 2. Semantic Caching (80% Similarity)

**Intelligent Matching:**
```
Query: "What is the total revenue?"
Cached: "Show me sum of all sales"
Similarity: 84% → HIT! ⚡ (threshold: 80%)
```

**Performance:**
- First query: 18.3s, $0.0028
- Similar query: 45ms, $0.0000
- **99.7% faster, 100% cost reduction**

### 3. PII Protection

**Auto-Detection & Encryption:**
- Emails: `john@email.com` → `[EMAIL_a3f2b1c4]`
- SSNs: `123-45-6789` → `[US_SSN_b7e8c2f1]`
- Names: `John Smith` → `[PERSON_d4a9f3e2]`
- Phones: `555-123-4567` → `[PHONE_c8b2e1a5]`

**Security:**
- Fernet encryption (AES-128-CBC with HMAC)
- SHA256 token generation
- Per-user isolation in SQLite

### 4. Complete Observability

**Downloadable Metrics:**
- 📋 `llm_calls.jsonl` - Detailed call logs
- 📈 `query_rollup.xlsx` - Query summary

**Tracks:**
- 8 agent calls per query
- Tokens, costs, latency
- Cache hits/misses
- Refinement attempts
- Confidence scores

---

## System Architecture

### High-Level Flow

```
USER QUERY
    ↓
SEMANTIC CACHE (80% similarity)
    ├─ HIT → Return (45ms) ⚡
    └─ MISS ↓
         ↓
AGENT PIPELINE (8 agents)
    │
    ├─ 1. Safety Guard
    ├─ 2. File Router + ANSWERABILITY CHECK
    ├─ 3. Planner (MULTI-STEP STRATEGY)
    ├─ 4. Data Analyst
    ├─ 5. Insights Generator (INSIGHTS FIRST)
    ├─ 6-8. Critics (Logic, Data, Insights)
    │
    └─ Confidence < 80%? → REFINEMENT (max 2×)
         │
         └─ Re-analyze → Re-generate → Re-validate
    ↓
PII RESTORE → USER
```

[See detailed architecture →](./ARCHITECTURE_DIAGRAM.md)

---

## Installation

```bash
# 1. Clone
git clone https://github.com/Dhananjaya12/Data-Analyst-Agent.git
cd Data-Analyst-Agent

# 2. Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Environment
cp .env.example .env
# Add: GROQ_API_KEY=your_key

# 5. Run
streamlit run app.py
```

---

## Security & Privacy

### PII Redaction Flow

```
1. Upload CSV → Presidio scan
2. Detect: john@email.com, John Smith
3. Tokenize: [EMAIL_a3f2b1c4], [PERSON_d4a9f3e2]
4. Encrypt with Fernet (AES-128)
5. Store in SQLite (per-user isolation)
6. LLM sees tokens only
7. User sees original values (restored)
```

**Why Both hashlib AND cryptography?**
- **hashlib** - Creates SHA256 tokens (`a3f2b1c4`)
- **cryptography** - Encrypts values with Fernet (AES-128)

---

## Observability

### Downloadable Files

**`llm_calls.jsonl` (Line-delimited JSON):**
```json
{
  "call_id": "019e22c3-7c61-7c71-b90d-a90fa40f1390",
  "query_id": "Q-744fd0d9",
  "agent": "SafetyGuard",
  "model": "llama-3.1-8b-instant",
  "timestamp": "2026-05-13T19:14:58.817652",
  "latency_ms": 1574.37,
  "input_tokens": 222,
  "output_tokens": 2,
  "total_tokens": 224,
  "cost_usd": 0.00001126,
  "cache_hit": false,
  "status": "ok",
  "prompt_preview": "You are a security classifier...",
  "response_preview": "SAFE"
}
```

**`query_rollup.xlsx` (Excel Summary):**
- One row per query
- 8 agent call metrics (model, tokens, latency, cost per agent)
- Total tokens, cost, time
- Confidence, refinements, cache hits
- Files used, execution path

---

## Deployment

### Streamlit Cloud

1. Push to GitHub
2. Deploy on [share.streamlit.io](https://share.streamlit.io)
3. Add secret: `GROQ_API_KEY=your_key`
4. Deploy! ✅

---

## Tech Stack

**Core:** Python 3.11, LangGraph 0.2.45, LangChain 0.3.0, Streamlit 1.40.2

**LLMs:** GROQ (llama-3.1-8b-instant, llama-3.3-70b-versatile, openai/gpt-oss-120b)

**ML/NLP:** sentence-transformers (all-MiniLM-L6-v2), Presidio 2.2.355, spaCy 3.8.2 (en_core_web_sm)

**Data:** pandas 2.1.0, openpyxl 3.1.5, python-dotenv 1.0.0

**Storage:** SQLite (PII encrypted database), JSON (semantic cache)

**Security:** 
- `cryptography 44.0.0` - Fernet encryption (AES-128-CBC with HMAC) for PII values
- `hashlib` (built-in) - SHA256 hashing for PII token generation