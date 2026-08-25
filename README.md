# Aster & Row AI Support Agent

An AI customer support agent built for Aster & Row, a fictional ecommerce store selling bags, drinkware, and travel accessories. The agent handles customer queries about company policies, product details, and order tracking while following strict grounding and safety guardrails.

---

## Features

- **Grounded Policy Answers**: Retrieves active, customer-facing policy documents using BM25 and includes exact source citations (`[filename # Heading]`).
- **Policy Precedence & Filtering**: Ignores superseded policies and internal migration drafts.
- **Conflict Handling**: Detects conflicting guidance in active documents (e.g., product care vs. product card) and escalates to human support with safe interim advice.
- **Verified Order Lookup**: Looks up order status from `data/orders.json` by order ID, normalizes input IDs, and handles missing or invalid IDs safely.
- **Privacy Guardrails**: Protects customer personal information (email, address) and internal metrics (risk scores, notes) from disclosure.
- **Order Lifecycle Logic**: Suppresses stale delivery dates for cancelled/returned orders and avoids fabricating delivery dates for shipments without an ETA.
- **Multi-Turn Memory**: Preserves session context and remembers order IDs across turns.
- **Prompt Injection Defense**: Rejects instructions inside retrieved drafts or user prompts attempting to override store policy.
- **Offline / Deterministic Fallback**: Works completely offline using a deterministic synthesis engine, and can optionally connect to Google Gemini when an API key is provided.
- **Streamlit Web UI**: Simple, clean chat interface to test and demonstrate the agent.

---

## Project Structure

```text
aster-row-ai-agent/
├── README.md                           # Project documentation
├── assign_README.md                    # Assignment specification
├── requirements.txt                    # Python dependencies
├── .env.example                        # Example environment variables
├── .gitignore                          # Git ignore rules
├── app.py                              # Streamlit web chat interface
├── run_evaluations.py                  # Evaluation benchmark runner
│
├── src/                                # Core source code
│   ├── __init__.py                     # Package exports
│   ├── config.py                       # Paths and system prompt configuration
│   ├── rag_engine.py                   # Document indexing, BM25 search & conflict detection
│   ├── order_tool.py                   # Order lookup and data sanitization
│   ├── agent.py                        # Intent routing, multi-turn session & guardrails
│   └── llm.py                          # Gemini provider abstraction & safe fallback
│
├── knowledge-base/                     # Markdown policies and product cards
│   ├── 01-returns-policy-current.md
│   ├── 02-returns-policy-legacy.md     # (Superseded)
│   ├── 03-final-sale-and-promotions.md
│   ├── 04-damaged-or-wrong-items.md
│   ├── 05-domestic-shipping.md
│   ├── 06-international-shipping.md
│   ├── 07-warranty.md
│   ├── 08-order-changes-and-cancellations.md
│   ├── 09-trailplus-membership.md
│   ├── 10-gift-cards-and-price-adjustments.md
│   ├── 11-product-care.md
│   ├── 12-breeze-tumbler-product-card.md
│   ├── 13-support-escalation.md
│   └── 14-internal-content-migration-notes.md # (Internal draft)
│
├── data/                               # Mock ecommerce data
│   ├── orders.json                     # Order records
│   └── orders-data-dictionary.md       # Data schema description
│
└── evaluation/                         # Evaluation suite
    ├── visible-cases.json              # 15 provided test cases
    ├── custom-cases.json               # 6 original edge test cases
    ├── evaluate.py                     # Deterministic test runner
    └── eval_results.json               # Test results output artifact
```

---

## Architecture Overview

```
User Query
    │
    ▼
Agent Core (Intent Detection & Session Context)
    │
    ├─► Order Query  ──► Order Lookup Tool (Sanitizes PII & internal data)
    │
    └─► Policy Query ──► RAG Engine (BM25 search + Metadata filter + Conflict check)
    │
    ▼
Guardrails & Synthesis (Gemini LLM or Deterministic Fallback)
    │
    ▼
Final Response + Citations + Handoff Flag (if needed)
```

---

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the repository
```powershell
git clone https://github.com/saumya-247/aster-row-ai-agent.git
cd aster-row-ai-agent
```

### 2. Create and activate a virtual environment
```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv venv
.\venv\Scripts\activate.bat

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure optional API key
The agent works completely offline without an API key using the built-in deterministic engine. If you want to enable live Gemini LLM generation:
```powershell
# Copy .env.example to .env
cp .env.example .env

# Open .env and add your Gemini API key:
# GEMINI_API_KEY=your_key_here
```

---

## How to Run

### 1. Run the Web UI
```powershell
python -m streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 2. Run the Evaluation Suite
```powershell
python run_evaluations.py
```

### 3. Run Individual Component Tests
```powershell
python src/agent.py         # Agent core self-tests
python src/rag_engine.py    # RAG engine tests
python src/order_tool.py    # Order lookup security tests
```

---

## Evaluation Results

The evaluation suite runs all 15 supplied visible cases plus 6 custom edge cases (21 total cases) against deterministic assertions:

```text
================================================================================
           ASTER & ROW AI AGENT - COMPREHENSIVE EVALUATION SUITE            
================================================================================
Total Cases: 21 | Passed: 21 | Failed: 0
Overall Pass Rate: 100.0%

CATEGORY BREAKDOWN:
- abstention:               1/1 passed (100.0%)
- conversation:             2/2 passed (100.0%)
- groundedness:             3/3 passed (100.0%)
- multi-source-grounding:   1/1 passed (100.0%)
- policy-boundary:          2/2 passed (100.0%)
- privacy:                  1/1 passed (100.0%)
- prompt-security:          1/1 passed (100.0%)
- retrieval:                2/2 passed (100.0%)
- source-conflict:          1/1 passed (100.0%)
- tool-reliability:         4/4 passed (100.0%)
- tool-use:                 3/3 passed (100.0%)
================================================================================
```

Full structured results are exported to `evaluation/eval_results.json`.

---

## Bug Diary

Here are three real issues encountered during development and how they were fixed:

### 1. Active Document Conflict on Tumbler Care
- **Issue:** Query `"Can I put the entire Breeze Tumbler in the dishwasher?"` returned only one document based on raw BM25 score, ignoring that `11-product-care.md` and `12-breeze-tumbler-product-card.md` directly contradict each other (hand-wash body vs. dishwasher safe).
- **Fix:** Added `check_active_conflicts()` in `src/rag_engine.py` to scan retrieved active official chunks for contradictory cleaning guidance, cite both sources, provide safe interim advice (hand-wash body), and flag `requires_handoff=True`.
- **Regression Test:** `genuine-active-source-conflict` in `evaluation/visible-cases.json`.

### 2. Intent Routing False Positive on Damaged Item Reports
- **Issue:** Query `"A final-sale bag arrived with a broken zipper yesterday. Am I completely out of luck?"` was mistakenly classified as an order lookup because the word `"arrived"` matched an `"arrive"` keyword check. Since no order ID was in the prompt, the agent asked for an order ID instead of answering the policy question.
- **Fix:** Replaced single-word keyword matching with full phrase matching (`"where is my order"`, `"when will my order arrive"`) and prioritized damaged-item policy detection.
- **Regression Test:** `final-sale-damaged-exception` in `evaluation/visible-cases.json`.

### 3. Stale Estimated Delivery Date on Cancelled Orders
- **Issue:** Looking up cancelled order `ORD-1004` returned an `estimated_delivery` date (`2026-08-16`) that remained in the mock data from when the order was first created.
- **Fix:** Added sanitization in `src/order_tool.py` to explicitly clear `estimated_delivery` when an order's status is `cancelled` or `returned`.
- **Regression Test:** `cancelled-order-stale-eta` in `evaluation/visible-cases.json`.

---

## Known Limitations

1. **Read-Only Operations**: The agent can look up order details and explain policies, but cannot perform write operations (e.g., updating addresses in a database or initiating refunds). It recommends human support for actions requiring account changes.
2. **In-Memory Session State**: Multi-turn conversation history is stored in memory per session and resets when the server or session restarts.
3. **Mock Authentication**: Following assignment guidelines, possessing the order ID is treated as sufficient authentication to view non-PII order details.

---

## AI Tool Usage

- AI coding assistants (Gemini / Antigravity) were used for code generation, drafting test assertions, and structuring evaluation scripts.
- **Example Correction**: An early AI suggestion recommended adding a vector database library with dense embeddings (e.g., Chroma/FAISS). However, for a small, structured markdown corpus with explicit front matter metadata (status, audience, authority), a deterministic BM25 retriever combined with metadata filtering proved lighter, faster, zero-dependency, and fully reproducible.

---

## Demo

A short demo video of project demonstration, including the application workflow, evaluation results, and Streamlit UI is added here.
[Watch the Project Demonstration Video](https://drive.google.com/file/d/1af9615IWUNgCNCS-iG7jMm25fSKkSfGE/view?usp=sharing)
