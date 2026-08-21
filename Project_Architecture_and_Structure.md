# Crop Disease Detector — Project Architecture & Structure
### Deep Learning + RAG + Agentic AI + Automation

---

## 1. Directory Layout

```
Crop_Disease_Detector/
│
├── .gitignore
├── README.md
├── requirements.txt
├── .env                          # API keys (NOT committed — in .gitignore)
│
├── docs/
│   ├── Crop_Disease_Detector_DayByDay_Plan.md
│   ├── Project_Architecture_and_Structure.md   (this file)
│   └── architecture_diagram.png                 # add once you generate one
│
├── data/
│   ├── real_world_test_images/    # your self-collected phone photos (gitignored)
│   └── knowledge_base/            # ICAR/FAO/university PDFs for RAG
│       ├── raw_pdfs/
│       └── processed_chunks/      # optional: cached chunked text
│
├── notebooks/
│   └── training/                  # copies of Kaggle notebooks/scripts for reference
│       ├── preprocessing.py
│       └── model_training.ipynb   # or .py, downloaded from Kaggle
│
├── models/
│   └── crop_disease_cnn.h5        # trained model file (gitignored — too large for GitHub)
│
├── src/
│   ├── __init__.py
│   │
│   ├── dl/                        # Deep Learning module
│   │   ├── preprocessing.py       # data loading, augmentation, class weights
│   │   ├── model.py               # MobileNetV2 architecture definition
│   │   ├── train.py               # training loop (run on Kaggle)
│   │   └── evaluate.py            # metrics, confusion matrix, real-world test eval
│   │
│   ├── rag/                       # RAG module
│   │   ├── chunker.py             # PDF text extraction + chunking
│   │   ├── embed.py               # sentence-transformers embedding generation
│   │   ├── vector_store.py        # ChromaDB setup + storage
│   │   ├── retrieve.py            # top-k retrieval logic
│   │   └── generate.py            # Groq LLM call for grounded advice
│   │
│   ├── agent/                      # Agentic AI module
│   │   ├── weather_check.py       # OpenWeatherMap API call
│   │   └── decision_agent.py      # multi-step decision logic (spray timing, etc.)
│   │
│   ├── automation/                 # Automation module
│   │   ├── telegram_bot.py
│   │   ├── twilio_alert.py
│   │   └── logger.py              # SQLite/CSV logging of detections
│   │
│   └── api/                        # Production API layer (Week 6)
│       ├── main.py                 # FastAPI app entrypoint
│       └── routes.py               # /predict endpoint, request/response schemas
│
├── frontend/
│   └── app.py                      # Streamlit frontend
│
├── deployment/
│   ├── Dockerfile
│   └── docker-compose.yml          # optional, if multiple services
│
└── tests/
    ├── test_preprocessing.py
    ├── test_rag_retrieval.py
    └── test_agent_decision.py
```

**Note:** You don't need to build this whole structure on Day 3 — it's the target shape to grow into as you complete each week of your plan. Right now you likely just have `preprocessing.py` at the root; that's fine, you can reorganize into `src/dl/` etc. once more files exist (Claude Code can help restructure without breaking anything).

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER (Farmer)                            │
│              Uploads photo via Streamlit frontend                │
└───────────────────────────┬───────────────────────────────────────┘
                             │ image
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND (/predict)                   │
└───────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 — DEEP LEARNING (CNN)                                    │
│  MobileNetV2 (transfer learning, trained on PlantVillage)         │
│  Input: leaf image  →  Output: disease label + confidence score   │
└───────────────────────────┬───────────────────────────────────────┘
                             │ disease label (e.g., "Tomato___Late_blight")
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2 — RAG (Retrieval-Augmented Generation)                   │
│  1. Embed query (disease name) — Sentence-Transformers            │
│  2. Retrieve top-k relevant chunks — ChromaDB                     │
│     (from ICAR/FAO/university treatment documents)                │
│  3. Generate grounded advice — Groq LLM (Llama 3.x)                │
│  Output: treatment recommendation text                            │
└───────────────────────────┬───────────────────────────────────────┘
                             │ treatment advice
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3 — AGENTIC AI (Decision Layer)                            │
│  1. Call Weather API (OpenWeatherMap) — check rain forecast        │
│  2. Decide: does advice involve spraying? Is rain imminent?        │
│  3. Branch: proceed as-is  OR  modify advice (delay spraying)      │
│  Output: final, context-aware recommendation                      │
└───────────────────────────┬───────────────────────────────────────┘
                             │ final recommendation
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4 — AUTOMATION                                              │
│  1. Send Telegram message (instant notification)                  │
│  2. Send SMS/WhatsApp via Twilio                                   │
│  3. Log detection (disease, confidence, timestamp, GPS) → SQLite   │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow summary:** Image → CNN (DL) → Disease Label → RAG (grounded advice) → Agent (weather-aware decision) → Automation (alerts + logging)

---

## 3. Agent Description

### Agent Name: **Spray Decision Agent**

**Type:** Rule-guided, tool-using decision agent (not a fully autonomous/open-ended agent — scoped deliberately for a college project, but genuinely multi-step and adaptive, which is what distinguishes it from plain automation)

**Goal:** Ensure the treatment advice given to the farmer is not just accurate (from RAG), but also **actionable and timing-aware** — i.e., don't tell a farmer to spray pesticide right before rain washes it away.

**Inputs:**
- Disease label + RAG-generated treatment advice (from Stage 2)
- Farmer's location (lat/long, from frontend or device GPS if available)

**Tools available to the agent:**
1. `weather_check(lat, long)` → calls OpenWeatherMap API, returns rain probability for next 6-24 hrs
2. `notify(channel, message)` → calls Telegram/Twilio to send the final message

**Decision logic (the "multi-step" part):**
```
1. Receive treatment advice from RAG
2. Does the advice mention spraying/application of a treatment? (keyword check e.g. "spray", "apply", "fungicide")
   → If NO: skip weather check, proceed directly to notify()
   → If YES: call weather_check(lat, long)
       → If rain probability > threshold (e.g., 60%) in next 6 hrs:
           - Modify advice: append a note recommending delay until after rain
       → Else:
           - Proceed with original advice unchanged
3. Call notify() with the final (possibly modified) advice
4. Log the decision (including whether weather modified the recommendation) for later review
```

**Why this counts as "agentic" rather than plain automation:**
- It makes a **decision** based on retrieved external information (weather), not just executing a fixed script
- It **adapts its output** conditionally rather than always doing the same thing
- It **chains multiple tool calls** (weather API → decision → notification) autonomously, without a human in the loop between steps

**Limitations (be upfront about these in your report):**
- Currently rule-based branching (keyword + threshold check), not a fully autonomous LLM-driven agent making open-ended tool choices — this is a deliberate scoping decision for reliability and explainability within a college project timeline
- Future work: could use an LLM function-calling loop to let the agent dynamically decide *which* tools to call and in what order, rather than a fixed sequence

---

## 4. Project Structure Summary (for your report/README)

| Layer | Technology | Role |
|---|---|---|
| **Deep Learning** | TensorFlow/Keras, MobileNetV2 (transfer learning) | Classifies leaf image into disease category |
| **RAG** | Sentence-Transformers, ChromaDB, Groq (Llama 3.x) | Retrieves and generates grounded treatment advice |
| **Agentic AI** | Custom decision logic + OpenWeatherMap API | Makes context-aware, multi-step decisions (e.g., spray timing) |
| **Automation** | Telegram Bot API, Twilio API, SQLite | Delivers alerts and logs detections automatically |
| **API Layer** | FastAPI | Exposes the full pipeline as a callable service |
| **Frontend** | Streamlit | Lets users upload images and view results |
| **Deployment** | Docker, Hugging Face Spaces / Render | Makes the app publicly accessible |
| **Version Control** | Git, GitHub | Tracks all code changes |

---

*This document is a living reference — update it as your actual folder structure evolves during the build (e.g., once you reorganize `preprocessing.py` into `src/dl/`).*
