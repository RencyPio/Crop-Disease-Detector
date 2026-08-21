# Crop Disease Detector — Day-by-Day Build Plan
### Deep Learning + RAG + Agentic AI + Automation

**A note on accuracy:** This plan only uses tools, APIs, and datasets that are real and verifiable as of today. Before you start each day, if you want me to check a library version, API pricing, or dataset link, ask me to **search the web** first rather than relying on memory — APIs and free-tier terms change often. Wherever I wasn't fully sure of a detail, I've marked it below with ⚠️ so you know to verify it yourself before building on top of it.

**Your setup:**
- **Local machine:** Windows 11, AMD Ryzen 7 5700U, 16GB RAM, AMD integrated graphics (no NVIDIA GPU) — fine for everything except CNN training
- **Code editor:** VS Code (local) — this is where you write, manage, and version all project code (RAG, agent, automation, integration)
- **CNN training platform:** Kaggle Notebooks (free T4 x2 GPU, 30 hrs/week quota) — PlantVillage dataset is already hosted there, so no manual download/upload needed
- **Backup training platform:** Google Colab (free T4, ~12 hr sessions) — only if Kaggle's weekly GPU quota runs out
- **Workflow:** write/prototype code in VS Code → copy training script into a Kaggle Notebook to actually run training → download the trained model file (`.h5`/`.pt`) back to your laptop → continue building RAG/agent/automation locally in VS Code, loading the trained model for inference (fast, no GPU needed)

---

## Week 1: Deep Learning Core (Image Classifier)

### Day 1 — Environment Setup + Dataset Download
**Do:**
- In VS Code: install Python extension + Jupyter extension, set up a Python virtual environment for the project
- Install locally (for everything except training): `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- Create a free Kaggle account — this is where CNN training will actually run (Day 3-4), not on your laptop
- On Kaggle: create a new Notebook, add the PlantVillage dataset via "Add Data" (searchable directly, no download needed), turn on GPU (T4 x2) in Notebook settings
- Explore folder structure — count classes, count images per class, view a few sample images (can do this either in VS Code with a small local sample, or directly in the Kaggle notebook)

**What to tell me next session:**
> "I've set up VS Code and created my Kaggle notebook with PlantVillage added. Here's my folder structure: [paste folder tree]. Help me write the data-loading script."

---

### Day 2 — Data Preprocessing
**Do:**
- Learn: image resizing, normalization, why train/val/test split matters, data augmentation (rotation, flip, zoom) and why it prevents overfitting
- Write this preprocessing script in VS Code first (easier to edit/debug), using a small local sample of images if you have one, or just write it against the known PlantVillage folder structure
- Build a data pipeline: resize images (e.g., 224x224 for MobileNet), split into train/val/test (e.g., 70/15/15), apply augmentation to training set only
- Once the script looks correct, this is what you'll paste into the Kaggle notebook on Day 3-4 to actually run against the full dataset

**What to tell me next session:**
> "Here's my data split code written in VS Code: [paste code]. Review it and help me add data augmentation."

---

### Day 3 — Model Building (Transfer Learning)
**Do:**
- Learn: what transfer learning is, why MobileNetV2/EfficientNet (pretrained on ImageNet) works well here, freezing vs. fine-tuning layers
- Draft the model architecture code in VS Code: load a pretrained MobileNetV2 (no top layer), add custom classification head for your number of classes, compile model (optimizer, loss function, metrics)
- Copy the finished script into your Kaggle Notebook (GPU already enabled from Day 1) to prepare for training

**What to tell me next session:**
> "I want to build a transfer learning model with MobileNetV2 for [X] classes. Here's my current code written in VS Code: [paste]. Help me set up the architecture before I move it to Kaggle."

---

### Day 4 — Training the Model (on Kaggle, not locally)
**Do:**
- Run everything in this step inside your **Kaggle Notebook** (GPU-enabled) — your laptop's integrated AMD graphics can't accelerate this, so training locally would be far too slow
- Train the model on a subset first (e.g., 5 classes) to confirm the pipeline works before running full training (saves GPU quota debugging)
- Learn: epochs, batch size, learning rate, callbacks (early stopping, model checkpointing)
- Run full training once subset works; keep an eye on your 30 hrs/week Kaggle GPU quota
- Once training completes, download the trained model file (`.h5`/`.pt`) from Kaggle to your laptop for use in later steps

**What to tell me next session:**
> "My model trained on Kaggle on a subset with [X]% accuracy. Here's my training code and loss/accuracy curves: [paste/describe]. Help me debug/improve before full training."

---

### Day 5 — Evaluation + Real-World Test Images
**Do:**
- Evaluate on test set (can do this on Kaggle right after training, or locally in VS Code after downloading the model — inference is fast, no GPU needed): accuracy, precision, recall, F1, confusion matrix
- Learn: why accuracy alone isn't enough (class imbalance issues)
- Test model on PlantDoc dataset images + your own phone photos to check real-world generalization (lab data ≠ real data) — this part is easiest done locally in VS Code with your downloaded model

**What to tell me next session:**
> "Here are my evaluation metrics and confusion matrix: [paste/describe]. Also tested on [X] real-world photos with these results: [describe]. Help me interpret this and decide if I need more training or data augmentation."

---

## Week 2: RAG Layer (Treatment Knowledge Retrieval)
*(Everything from here on runs locally in VS Code on your laptop — no GPU needed, your 16GB RAM handles this fine.)*

### Day 6 — Build the Knowledge Base
**Do:**
- Collect 10-15 real agricultural PDFs/documents (ICAR advisories, FAO guides, university extension publications) covering diseases in your dataset's classes
- **Do NOT let me invent document content or URLs from memory** — ask me to web-search for real ICAR/FAO documents so we only use verified sources

**What to tell me next session:**
> "Search the web for real ICAR/FAO/university PDF guides on [specific disease, e.g., Tomato Late Blight] treatment. I need 2-3 real, verifiable sources."

---

### Day 7 — Chunking + Embeddings
**Do:**
- Extract text from PDFs (PyPDF2 or pdfplumber)
- Chunk text (~200-300 words with overlap)
- Generate embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`, free, local, no API key)

**What to tell me next session:**
> "Here's my PDF text extraction output: [paste sample]. Help me write the chunking + embedding script."

---

### Day 8 — Vector DB + Retrieval
**Do:**
- Set up ChromaDB (free, local)
- Store chunks with metadata (disease name, source document)
- Write retrieval function: given a disease name, fetch top-k relevant treatment chunks

**What to tell me next session:**
> "I've embedded my documents into ChromaDB. Here's my retrieval function: [paste]. Test it with me on a query like 'Tomato Late Blight treatment' and check if results look relevant."

---

### Day 9 — LLM Integration for Grounded Advice
**Do:**
- Set up Groq API key (free tier, no card needed) — ⚠️ verify current free-tier limits before relying on them, as free-tier terms can change
- Write a prompt that takes retrieved chunks + disease name → generates a clear, farmer-friendly treatment recommendation, explicitly instructed to only use the provided context

**What to tell me next session:**
> "Here's my RAG prompt and Groq API call: [paste]. Test with disease X and check if the output is grounded in retrieved chunks and not hallucinated."

---

### Day 10 — Connect DL Output to RAG Input
**Do:**
- Chain the pipeline: image → CNN prediction (disease label) → RAG retrieval → LLM-generated advice
- Test end-to-end on 5-10 sample images

**What to tell me next session:**
> "I've connected my CNN output to the RAG pipeline. Here's the full chain code: [paste]. Here's the output on test image X: [paste]. Help me debug/refine."

---

## Week 3: Agentic AI Layer

### Day 11 — Learn Agentic AI Concepts
**Do:**
- Learn: difference between a single LLM call, RAG, and an agent (agents make multi-step decisions, can call multiple tools, and adapt based on intermediate results)
- Learn: basic agent frameworks — LangChain agents or a simple custom function-calling loop (no need for heavy frameworks at this scale)

**What to tell me next session:**
> "Explain agent frameworks I could use for a simple 2-3 step decision agent (weather check → decide action → trigger alert). Keep it simple, don't over-engineer."

---

### Day 12 — Weather API Integration
**Do:**
- Sign up for OpenWeatherMap (free, no credit card, just email) — use the **basic Free Weather API access path** (current weather, 3-hour forecast, geocoding), which gives 60 calls/minute and 1,000,000 calls/month, more than enough for this project
- Avoid the separate "One Call 3.0/4.0" product — it's a metered subscription-style feature (1,000 free calls/day, then billed), not needed for your use case
- Write a function: given location (lat/long), fetch current/forecasted weather (rain probability, next 6-24 hrs)
- *(Verified as of Aug 2026 — confirm again on openweathermap.org/price if it's been a while since you read this.)*

**What to tell me next session:**
> "I have a weather API key from [provider]. Help me write a function to check rain forecast for the next 6 hours given lat/long."

---

### Day 13 — Build the Agent Decision Logic
**Do:**
- Write agent logic: after RAG generates treatment advice, agent checks weather → if rain forecasted soon and treatment involves spraying, agent delays the recommendation and explains why → otherwise proceeds
- This is your "multi-step autonomous decision" — document this clearly, it's your key differentiator for viva

**What to tell me next session:**
> "Here's my agent decision function: [paste]. Walk through the logic with me and check if it correctly handles the case where rain is forecasted vs. not."

---

### Day 14 — Test the Full Agent Pipeline
**Do:**
- Run the full chain: image → CNN → RAG → agent (weather check + decision) → final recommendation
- Test with different weather scenarios (mock rain vs. no rain) to confirm branching logic works

**What to tell me next session:**
> "Here's my full pipeline test with 2 scenarios (rain/no rain): [paste outputs]. Help me verify this is working as intended."

---

## Week 4: Automation Layer

### Day 15 — Telegram Bot Setup
**Do:**
- Create bot via @BotFather on Telegram (free, ~5 mins)
- Write function to send a message via Telegram Bot API

**What to tell me next session:**
> "I've created my Telegram bot and have the token. Help me write a Python function to send a message with the diagnosis + advice."

---

### Day 16 — Twilio SMS/WhatsApp Setup
**Do:**
- Sign up for a free Twilio trial account — no credit card required, expires after 30 days
- Free trial includes: 100 SMS messages, 100 WhatsApp messages, 3,000 emails, 75 voice minutes (product-specific free units, not a shared dollar balance)
- For WhatsApp: activate the **Twilio Sandbox for WhatsApp** (in the legacy Console) — it uses a shared Twilio number, and you don't need your own WhatsApp Business Account to test
- **Important trial limitation:** you can only send messages to phone numbers you've manually verified in the Twilio console (for SMS) or that have joined your sandbox by texting the join code (for WhatsApp) — so verify your own number (and maybe 1-2 others) *before* Day 18's full pipeline test, since your demo/viva will need working recipient numbers
- Write function to send SMS/WhatsApp message with diagnosis + treatment steps
- *(Verified as of Aug 2026 — confirm again on twilio.com/docs/usage/trials if it's been a while since you read this.)*

**What to tell me next session:**
> "I've set up my Twilio trial account. Here's my account SID/setup (not sharing actual keys). Help me write the send-message function."

---

### Day 17 — Dashboard Logging (Optional but Recommended)
**Do:**
- Simple logging: store each detection (timestamp, disease, location if available, confidence score) in a CSV or lightweight database (SQLite)
- Optional: build a simple dashboard (Streamlit) to visualize disease detections over time/location

**What to tell me next session:**
> "Help me set up a simple SQLite table to log detections, and a basic Streamlit dashboard to display them."

---

### Day 18 — Full Pipeline Integration
**Do:**
- Chain everything: photo upload → CNN prediction → RAG advice → agent decision (weather check) → automated SMS/WhatsApp/Telegram alert → dashboard log
- Test end-to-end with 5-10 real/sample images

**What to tell me next session:**
> "Here's my full integrated pipeline code: [paste or describe structure]. Help me test it end-to-end and fix any breaks in the chain."

---

## Week 5: Evaluation, Polish, and Report

### Day 19 — Manual Evaluation of RAG + Agent Outputs
**Do:**
- Since there's no ground-truth for "correct treatment advice," manually review 15-20 outputs yourself
- Score them: is the advice accurate? grounded in retrieved docs? does the agent's weather decision make sense?
- Calculate a rough precision/accuracy metric for your report

**What to tell me next session:**
> "Here are my manual evaluation notes on 15 test cases: [paste/describe]. Help me turn this into a proper evaluation metrics table for my report."

---

### Day 20 — Architecture Diagram + Documentation
**Do:**
- Draw the full system architecture (I can generate this as a diagram if you ask)
- Write a "why" log: why CNN over other models, why RAG over plain LLM, why agent over plain automation

**What to tell me next session:**
> "Help me create an architecture diagram for my full pipeline: [describe components]. Also help me write the methodology section explaining each design choice."

---

### Day 21 — Final Report + Resume/Presentation Prep
**Do:**
- Compile results, screenshots, metrics into your project report
- Prepare your viva defense — be ready to explain DL vs RAG vs Agentic AI vs Automation clearly (you already have this breakdown)

**What to tell me next session:**
> "Help me draft my final report structure/abstract using all the results and decisions from this project."

---

## Week 6: Production Deployment
*(Claude Code in VS Code is a good fit for this whole week — lots of multi-file work: API routes, Dockerfile, deployment config. Ask it to create/edit files directly rather than pasting code back and forth.)*

### Day 22 — Wrap the Model in a REST API
**Do:**
- Learn: what a REST API is, why FastAPI is a good fit for ML serving (async, lightweight, auto-generates docs)
- Build a FastAPI app with a `/predict` endpoint: accepts an image upload, runs it through your CNN, returns disease + confidence score
- Add basic input validation (reject non-image files, reject corrupted uploads)
- Test locally with a few sample images (curl or FastAPI's built-in Swagger UI at `/docs`)

**What to tell me/Claude Code next session:**
> "Help me build a FastAPI app with a /predict endpoint that loads my trained model [path] and returns disease + confidence for an uploaded image."

---

### Day 23 — Connect RAG + Agent + Automation to the API
**Do:**
- Extend the API: after `/predict` returns a disease label, chain it into your RAG retrieval → agent decision (weather check) → automated alert, all within the same request/response flow (or as a background task if it's slow)
- Add environment variables (`.env` file) for all your API keys (Groq, Twilio, OpenWeatherMap, Telegram) — never hardcode keys in code; add `.env` to `.gitignore` if using Git
- Add basic logging: log every prediction, retrieval, and agent decision to a file or SQLite table (you already planned this in Day 17 — connect it here)

**What to tell me/Claude Code next session:**
> "Help me wire my RAG + agent + automation pipeline into the FastAPI endpoint, and set up environment variables for my API keys."

---

### Day 24 — Containerize with Docker
**Do:**
- Learn: what Docker/containers are and why they matter (consistent environment, easy deployment, ties into your Cloud Computing/MLOps coursework)
- Write a `Dockerfile` for your FastAPI app (base Python image, install dependencies, copy code, expose port, run with `uvicorn`)
- Build and run the container locally to confirm it works identically to your local (non-Docker) version

**What to tell me/Claude Code next session:**
> "Help me write a Dockerfile for my FastAPI app and test building/running it locally."

---

### Day 25 — Deploy to a Free Hosting Platform
**Do:**
- Pick a free hosting option — **Hugging Face Spaces** (very easy for ML demos, supports Docker) or **Render** (free tier, good for FastAPI apps) — verify current free-tier limits before committing, since these change
- Push your Dockerized app and get a live public URL
- Test the deployed API with real requests (not just localhost)

**What to tell me/Claude Code next session:**
> "I want to deploy this Dockerized FastAPI app to [Hugging Face Spaces / Render]. Help me set up the deployment config."

---

### Day 26 — Build a Simple Frontend
**Do:**
- Build a lightweight frontend so anyone can use it via a link, not just API calls — **Streamlit** is fastest to build (image upload widget, calls your API, displays results)
- Keep it mobile-friendly and low-bandwidth conscious (compress images before upload, show loading indicators)
- Deploy the frontend alongside or pointing to your hosted API

**What to tell me/Claude Code next session:**
> "Help me build a simple Streamlit frontend that uploads an image to my deployed API and displays the diagnosis + treatment advice."

---

### Day 27 — Production Considerations Write-Up (for your report)
**Do:**
- Document what you built (FastAPI + Docker + hosted deployment + frontend) as your "production-ready prototype"
- Write a **"Future Work / Production Roadmap"** section covering what a *fully* production system would still need: larger/retrained datasets, model versioning (MLflow/DVC), multi-language support, native mobile app, monitoring dashboards, moving off Twilio trial/free tiers, rate limiting, queue-based processing for scale
- This section shows examiners you understand the real gap between a working prototype and true production — you don't need to build all of it, just show you know it exists

**What to tell me next session:**
> "Help me write the Future Work section covering production considerations beyond what I've built."

---

## General Rule for Every Session
When you come back to continue, **always tell me:**
1. Which Day/step you're on
2. Paste your actual current code/output (don't describe from memory — paste real output)
3. Any error messages exactly as shown
4. What specifically you want help with (debug / explain / extend)

This keeps me grounded in what you've actually built rather than guessing or assuming progress that hasn't happened — which is what keeps this plan accurate instead of hallucinated.
