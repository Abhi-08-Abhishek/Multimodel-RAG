# NovaCore Multimodal RAG (Monolithic Architecture)

A production-grade, humanized **Multimodal Retrieval-Augmented Generation (RAG)** application built with **FastAPI** (Python backend) and **Next.js 16 + TypeScript + Tailwind CSS** (Frontend).

This application moves beyond standard text-only RAG by seamlessly indexing and querying across three distinct enterprise modalities:
1. **Narrative Text** (Executive summaries, strategies, commentary).
2. **Structured Tables** (Financial tables, SLA metrics, regional performance - extracted into GitHub-flavored Markdown).
3. **Visual Assets** (Quarterly revenue line graphs, regional bar charts, Penang facility solar pie charts, supply-chain flow diagrams).

---

## 🏛️ System Architecture

```
Multimodal-RAG-NovaCore/
├── backend/                       # Python FastAPI Backend
│   ├── app/
│   │   ├── api/v1/                # REST API Endpoints (/chat, /ingest, /documents, /health)
│   │   ├── core/                  # Pydantic v2 settings, logging, custom domain exceptions
│   │   ├── schemas/               # Request & response validation models
│   │   ├── services/              # Business logic:
│   │   │   ├── pdf_extractor.py   # PyMuPDF text, table markdown & visual extractor
│   │   │   ├── vision_service.py  # Groq Vision LLM summarizer & direct VLM QA
│   │   │   ├── embedding_service.py # SentenceTransformers (all-MiniLM-L6-v2)
│   │   │   ├── vector_store.py    # Pinecone serverless vector database
│   │   │   └── rag_orchestrator.py # Modality classifier & multi-modal answer router
│   │   └── storage/
│   │       ├── uploads/           # Uploaded PDF documents
│   │       └── extracted_images/  # Extracted charts/diagrams served via static mount
│   ├── requirements.txt
│   └── .env                       # API keys & configuration
│
└── frontend/                      # Next.js 16 + TypeScript + Tailwind CSS Frontend
    ├── src/
    │   ├── app/                   # App Router (layout.tsx, page.tsx, globals.css)
    │   ├── components/
    │   │   ├── chat/              # MessageBubble (Markdown + Citations), ChatInput
    │   │   ├── layout/            # Header (status pills), Sidebar (preset queries)
    │   │   ├── visuals/           # VisualModal (High-res zoomable chart lightbox)
    │   │   └── documents/         # IngestionDrawer (Pipeline stepper & stats)
    │   ├── lib/                   # Axios API client & utils
    │   └── types/                 # Shared TypeScript interfaces
    └── package.json
```

---

## ⚡ Multimodal Pipeline Flow

### 1. Ingestion (`POST /api/v1/ingest`)
```
PDF Document
   ├── 1. Text Extractor   ──> LangChain Text Documents
   ├── 2. Table Extractor  ──> Pandas DataFrame ──> Markdown Tables ──> LangChain Table Documents
   └── 3. Image Extractor  ──> PNG on disk ──> Groq Vision LLM ──> Text Summary ──> LangChain Visual Documents
                                                          │
All Documents (Text + Table + Visual Summaries) ──────────┴──> SentenceTransformers (384-d) ──> Pinecone Index
```

### 2. Retrieval & Generation (`POST /api/v1/chat`)
```
User Question ──> Pinecone Similarity Search (Top-5 Chunks)
                       │
                       ▼
            Are any Visual chunks retrieved?
             ├── NO  ──> Groq Text Model (`llama-3.3-70b-versatile`)
             └── YES ──> Groq Vision Model (`llama-3.2-11b-vision-preview`)
                          • Attaches retrieved Context + Question
                          • Attaches original High-Res Extracted Images (Base64)
                               │
                               ▼
        Unified Response with Page Citations & Image Previews
```

---

## 🚀 Getting Started

### 1. Backend Setup

```bash
cd backend

# Create virtual environment (Python 3.12 recommended)
uv venv .venv --python 3.12
# Or: python -m venv .venv

# Activate environment
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Or using uv:
uv pip install -r requirements.txt
```

#### Configure `.env`:
Create/edit `backend/.env` with your API keys:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
PINECONE_API_KEY=pcsk_your_pinecone_api_key_here

PINECONE_INDEX_NAME=novacore-multimodal-rag
PINECONE_NAMESPACE=fy2026-demo
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

TEXT_MODEL=llama-3.3-70b-versatile
VISION_MODEL=llama-3.2-11b-vision-preview
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

#### Start FastAPI Server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be accessible at: `http://localhost:8000/docs`

---

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start Next.js Development Server
npm run dev
```
Open your browser at: `http://localhost:3000`

---

## 🧪 Testing the 7 Practice Notebook Demos

In the frontend UI, click on any of the preset question chips in the left sidebar:

1. **Text Retrieval**: *"What does NovaCore Systems do and where is the company headquartered?"*
2. **Table Query**: *"Which region had the highest year-over-year revenue growth?"*
3. **Revenue Graph**: *"According to the revenue graph, which quarter had the highest revenue?"*
4. **Supply Chain Diagram**: *"According to the supply-chain diagram, what is the critical quality-control point?"*
5. **Support Resolution SLA**: *"How did average support resolution time change from January to August?"*
6. **Sustainability Mix**: *"What percentage of electricity at the Penang facility came from solar?"*
7. **Cross-Modal Synthesis**: *"Give me a short FY2026 performance summary with revenue and margins."*
