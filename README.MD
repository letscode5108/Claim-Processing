# 🏥 Claim Processing Pipeline

A FastAPI service that processes medical insurance claim PDFs using **LangGraph** to orchestrate multi-agent document extraction powered by **Groq Vision (Llama 4 Scout)**.

---

## 📌 Overview

Upload a PDF claim → AI classifies every page → Specialist agents extract structured data → Get clean JSON back.

```
POST /api/process  →  Segregator  →  ID Agent
                                  →  Discharge Agent  →  Aggregator  →  JSON
                                  →  Bill Agent
```

---

## 🚀 Features

- **AI-powered page classification** — Groq Vision reads each page and classifies it into 9 document types
- **Multi-agent extraction** — 3 specialist agents each process only their assigned pages
- **Handles scanned PDFs** — works entirely on page images, no text extraction required
- **LangGraph orchestration** — clean state machine with typed state passing between nodes
- **Fast** — Groq's free tier handles the full pipeline with no rate limit issues

---



## ⚙️ Setup

### Prerequisites

- Python 3.11
- Groq API key → [https://console.groq.com](https://console.groq.com)

### Installation

```bash


# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run

```bash
python main.py
```

Server starts at `http://localhost:8000`

---

## 📡 API Usage

### `POST /api/process`

**Request** (multipart/form-data):

| Field | Type | Description |
|---|---|---|
| `claim_id` | string | Unique identifier for the claim |
| `file` | file | PDF file to process |

**Example (curl):**

```bash
curl -X POST http://localhost:8000/api/process \
  -F "claim_id=CLAIM001" \
  -F "file=@/path/to/claim.pdf"
```

**Example Response:**

```json
{
  "claim_id": "CLAIM001",
  "page_classification": {
    "identity_document": [2],
    "discharge_summary": [3],
    "itemized_bill": [8, 9],
    "claim_forms": [0, 14],
    "prescription": [4],
    "investigation_report": [5, 10, 11],
    "cash_receipt": [6],
    "cheque_or_bank_details": [1]
  },
  "extracted_data": {
    "identity": {
      "patient_name": "JOHN MICHAEL SMITH",
      "date_of_birth": "15-MAR-1985",
      "id_number": "ID-987-654-321",
      "policy_number": null,
      "insurance_provider": null,
      "gender": "Male",
      "address": "456 Oak Street, Apt 12B, Springfield, IL 62701"
    },
    "discharge_summary": {
      "hospital_name": "CITY MEDICAL CENTER",
      "admission_date": "January 20, 2025",
      "discharge_date": "January 25, 2025",
      "total_days_admitted": 5,
      "primary_diagnosis": "Community Acquired Pneumonia (CAP)",
      "treating_physician": "Dr. Sarah Johnson, MD",
      "discharge_condition": "Stable, improved"
    },
    "itemized_bill": {
      "items": [...],
      "subtotal": 6379.30,
      "taxes": 317.33,
      "grand_total": 6696.63,
      "currency": "USD"
    }
  }
}
```

### `GET /`

Health check — returns `{"status": "running"}`.

---

## 🧠 How It Works

### LangGraph Workflow

```
START
  │
  ▼
[Segregator Agent]     ← classifies all pages via Groq Vision
  │
  ├──────────────────┬──────────────────┐
  ▼                  ▼                  ▼
[ID Agent]   [Discharge Agent]   [Bill Agent]
  │                  │                  │
  └──────────────────┴──────────────────┘
                     │
                     ▼
               [Aggregator]
                     │
                    END
```

### Segregator Agent

1. Converts every PDF page to a base64 PNG using PyMuPDF
2. Sends each page image to `meta-llama/llama-4-scout-17b-16e-instruct` on Groq
3. Model reads layout, headings, tables and returns `{"doc_type": "...", "confidence": 1.0}`
4. Builds a classification map: `{"identity_document": [2], "itemized_bill": [8, 9], ...}`

### Supported Document Types

| Type | Description |
|---|---|
| `claim_forms` | Insurance claim forms |
| `identity_document` | Govt ID, Aadhaar, Passport |
| `discharge_summary` | Hospital discharge records |
| `itemized_bill` | Itemized hospital / pharmacy bills |
| `prescription` | Doctor prescriptions |
| `investigation_report` | Lab reports, blood work |
| `cash_receipt` | Payment receipts |
| `cheque_or_bank_details` | Bank / cheque documents |
| `other` | Anything not matching above |

### Extraction Agents

Each agent receives **only its assigned pages** (not the full PDF):

| Agent | Pages Used | Extracts |
|---|---|---|
| `id_agent` | `identity_document` pages | Name, DOB, ID number, policy, gender, address |
| `discharge_agent` | `discharge_summary` pages | Hospital, dates, diagnosis, physician, procedures |
| `bill_agent` | `itemized_bill` pages | All line items, subtotal, tax, grand total, currency |

---

## 📦 Dependencies

```txt
fastapi
uvicorn[standard]
python-multipart
langgraph
langchain-core
groq
pymupdf
pydantic
python-dotenv
```

Install all:

```bash
pip install -r requirements.txt
```

---

## 🔑 Getting a Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up / log in
3. Click **API Keys** → **Create API Key**
4. Copy and paste into your `.env` file

Groq free tier: **1000 requests/minute** — more than enough for this pipeline.

---

## 🧪 Testing via Swagger UI

After starting the server, open:

```
http://localhost:8000/docs
```

Click `/api/process` → **Try it out** → upload your PDF → **Execute**

---

## 📝 Notes

- Works best with PDFs that have clear page-level document separation
- Scanned/image-based PDFs are fully supported (no OCR needed — Vision model reads images directly)
- Pages classified as `other` are ignored by extraction agents but included in `page_classification`
- The model used is `meta-llama/llama-4-scout-17b-16e-instruct` — a multimodal model that supports image input up to 20MB

---

