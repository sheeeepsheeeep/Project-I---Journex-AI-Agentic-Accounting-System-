# Journex — AI Agentic Accounting System

**Journex** is an automated accounting and error detection web application designed to streamline daily business accounting operations. It combines deterministic accounting rules and exact decimal arithmetic with Large Language Model (LLM) reasoning to automate document collection, data extraction, double-entry journal drafting, anomaly auditing, root-cause investigation, and MFRS 101 financial reporting.

---

## Key Features

- **Company Onboarding & Opening Balances:** Multi-company profile setup, hierarchical Chart of Accounts configuration, and automated extraction of opening balances from setup documents.
- **Multi-Source Document Collection:** Collects financial documents (invoices, receipts, payment vouchers, purchase returns, orders, stock take sheets, and bank transactions) via manual uploads, monitored local folders, connected Gmail inboxes (OAuth2), and bank feeds.
- **Automated Extraction & Front-Door Validation (Agent 1):** Extracts structured fields and line items from PDFs and images while enforcing SHA-256 / document-number duplicate detection, Entity Concept verification, BNM foreign-currency conversion, and deterministic `Decimal` calculation checks.
- **Accounting Classification & Double-Entry Drafting (Agent 2):** Classifies transactions into the Chart of Accounts and Books of Prime Entry (*Sales, Purchases, Returns, Cash Book, General Journal*) and generates balanced double-entry journal drafts.
- **Automated Auditing & Abnormal Balance Shield (Agent 3):** Audits pending journal entries against accounting rules, historical records (RAG), and live General Ledger balances to prevent abnormal negative balances (such as unfunded petty cash deficits or customer overpayments).
- **Error Investigation & Human Review Queue (Agent 4):** Provides root-cause explanations and proposed resolutions in the Audit Queue, allowing users to apply fixes, upload missing supporting proof, override holds with justification, or cancel invalid unposted drafts.
- **General Ledger & Financial Reporting:** Generates chronological General Ledger T-accounts, Trial Balance, Statement of Profit or Loss, and Statement of Financial Position (MFRS 101) with PDF export.

---

## Tech Stack

- **Frontend:** React 18, Vite, Lucide React
- **Backend:** Python, FastAPI, Uvicorn, Pydantic
- **Database:** SQLite (WAL Mode), SQLAlchemy ORM
- **Document & PDF Processing:** `pdfplumber`, `Pillow`, `ReportLab`
- **External Integrations:** Groq LLM API, Bank Negara Malaysia (BNM) Exchange Rate API, Gmail OAuth2 API

---

## Project Structure

```text
Source Codes/
├── AccountingDocument-Ms.Wani/   # Sample PDFs for onboarding, daily documents, and bank transactions
├── backend/                      # FastAPI backend, agents, services, SQLite models, and pytest suite
│   ├── app/
│   ├── tests/
│   └── requirements.txt
├── frontend/                     # React + Vite web dashboard
│   ├── src/
│   └── package.json
└── README.md
```

---

## Installation & Setup Guide

### Prerequisites
- **Python** 3.10+
- **Node.js** 18+ and `npm`

### 1. Backend Setup (FastAPI)

Open a terminal in the project root and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

Create a `.env` file inside `backend/` (optional for AI audit features):

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3-32b
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

Start the FastAPI backend server:

```powershell
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
The backend API will run at `http://127.0.0.1:8000` (Swagger docs at `http://127.0.0.1:8000/docs`).

*(Optional)* To run the local mock bank feed server on port `8010`:

```powershell
cd backend
python mock_bank_server.py
```

### 2. Frontend Setup (React + Vite)

Open a second terminal and run:

```powershell
cd frontend
npm install
npm run dev
```
Open your browser and navigate to the local URL shown in the terminal (typically `http://localhost:5173` or `http://localhost:3000`).

---

## Sample Documents for Testing

Sample testing documents are provided in [`AccountingDocument-Ms.Wani/`](./AccountingDocument-Ms.Wani):
- **`Start-Up-Documents/`**: Use these 5 PDFs during Step 2 of **Company Onboarding** to populate the Opening Trial Balance, Stock, Receivables, Payables, and Other Creditors.
- **`DocumentUpload-Ms.Wani/`**: Use these 13 operational PDFs to test daily document uploads or local folder collection.
- **`BankTransaction-Ms.Wani/`**: 18 sample bank transaction PDFs (`TXN-0001` to `TXN-0018`).
