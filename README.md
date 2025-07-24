# 🤖 LLM-Contract-Analyzer

LLM-Contract-Analyzer is a system designed to answer user queries about policies, contracts, or legal documents using natural language. The system reads documents (such as PDFs), understands user questions, finds the most relevant sections, and returns a structured answer with justification.

## 🧩 Core Features

- Accepts plain-English queries (e.g., "Is knee surgery covered in a 3-month-old policy?")
- Extracts and embeds document text for semantic search
- Retrieves the most relevant clauses based on query meaning
- Uses a Large Language Model (LLM) to reason over the results
- Outputs a clear decision (e.g., Approved/Rejected) with referenced clauses

## ⚙️ Workflow Overview

1. **Embed documents** into vector space using `SentenceTransformer`
2. **Store & search** chunks using FAISS
3. **Accept query**, retrieve relevant clauses
4. **LLM interprets** the query and supporting clauses
5. **Return structured JSON**: decision, amount, and justification

## 🚀 Key Technologies

- Python
- PyMuPDF
- SentenceTransformers
- FAISS
- LLMs via OpenRouter or Together.ai

