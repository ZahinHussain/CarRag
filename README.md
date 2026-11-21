# EV Brochure Chat (RAG)

Chat with electric car brochures. Ask about range, battery, charging, boot space or trims, and get answers with page citations from the original PDFs.

Runs fully locally: no API keys, no cloud.

## How it works

```
PDFs ──► PyMuPDF (text per page) ──► chunks (~300 words)
     ──► qwen3-embedding:0.6b (Ollama) ──► Qdrant (vectors + car/page metadata)

Question ──► embed ──► Qdrant top-5 search ──► Qwen3-8B (Ollama) ──► answer + [Car, p.X] citations
```

## Stack

- **Embeddings:** `qwen3-embedding:0.6b` via Ollama (1024-dim)
- **Vector DB:** Qdrant (Docker)
- **LLM:** `qwen3:8b` via Ollama
- **PDF parsing:** PyMuPDF

## Setup

```bash
# 1. Models
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:8b

# 2. Vector database
docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant

# 3. Python deps
pip install -r requirements.txt
```

Put brochure PDFs in `brochures/`. The filename becomes the car name (e.g. `Kia EV4.pdf`).

## Run

```bash
python ingest.py   # once, loads brochures into Qdrant
python chat.py     # ask questions
```

Settings (models, chunk size, top-k) are in the config block at the top of each script.

## Example

```
You: What's the WLTP range of the Kia EV4?
Bot: ...
```

_(add your own example Q&As here)_

## Design notes

- **No fine-tuning (LoRA):** RAG supplies the facts, and the system prompt handles format, citations and "not in the brochure" refusals. Prompting was enough, so fine-tuning wasn't justified.
- **Grounded answers:** the model is told to answer only from retrieved context and cite the car and page.

## Possible next steps

- Hybrid search (BM25 + dense) for exact model/trim names
- Reranker (`bge-reranker-v2-m3`)
- Structured extraction of spec tables for numeric comparisons
- Gradio UI
