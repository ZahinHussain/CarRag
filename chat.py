"""
chat.py - Ask questions about the EV brochures.

    python chat.py

Type a question, press Enter. Type 'quit' to exit.
"""

# ================= CONFIG =================
QDRANT_URL = "http://localhost:6333"
COLLECTION = "ev_brochures"
EMBED_MODEL = "qwen3-embedding:0.6b"  # must match ingest.py
QUERY_INSTRUCTION = "Given a question about an electric car, retrieve brochure passages that answer it"
LLM_MODEL = "qwen3:8b"
TOP_K = 5                             # chunks retrieved per question
SHOW_SOURCES = True                   # print which chunks were used
# ==========================================

import ollama
from qdrant_client import QdrantClient

SYSTEM_PROMPT = """You are an assistant that answers questions about electric cars using ONLY the brochure excerpts provided.

Rules:
- Use only the information in the context. Do not use outside knowledge.
- Cite the source for each fact like this: [Kia EV4, p.12]
- If the answer is not in the context, say: "That's not in the brochures I have."
- When comparing cars, use a short table.
- Be concise."""


def retrieve(client, question):
    # Qwen3-Embedding: queries get an instruction prefix, documents don't
    query = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {question}"
    vec = ollama.embed(model=EMBED_MODEL, input=[query])["embeddings"][0]
    result = client.query_points(collection_name=COLLECTION, query=vec, limit=TOP_K)
    return result.points


def build_context(points):
    blocks = []
    for p in points:
        car, page, text = p.payload["car"], p.payload["page"], p.payload["text"]
        blocks.append(f"[{car}, p.{page}]\n{text}")
    return "\n\n---\n\n".join(blocks)


def answer(question, context):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    stream = ollama.chat(model=LLM_MODEL, messages=messages, stream=True, think=False)
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print()


def main():
    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(COLLECTION):
        raise SystemExit(f"Collection '{COLLECTION}' not found. Run ingest.py first.")

    print("EV Brochure Chat - type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break

        points = retrieve(client, question)
        if SHOW_SOURCES:
            used = ", ".join(f"{p.payload['car']} p.{p.payload['page']} ({p.score:.2f})" for p in points)
            print(f"  [retrieved: {used}]")

        print("Bot: ", end="")
        answer(question, build_context(points))
        print()


if __name__ == "__main__":
    main()
