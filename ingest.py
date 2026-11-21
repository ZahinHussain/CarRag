"""
rag_app.py - Chat with EV brochures (RAG).

Structure follows freeCodeCamp's "RAG Explained Simply with a Real Project",
but runs fully locally with Ollama instead of the Gemini API:
    Load PDFs -> chunk -> embed (Qwen3-Embedding) -> store (ChromaDB)
    -> retrieve -> prompt -> LLM (Qwen3) -> answer

Just run:  python rag_app.py
"""

# ================= CONFIG =================
BROCHURE_DIR = "brochures"             # folder of PDFs, named by car e.g. "Kia EV4.pdf"
DB_DIR = "./chroma_db"                 # local vector database folder
REBUILD_DB = False                     # True = wipe and re-embed (do this after adding brochures)

EMBED_MODEL = "qwen3-embedding:0.6b"   # Ollama embedding model
LLM_MODEL = "qwen3:8b"                 # Ollama chat model (use qwen3:4b on a 16GB Mac)

CHUNK_SIZE = 500                       # characters per chunk (as in the article)
CHUNK_OVERLAP = 50                     # characters of overlap (as in the article)
TOP_K = 4                              # chunks retrieved per question (article uses 2)

USE_GRADIO = False                     # True = browser chat UI instead of terminal
# ==========================================

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# ---------- Embeddings ----------
class QwenEmbeddings(OllamaEmbeddings):
    """Qwen3-Embedding works best with an instruction on queries (not on documents)."""

    def embed_query(self, text: str) -> list[float]:
        instruction = "Given a question about an electric car, retrieve brochure passages that answer it"
        return super().embed_query(f"Instruct: {instruction}\nQuery: {text}")


embeddings = QwenEmbeddings(model=EMBED_MODEL)


# ---------- Steps 1-4: Load, chunk, embed, store ----------
def build_vector_db():
    # Step 1: load every PDF in the folder, one Document per page,
    # tagged with the car name (from the filename) and page number
    print(f"Loading PDFs from '{BROCHURE_DIR}/'...")
    documents = []
    for pdf in sorted(Path(BROCHURE_DIR).glob("*.pdf")):
        for page_num, page in enumerate(PdfReader(pdf).pages):
            text = (page.extract_text() or "").strip()
            if text:  # skip image-only pages
                documents.append(Document(
                    page_content=text,
                    metadata={"car": pdf.stem, "page": page_num, "source": pdf.name},
                ))
    if not documents:
        raise SystemExit(f"No PDFs (with text) found in '{BROCHURE_DIR}/'")
    print(f"Loaded {len(documents)} pages. First page preview:\n{documents[0].page_content[:300]}\n")

    # Step 2: chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # Steps 3-4: embed and store in ChromaDB
    print("Embedding and storing (first run takes a minute)...")
    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)


def load_or_build_db():
    if REBUILD_DB and Path(DB_DIR).exists():
        shutil.rmtree(DB_DIR)
    if Path(DB_DIR).exists():
        print(f"Using existing database in '{DB_DIR}'.")
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return build_vector_db()


# ---------- Step 5: Retriever + prompt ----------
TEMPLATE = """You are an assistant answering questions about electric cars.
Use ONLY the brochure excerpts below to answer the question.
If you don't know the answer, just say that you don't know. Don't make anything up.
Cite the source for facts like this: [Kia EV4, p.12]
Keep the answer to three sentences maximum, unless comparing cars (then use a short table).

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(TEMPLATE)


def format_docs(docs):
    # Label each chunk with car + page so the model can cite it
    return "\n\n".join(
        f"[{d.metadata['car']}, p.{d.metadata.get('page', 0) + 1}]\n{d.page_content}" for d in docs
    )


# ---------- Step 6: LLM + chain ----------
def build_chain(vector_db):
    retriever = vector_db.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOllama(model=LLM_MODEL, temperature=0, reasoning=False)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# ---------- Steps 7-8: Ask questions ----------
def terminal_loop(rag_chain):
    print("\nEV Brochure Chat - type 'exit' or 'quit' to stop.")
    while True:
        question = input("\nYour Question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        print("Answer: ", end="")
        for piece in rag_chain.stream(question):
            print(piece, end="", flush=True)
        print()


def gradio_ui(rag_chain):
    import gradio as gr

    def respond(message, history):
        answer = ""
        for piece in rag_chain.stream(message):
            answer += piece
            yield answer

    gr.ChatInterface(respond, title="EV Brochure Chat").launch()


if __name__ == "__main__":
    db = load_or_build_db()
    chain = build_chain(db)
    if USE_GRADIO:
        gradio_ui(chain)
    else:
        terminal_loop(chain)