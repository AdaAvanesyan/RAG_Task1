# rag_core.py
import io
import re
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Global in-memory state
GLOBAL_MODEL = None
FAISS_INDEX = None
CHUNKS = []
METAS = []


# -----------------------
# PDF Extraction
# -----------------------

def extract_text_from_pdf_bytes(pdf_bytes: bytes):
    """Extract pages from PDF bytes (not file path)."""
    pages = []
    file_like = io.BytesIO(pdf_bytes)
    reader = PyPDF2.PdfReader(file_like)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i + 1, text))
    return pages

# -----------------------
# Chunking
# -----------------------

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


# -----------------------
# Ingest PDF
# -----------------------

async def ingest_pdf_bytes(pdf_bytes: bytes):
    """Reads PDF bytes, chunks them, embeds them, builds FAISS index."""

    global GLOBAL_MODEL, FAISS_INDEX, CHUNKS, METAS

    pages = extract_text_from_pdf_bytes(pdf_bytes)
    if not pages:
        return 0, 0

    CHUNKS = []
    METAS = []

    chunk_id = 0
    for page_num, text in pages:
        for chunk in chunk_text(text):
            CHUNKS.append(chunk)
            METAS.append({"id": chunk_id, "page": page_num})
            chunk_id += 1

    # Build embeddings + FAISS
    GLOBAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = GLOBAL_MODEL.encode(CHUNKS, convert_to_numpy=True)

    dim = embeddings.shape[1]
    FAISS_INDEX = faiss.IndexFlatL2(dim)
    FAISS_INDEX.add(embeddings.astype("float32"))

    return len(pages), len(CHUNKS)


# -----------------------
# Retrieval
# -----------------------

async def retrieve_chunks(question: str, top_k=5):
    """Return top-k retrieved chunks and metadata."""

    global GLOBAL_MODEL, FAISS_INDEX, CHUNKS, METAS

    if FAISS_INDEX is None:
        return []

    q_emb = GLOBAL_MODEL.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = FAISS_INDEX.search(q_emb, top_k)

    hits = []
    for rank, idx in enumerate(indices[0]):
        page = METAS[idx]["page"]
        text = CHUNKS[idx]
        hits.append((page, text))

    return hits


def is_index_ready():
    """Check if ingestion has happened."""
    return FAISS_INDEX is not None














# # rag_core.py

# import io
# import re
# from typing import List, Dict, Optional

# import faiss
# import numpy as np
# import PyPDF2
# from sentence_transformers import SentenceTransformer

# # Global state for current document
# sentence_model: Optional[SentenceTransformer] = None
# faiss_index: Optional[faiss.IndexFlatL2] = None
# chunks: List[str] = []
# metas: List[Dict] = []  # each: {"id": int, "page": int}


# def get_sentence_model() -> SentenceTransformer:
#     """
#     Lazily load and return the sentence-transformer model used for embeddings.

#     Returns:
#         SentenceTransformer: Loaded embedding model (cached globally).
#     """
#     global sentence_model
#     if sentence_model is None:
#         sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
#     return sentence_model


# def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[tuple]:
#     """
#     Extract text from a PDF provided as raw bytes.

#     The function reads each page of the PDF and returns a list of (page_number, text)
#     tuples, skipping empty pages.

#     Args:
#         pdf_bytes (bytes): Binary content of the PDF file.

#     Returns:
#         List[tuple]: List of (page_number, cleaned_text) for each non-empty page.
#     """
#     pages = []
#     reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
#     for i, page in enumerate(reader.pages):
#         text = page.extract_text() or ""
#         text = re.sub(r"\s+", " ", text).strip()
#         if text:
#             pages.append((i + 1, text))
#     return pages


# def chunk_text(text: str, chunk_size: int = 200) -> List[str]:
#     """
#     Split a large text into smaller chunks of approximately `chunk_size` words.

#     This improves embedding quality and retrieval granularity.

#     Args:
#         text (str): The text to split.
#         chunk_size (int, optional): Maximum number of words per chunk. Defaults to 200.

#     Returns:
#         List[str]: List of text chunks.
#     """
#     words = text.split()
#     return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# def build_index_for_chunks(all_chunks: List[str]) -> faiss.IndexFlatL2:
#     """
#     Build a FAISS index for the given list of text chunks.

#     Encodes all chunks into vectors using the sentence-transformer model and
#     stores them in a FAISS IndexFlatL2.

#     Args:
#         all_chunks (List[str]): List of text chunks to index.

#     Returns:
#         faiss.IndexFlatL2: FAISS index containing all chunk embeddings.
#     """
#     model = get_sentence_model()
#     embeddings = model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=False)
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings.astype("float32"))
#     return index


# def ingest_pdf_bytes(pdf_bytes: bytes, chunk_size: int = 200) -> Dict[str, int]:
#     """
#     Ingest a PDF (as bytes), extract its text, chunk it, embed it, and build a FAISS index.

#     This function populates the global state: `chunks`, `metas`, and `faiss_index`.

#     Args:
#         pdf_bytes (bytes): Binary content of the PDF file.
#         chunk_size (int, optional): Number of words per chunk. Defaults to 200.

#     Returns:
#         Dict[str, int]: Summary information with number of pages and chunks created.
#     """
#     global chunks, metas, faiss_index

#     pages = extract_text_from_pdf_bytes(pdf_bytes)
#     if not pages:
#         return {"pages": 0, "chunks": 0}

#     all_chunks: List[str] = []
#     all_metas: List[Dict] = []
#     chunk_id = 0
#     for page_num, text in pages:
#         for ch in chunk_text(text, chunk_size=chunk_size):
#             all_chunks.append(ch)
#             all_metas.append({"id": chunk_id, "page": page_num})
#             chunk_id += 1

#     index = build_index_for_chunks(all_chunks)

#     chunks = all_chunks
#     metas = all_metas
#     faiss_index = index

#     return {"pages": len(pages), "chunks": len(chunks)}


# def retrieve(query: str, top_k: int = 5) -> List[Dict]:
#     """
#     Retrieve the top-k most relevant chunks for a given natural language query.

#     Uses the same sentence-transformer model to embed the query and performs
#     a nearest-neighbor search over the FAISS index.

#     Args:
#         query (str): User's question in natural language.
#         top_k (int, optional): Number of chunks to return. Defaults to 5.

#     Returns:
#         List[Dict]: List of dictionaries, each containing:
#             - rank (int): 1-based rank of the result
#             - score (float): distance score from FAISS
#             - page (int): page number in the original PDF
#             - text (str): the retrieved chunk text

#     Raises:
#         RuntimeError: If no document has been ingested yet.
#     """
#     global faiss_index, chunks, metas

#     if faiss_index is None or not chunks:
#         raise RuntimeError("No document ingested yet. Call /ingest first.")

#     model = get_sentence_model()
#     q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
#     D, I = faiss_index.search(q_emb, top_k)
#     results: List[Dict] = []
#     for rank, idx in enumerate(I[0]):
#         meta = metas[idx]
#         results.append(
#             {
#                 "rank": rank + 1,
#                 "score": float(D[0][rank]),
#                 "page": meta["page"],
#                 "text": chunks[idx],
#             }
#         )
#     return results