# server.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_core import ingest_pdf_bytes, retrieve_chunks, is_index_ready
from llm import generate_answer, stream_answer


app = FastAPI(
    title="RAG PDF QA API",
    description="Upload a PDF, build a vector index, and ask questions using retrieval + OpenAI",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/")
async def root():
    return {"message": "RAG server is running"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    pages, chunks = await ingest_pdf_bytes(pdf_bytes)

    if pages == 0:
        raise HTTPException(status_code=400, detail="No text extracted from PDF")

    return {"pages": pages, "chunks": chunks}


@app.post("/ask")
async def ask(req: AskRequest):
    if not is_index_ready():
        raise HTTPException(status_code=400, detail="No PDF ingested yet. Call /ingest first.")

    hits = await retrieve_chunks(req.question, req.top_k)
    if not hits:
        return {
            "answer": "No relevant text found in the document.",
            "sources": [],
        }

    answer = generate_answer(req.question, hits)

    return {
        "answer": answer,
        "sources": [{"page": p, "text": t} for p, t in hits],
    }


@app.post("/ask_stream")
async def ask_stream(req: AskRequest):
    if not is_index_ready():
        raise HTTPException(status_code=400, detail="No PDF ingested yet. Call /ingest first.")

    hits = await retrieve_chunks(req.question, req.top_k)
    if not hits:
        def empty():
            yield "No relevant text found in the document."
        return StreamingResponse(empty(), media_type="text/plain")

    def token_generator():
        for token in stream_answer(req.question, hits):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)