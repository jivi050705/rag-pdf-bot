import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from rag_engine import process_pdf, get_qa_chain
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="RAG PDF Intelligence Bot")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore_store = {}
chain_store = {}
retriever_store = {}
chat_history_store = {}

class QuestionRequest(BaseModel):
    session_id: str
    question: str

@app.post("/upload-pdf/")
async def upload_pdf(session_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    save_path = f"uploaded_pdfs/{session_id}_{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    vectorstore = process_pdf(save_path)
    chain, retriever = get_qa_chain(vectorstore)

    vectorstore_store[session_id] = vectorstore
    chain_store[session_id] = chain
    retriever_store[session_id] = retriever
    chat_history_store[session_id] = []

    return {"message": f"PDF '{file.filename}' processed successfully!", "session_id": session_id}

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    if request.session_id not in chain_store:
        raise HTTPException(status_code=404, detail="No PDF uploaded for this session.")

    chain = chain_store[request.session_id]
    retriever = retriever_store[request.session_id]
    chat_history = chat_history_store[request.session_id]

    source_docs = retriever.invoke(request.question)
    source_pages = list(set([doc.metadata.get("page", 0) + 1 for doc in source_docs]))

    answer = chain.invoke({
        "question": request.question,
        "chat_history": chat_history
    })

    chat_history.append(HumanMessage(content=request.question))
    chat_history.append(AIMessage(content=answer))
    chat_history_store[request.session_id] = chat_history

    return {
        "answer": answer,
        "source_pages": source_pages
    }

@app.get("/")
def root():
    return {"message": "RAG PDF Bot is running!"}

@app.get("/app")
def serve_frontend():
    return FileResponse("static/index.html")