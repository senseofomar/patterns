import os, sys, shutil, subprocess
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List

from utils.semantic_utils import load_semantic_index, semantic_search
from utils.answer_generator import generate_answer

app = FastAPI(title="BookFriend API (V1 Complete)", version="1.0")

class State:
    index = None
    mapping = None
    chapter_limit = 999999

state = State()

class AskRequest(BaseModel):
    query: str

class ProgressRequest(BaseModel):
    chapter_limit: int

class SearchResponse(BaseModel):
    answer: str
    sources: List[str]

@app.on_event("startup")
def startup():
    reload_index()

def reload_index():
    try:
        state.index, state.mapping = load_semantic_index()
    except Exception:
        pass

@app.get("/")
def home():
    return {
        "status": "online",
        "current_chapter_limit": state.chapter_limit,
        "brain_loaded": state.index is not None
    }

@app.post("/set-progress")
def set_progress(req: ProgressRequest):
    state.chapter_limit = req.chapter_limit
    return {"message": f"Spoiler shield set to Chapter {req.chapter_limit}"}

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    tmp = f"uploaded_{file.filename}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)

    target = "lord_of_mysteries.pdf"
    if os.path.exists(target): os.remove(target)
    shutil.move(tmp, target)

    try:
        subprocess.run([sys.executable, "ingest.py"], check=True)
        subprocess.run([sys.executable, "build_index.py"], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Processing failed: {e}")

    reload_index()
    return {"message": "Book processed and indexed!"}

@app.post("/ask", response_model=SearchResponse)
def ask(req: AskRequest):
    if not state.index:
        raise HTTPException(503, "Index not loaded")

    results = semantic_search(req.query, state.index, state.mapping, top_k=50)
    safe = []

    for fname, chunk, _ in results:
        try:
            n = int("".join(filter(str.isdigit, fname)))
            if n <= state.chapter_limit:
                safe.append((fname, chunk))
        except ValueError:
            safe.append((fname, chunk))

    ctx = safe[:3]
    if not ctx:
        return {"answer": "Spoiler Shield Active.", "sources": []}

    try:
        ans = generate_answer(req.query, [c for _, c in ctx])
        return {"answer": ans, "sources": [f for f, _ in ctx]}
    except Exception as e:
        raise HTTPException(500, str(e))
