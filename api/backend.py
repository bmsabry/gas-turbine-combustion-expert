import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import re
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

from api.admin import setup_admin_routes, load_settings

app = FastAPI(title="Gas Turbine Combustion Expert API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

PROJECT_DIR = Path(__file__).parent.parent
STATIC_DIR = Path("/app/static")

static_dir = STATIC_DIR if STATIC_DIR.exists() else PROJECT_DIR / "static"
if static_dir.exists() and (static_dir / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")


class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict] = []
    conflicts: List[str] = []
    single_study_notes: List[str] = []


_chunks = []
_tfidf_matrix = None
_vectorizer = None
_entities = []
_relationships = []
_contradictions = []
_data_loaded = False


def load_all_data():
    global _chunks, _tfidf_matrix, _vectorizer, _entities, _relationships, _contradictions, _data_loaded
    if _data_loaded:
        return
    print("Loading data...")
    chunks_dir = PROJECT_DIR / "chunks"
    all_chunks = []
    if chunks_dir.exists():
        for chunk_file in sorted(chunks_dir.glob("*.json")):
            try:
                with open(chunk_file) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get("text"):
                            all_chunks.append(item)
                elif isinstance(data, dict) and data.get("text"):
                    all_chunks.append(data)
            except Exception:
                pass
    _chunks = all_chunks
    print(f"Loaded {len(_chunks)} chunks")
    if _chunks:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts = [c.get("text", "") for c in _chunks]
            _vectorizer = TfidfVectorizer(max_features=50000, stop_words="english", ngram_range=(1, 2))
            _tfidf_matrix = _vectorizer.fit_transform(texts)
            print(f"TF-IDF index built: {_tfidf_matrix.shape}")
        except Exception as e:
            print(f"TF-IDF failed: {e}")
    kg_dir = PROJECT_DIR / "knowledge_graph"
    try:
        with open(kg_dir / "entities.json") as f:
            _entities = json.load(f)
    except Exception:
        pass
    try:
        with open(kg_dir / "relationships.json") as f:
            _relationships = json.load(f)
    except Exception:
        pass
    try:
        with open(kg_dir / "contradictions.json") as f:
            _contradictions = json.load(f)
    except Exception:
        pass
    print(f"KG: {len(_entities)} entities, {len(_contradictions)} conflicts")
    _data_loaded = True


def search_chunks(query: str, top_k: int = 15) -> List[Dict]:
    if not _data_loaded:
        load_all_data()
    if _vectorizer is not None and _tfidf_matrix is not None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            qv = _vectorizer.transform([query])
            scores = cosine_similarity(qv, _tfidf_matrix)[0]
            top_idx = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_idx:
                if scores[idx] > 0:
                    c = _chunks[idx].copy()
                    c["score"] = float(scores[idx])
                    results.append(c)
            return results
        except Exception as e:
            print(f"TF-IDF error: {e}")
    keywords = [w.lower() for w in re.split(r"\W+", query) if len(w) > 3]
    scored = []
    for chunk in _chunks:
        text = chunk.get("text", "").lower()
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            c = chunk.copy()
            c["score"] = float(score)
            scored.append(c)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def find_conflicts(query: str) -> List[str]:
    query_lower = query.lower()
    results = []
    for conflict in _contradictions[:100]:
        entity = conflict.get("entity", "").lower()
        if entity and entity in query_lower:
            pa = conflict.get("paper_a", "Study A")
            pb = conflict.get("paper_b", "Study B")
            fa = conflict.get("paper_a_finding", "makes a claim")
            fb = conflict.get("paper_b_finding", "contradicts")
            results.append(
                f"CONFLICTING EVIDENCE on '{entity}': {pa} states: {fa}. However, {pb} states: {fb}."
            )
    return results[:3]


@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_all_data)


@app.get("/")
async def root():
    for p in [STATIC_DIR / "index.html", PROJECT_DIR / "static" / "index.html"]:
        if p.exists():
            return FileResponse(str(p), media_type="text/html")
    return {"name": "Gas Turbine Combustion Expert API", "version": "1.0.0", "status": "running"}


@app.get("/api")
async def api_info():
    return {"name": "Gas Turbine Combustion Expert API", "version": "1.0.0", "status": "running"}


@app.get("/api/health")
async def health():
    if not _data_loaded:
        load_all_data()
    return {
        "status": "healthy",
        "papers": len(set(c.get("paper_id", "") for c in _chunks)),
        "chunks": len(_chunks),
        "entities": len(_entities),
        "relationships": len(_relationships),
        "conflicts": len(_contradictions),
        "tfidf_loaded": _tfidf_matrix is not None
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not _data_loaded:
        load_all_data()

    retrieved = search_chunks(request.message, top_k=15)
    if not retrieved:
        return ChatResponse(
            response="No relevant information found in the research papers. Please rephrase your question.",
            sources=[], conflicts=[], single_study_notes=[]
        )

    sources = []
    seen = set()
    for chunk in retrieved[:10]:
        title = chunk.get("title", chunk.get("paper_id", "Unknown"))
        if title not in seen:
            seen.add(title)
            sources.append({
                "title": title,
                "year": chunk.get("year", ""),
                "credibility_score": chunk.get("credibility_score", 3),
                "section": chunk.get("section", ""),
                "relevance_score": round(chunk.get("score", 0), 3)
            })

    conflicts = find_conflicts(request.message)

    context_parts = []
    for i, chunk in enumerate(retrieved[:10]):
        title = chunk.get("title", "Unknown")
        year = chunk.get("year", "")
        text = chunk.get("text", "")
        section = chunk.get("section", "")
        context_parts.append(f"[{i+1}] {title} ({year}) - {section}:\n{text}\n")
    context = "\n".join(context_parts)

    settings = load_settings()
    api_key = settings.get("llm_api_key", "").strip()
    api_url = settings.get("llm_api_url", "").strip().rstrip("/")
    model = settings.get("llm_model", "google/gemini-2.0-flash-001")

    if not api_key:
        preview_parts = []
        for i, c in enumerate(retrieved[:5]):
            t = c.get("title", "Unknown")
            y = c.get("year", "")
            txt = c.get("text", "")[:400]
            preview_parts.append(f"**[{i+1}] {t} ({y})**\n{txt}...")
        preview = "\n\n".join(preview_parts)
        return ChatResponse(
            response=(
                "## LLM Not Configured\n\n"
                "Please configure an LLM API key in the Admin Panel:\n"
                "1. Go to `/admin`\n"
                "2. Login: admin / admin123\n"
                "3. Enter your API key\n\n---\n\n"
                "**Top retrieved papers:**\n\n" + preview
            ),
            sources=sources, conflicts=conflicts, single_study_notes=[]
        )

    conflict_ctx = ""
    if conflicts:
        conflict_ctx = "\n\nLITERATURE CONFLICTS:\n" + "\n".join(conflicts)

    system_prompt = (
        "You are a gas turbine combustion expert AI. Answer ONLY from the provided research excerpts.\n\n"
        "Rules:\n"
        "1. Never use outside knowledge - if not in context, say so explicitly.\n"
        "2. Cite agreeing papers together: (Author et al., Year)\n"
        "3. Flag disagreements: CONFLICTING EVIDENCE: [Paper A] states X. However [Paper B] states Y.\n"
        "4. Single-study findings: NOTE: Single study - not independently corroborated.\n"
        "5. For design questions: answer component by component.\n"
        "6. Do NOT include any citations, reference lists, or source lists. Just answer directly.\n"
        "7. Confidence levels: Literature strongly supports / Evidence suggests / Limited evidence.\n"
        "8. Be thorough and technical for combustion engineers."
    )

    user_msg = (
        f"Research Paper Context:\n{context}{conflict_ctx}\n\n"
        f"Question: {request.message}\n\n"
        "Provide a comprehensive, technically accurate answer based solely on the provided research papers."
    )

    if "anthropic" in api_url.lower():
        endpoint = f"{api_url}/v1/messages"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": model,
            "max_tokens": 2000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_msg}]
        }
    elif "openrouter" in api_url.lower():
        endpoint = f"{api_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://gas-turbine-combustion-expert.onrender.com",
            "X-Title": "Gas Turbine Combustion Expert"
        }
        payload = {
            "model": model,
            "max_tokens": 2000,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ]
        }
    else:
        endpoint = f"{api_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "max_tokens": 2000,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ]
        }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if "anthropic" in api_url.lower():
            llm_text = data["content"][0]["text"]
        else:
            llm_text = data["choices"][0]["message"]["content"]

        return ChatResponse(
            response=llm_text,
            sources=sources,
            conflicts=conflicts,
            single_study_notes=[]
        )

    except httpx.HTTPStatusError as e:
        err = e.response.text[:500] if e.response else str(e)
        return ChatResponse(
            response=f"## API Error\n\nStatus: {e.response.status_code}\n\n{err}\n\nCheck your API key in Admin Panel.",
            sources=sources, conflicts=conflicts, single_study_notes=[]
        )
    except Exception as e:
        return ChatResponse(
            response=f"## Error\n\n{str(e)}\n\nCheck API settings in Admin Panel.",
            sources=sources, conflicts=conflicts, single_study_notes=[]
        )


setup_admin_routes(app)


@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    for p in [STATIC_DIR / "index.html", PROJECT_DIR / "static" / "index.html"]:
        if p.exists():
            return FileResponse(str(p), media_type="text/html")
    raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
