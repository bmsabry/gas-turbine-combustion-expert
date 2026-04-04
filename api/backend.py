"""
Gas Turbine Combustion Expert - FastAPI Backend
Advanced RAG system with Knowledge Graph
"""
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import asyncio

# Import admin module
from api.admin import setup_admin_routes, load_settings

app = FastAPI(title="Gas Turbine Combustion Expert API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Project directory
PROJECT_DIR = Path(__file__).parent.parent
STATIC_DIR = Path("/app/static")  # Docker container path

# Mount static files for frontend
static_dir = PROJECT_DIR / "static"
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")

@app.get("/api")
async def api_info():
    return {
        "name": "Gas Turbine Combustion Expert API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/")
async def root():
    # Try Docker path first, then local path
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        index_file = PROJECT_DIR / "static" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    return {"message": "Gas Turbine Combustion Expert API", "docs": "/docs"}


class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict] = []
    conflicts: List[str] = []
    single_study_notes: List[str] = []

# ─── Data Loading ────────────────────────────────────────────────────────────

def load_embeddings():
    embeddings_file = PROJECT_DIR / "embeddings" / "embeddings.json"
    if embeddings_file.exists():
        with open(embeddings_file) as f:
            return json.load(f)
    return []

def load_chunks():
    chunks = []
    chunks_dir = PROJECT_DIR / "chunks"
    if chunks_dir.exists():
        for chunk_file in chunks_dir.glob("*_chunks.json"):
            try:
                with open(chunk_file) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        chunks.extend(data)
                    elif isinstance(data, dict):
                        chunks.append(data)
            except Exception as e:
                print(f"Error loading {chunk_file}: {e}")
    return chunks

def load_knowledge_graph():
    entities_file = PROJECT_DIR / "knowledge_graph" / "entities.json"
    relationships_file = PROJECT_DIR / "knowledge_graph" / "relationships.json"
    contradictions_file = PROJECT_DIR / "knowledge_graph" / "contradictions.json"

    entities, relationships, contradictions = [], [], []
    try:
        if entities_file.exists():
            with open(entities_file) as f:
                entities = json.load(f)
    except Exception as e:
        print(f"Error loading entities: {e}")
    try:
        if relationships_file.exists():
            with open(relationships_file) as f:
                relationships = json.load(f)
    except Exception as e:
        print(f"Error loading relationships: {e}")
    try:
        if contradictions_file.exists():
            with open(contradictions_file) as f:
                contradictions = json.load(f)
    except Exception as e:
        print(f"Error loading contradictions: {e}")

    return entities, relationships, contradictions

# ─── Load FAISS index ─────────────────────────────────────────────────────────

def load_faiss_index():
    index_file = PROJECT_DIR / "embeddings" / "faiss_index.bin"
    ids_file = PROJECT_DIR / "embeddings" / "faiss_index.ids"
    if index_file.exists():
        index = faiss.read_index(str(index_file))
        if ids_file.exists():
            with open(ids_file) as f:
                ids = json.load(f)
        else:
            ids = list(range(index.ntotal))
        return index, ids
    return None, []

# ─── Startup ─────────────────────────────────────────────────────────────────

print("Loading embeddings...")
embeddings_data = load_embeddings()
print(f"Loaded {len(embeddings_data)} embeddings")

print("Loading chunks...")
all_chunks = load_chunks()
print(f"Loaded {len(all_chunks)} chunks")

print("Loading knowledge graph...")
entities, relationships, conflicts = load_knowledge_graph()
print(f"Loaded {len(entities)} entities, {len(relationships)} relationships, {len(conflicts)} conflicts")

print("Loading FAISS index...")
faiss_index, faiss_ids = load_faiss_index()
if faiss_index:
    print(f"FAISS index loaded: {faiss_index.ntotal} vectors")
else:
    print("FAISS index NOT found — will use keyword fallback")

# Build chunk lookup by index position
chunk_by_idx = {i: c for i, c in enumerate(all_chunks)}

# ─── Semantic search using FAISS ──────────────────────────────────────────────

def semantic_search(query: str, top_k: int = 10) -> List[Dict]:
    """Search chunks using FAISS semantic similarity"""
    if faiss_index is None or not embeddings_data:
        return keyword_fallback(query, top_k)

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vec = model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        distances, indices = faiss_index.search(query_vec, min(top_k, faiss_index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            # Map FAISS index to chunk
            if idx < len(all_chunks):
                chunk = all_chunks[idx].copy()
                chunk["_score"] = float(dist)
                results.append(chunk)
        return results
    except Exception as e:
        print(f"FAISS search error: {e} — falling back to keyword")
        return keyword_fallback(query, top_k)


def keyword_fallback(query: str, top_k: int = 10) -> List[Dict]:
    """Keyword-based fallback retrieval"""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    domain_terms = [
        "nox", "emission", "swirl", "pressure", "flame", "combustion",
        "stability", "flashback", "residence", "time", "temperature",
        "fuel", "air", "equivalence", "ratio", "lean", "rich", "premix",
        "diffusion", "turbulence", "blowout", "ignition", "liner",
        "injector", "dilution", "cooling", "pattern", "factor"
    ]

    scored = []
    for chunk in all_chunks:
        text = chunk.get("text", "").lower()
        tags = [t.lower() for t in chunk.get("topic_tags", [])]
        score = 0
        for word in query_words:
            if word in text:
                score += 2
            if word in tags:
                score += 3
        for term in domain_terms:
            if term in query_lower and term in text:
                score += 1
        if score > 0:
            c = chunk.copy()
            c["_score"] = score
            scored.append(c)

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:top_k]


# ─── Health & Stats endpoints ─────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "papers": 317,
        "chunks": len(all_chunks),
        "entities": len(entities),
        "relationships": len(relationships),
        "conflicts": len(conflicts),
        "faiss_loaded": faiss_index is not None
    }

@app.get("/api/stats")
async def get_stats():
    return {
        "papers_processed": 317,
        "chunks": len(all_chunks),
        "embeddings": len(embeddings_data),
        "entities": len(entities),
        "relationships": len(relationships),
        "conflicts": len(conflicts)
    }


# ─── Main Chat Endpoint ───────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with FAISS RAG retrieval and LLM synthesis"""

    # Get settings
    settings = load_settings()
    api_key = settings.get("llm_api_key", "").strip()
    api_url = settings.get("llm_api_url", "").rstrip("/")
    model = settings.get("llm_model", "google/gemini-2.0-flash-001")

    # ── Semantic retrieval ────────────────────────────────────────────────────
    relevant_chunks = semantic_search(request.message, top_k=10)

    # Deduplicate by text
    seen_texts: set = set()
    unique_chunks = []
    for c in relevant_chunks:
        txt = c.get("text", "")[:100]
        if txt not in seen_texts:
            seen_texts.add(txt)
            unique_chunks.append(c)
    relevant_chunks = unique_chunks[:8]

    # ── Check for conflicts ───────────────────────────────────────────────────
    query_lower = request.message.lower()
    relevant_conflicts = []
    for conflict in conflicts:
        e1 = conflict.get("entity_1", "").lower()
        e2 = conflict.get("entity_2", "").lower()
        if e1 in query_lower or e2 in query_lower:
            pa = conflict.get("paper_a", "Unknown")
            pb = conflict.get("paper_b", "Unknown")
            fa = conflict.get("paper_a_finding", "states a different conclusion")
            fb = conflict.get("paper_b_finding", "states a different conclusion")
            notes = conflict.get("resolution_notes", "Requires expert review")
            relevant_conflicts.append(
                f"⚠️ CONFLICTING EVIDENCE: [{pa}] states: {fa}. "
                f"However, [{pb}] states: {fb}. "
                f"Expert note: {notes}"
            )

    # ── Build sources list ────────────────────────────────────────────────────
    sources = []
    seen_titles: set = set()
    for chunk in relevant_chunks:
        title = chunk.get("title", "Unknown")
        if title not in seen_titles:
            sources.append({
                "title": title,
                "year": chunk.get("year", "Unknown"),
                "authors": chunk.get("authors", []),
                "chunk_id": chunk.get("chunk_id", "")
            })
            seen_titles.add(title)

    # ── Build context for LLM ─────────────────────────────────────────────────
    context_parts = []
    for i, chunk in enumerate(relevant_chunks):
        title = chunk.get("title", "Unknown")
        year = chunk.get("year", "Unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[Paper {i+1}] {title} ({year}):\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    # ── No API key: return clear message ──────────────────────────────────────
    if not api_key:
        return ChatResponse(
            response=(
                "⚠️ **LLM Not Configured**\n\n"
                "Please configure an LLM in the admin panel (/admin).\n"
                "Login: admin / admin123"
            ),
            sources=sources,
            conflicts=relevant_conflicts[:3],
            single_study_notes=[]
        )

    # ── Build system prompt ───────────────────────────────────────────────────
    system_prompt = (
        "You are a gas turbine combustion expert AI assistant with access to 317 scientific research papers.\n\n"
        "RULES:\n"
        "1. Answer ONLY based on the provided research paper excerpts below.\n"
        "2. If the context does not contain enough information, say so explicitly - never hallucinate.\n"
        "3. When multiple papers agree, synthesize them and cite all: (Smith et al., 2021; Jones et al., 2019).\n"
        "4. When papers DISAGREE, explicitly flag it with: ⚠️ CONFLICTING EVIDENCE\n"
        "5. When a finding comes from only ONE paper, flag it with: 📌 NOTE: Single study finding\n"
        "6. For design questions, answer component by component, citing the relevant paper for each point.\n"
        "7. End with a ## Sources Used section listing every paper cited.\n"
        "8. Use confidence language: The literature strongly supports... / Evidence suggests... / Limited evidence indicates..."
    )

    user_message = (
        f"Based on the following research paper excerpts, answer the question.\n\n"
        f"RESEARCH PAPER EXCERPTS:\n{context}\n\n"
        f"QUESTION: {request.message}\n\n"
        f"Provide a detailed, expert-level answer with proper citations."
    )

    # ── Call LLM via OpenAI-compatible API (works with OpenRouter, OpenAI, etc.) ──
    llm_response = None
    error_message = None

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://gas-turbine-combustion-expert.onrender.com",
                "X-Title": "Gas Turbine Combustion Expert"
            }

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 2048,
                "temperature": 0.1
            }

            # OpenAI-compatible endpoint (works for OpenRouter, OpenAI, etc.)
            endpoint = f"{api_url}/chat/completions"
            print(f"Calling LLM: {endpoint} with model {model}")

            response = await client.post(endpoint, headers=headers, json=payload)
            print(f"LLM response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                llm_response = data["choices"][0]["message"]["content"]
            else:
                error_message = f"LLM API error {response.status_code}: {response.text[:500]}"
                print(error_message)

    except Exception as e:
        error_message = f"LLM call failed: {str(e)}"
        print(error_message)

    # ── Return LLM response ───────────────────────────────────────────────────
    if llm_response:
        return ChatResponse(
            response=llm_response,
            sources=sources,
            conflicts=relevant_conflicts[:5],
            single_study_notes=[]
        )

    # ── Fallback: show error + raw context ───────────────────────────────────
    fallback = f"⚠️ **LLM Error**: {error_message}\n\n" if error_message else ""
    fallback += "**Retrieved Context from Papers (LLM unavailable):**\n\n"
    for chunk in relevant_chunks[:3]:
        title = chunk.get("title", "Unknown")
        year = chunk.get("year", "Unknown")
        text = chunk.get("text", "")[:400]
        fallback += f"**{title}** ({year}):\n{text}...\n\n"

    return ChatResponse(
        response=fallback,
        sources=sources,
        conflicts=relevant_conflicts[:3],
        single_study_notes=[]
    )


# Setup admin routes
app = setup_admin_routes(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
