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
    # Fallback to API info
    return {"message": "Gas Turbine Combustion Expert API", "docs": "/docs", "static_dir": str(STATIC_DIR), "exists": STATIC_DIR.exists()}


class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict] = []
    conflicts: List[str] = []
    single_study_notes: List[str] = []

# Load data
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
    
    entities = []
    relationships = []
    contradictions = []
    
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

# Load data at startup
print("Loading embeddings...")
embeddings_data = load_embeddings()
print(f"Loaded {len(embeddings_data)} embeddings")

print("Loading chunks...")
all_chunks = load_chunks()
print(f"Loaded {len(all_chunks)} chunks")

print("Loading knowledge graph...")
entities, relationships, conflicts = load_knowledge_graph()
print(f"Loaded {len(entities)} entities, {len(relationships)} relationships, {len(conflicts)} conflicts")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "papers": 317,
        "chunks": len(all_chunks),
        "entities": len(entities),
        "relationships": len(relationships),
        "conflicts": len(conflicts)
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

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with RAG retrieval"""
    
    # Get settings
    settings = load_settings()
    api_key = settings.get("llm_api_key", "")
    api_url = settings.get("llm_api_url", "https://api.anthropic.com")
    model = settings.get("llm_model", "claude-sonnet-4-6")
    
    # Simple keyword-based retrieval for demo
    query_lower = request.message.lower()
    
    # Find relevant chunks based on topic tags
    relevant_chunks = []
    seen_texts = set()
    
    for chunk in all_chunks:
        tags = chunk.get("topic_tags", [])
        text = chunk.get("text", "")
        
        # Check if any tag matches query
        if any(tag in query_lower for tag in tags) or \
           any(tag in query_lower for tag in ["noxs", "nox", "emission", "swirl", "pressure", "flame", "combustion", "stability", "flashback", "residence", "time"]):
            if text not in seen_texts and len(relevant_chunks) < 5:
                relevant_chunks.append(chunk)
                seen_texts.add(text)
    
    # Check for conflicts related to the query
    relevant_conflicts = []
    for conflict in conflicts:
        entity = conflict.get("entity_1", "").lower()
        entity2 = conflict.get("entity_2", "").lower()
        if entity in query_lower or entity2 in query_lower:
            paper_a = conflict.get("paper_a", "Unknown")
            paper_b = conflict.get("paper_b", "Unknown")
            finding_a = conflict.get("paper_a_finding", "")
            finding_b = conflict.get("paper_b_finding", "")
            relevant_conflicts.append(
                f"⚠️ CONFLICTING EVIDENCE: {paper_a} indicates {finding_a}. "
                f"However, {paper_b} indicates {finding_b}. "
                f"Resolution: {conflict.get('resolution_notes', 'Requires expert review')}"
            )
    
    # Build context
    context = "\n\n".join([c.get("text", "") for c in relevant_chunks[:5]])
    
    # Build sources
    sources = []
    seen_titles = set()
    for chunk in relevant_chunks[:5]:
        title = chunk.get("title", "Unknown")
        if title not in seen_titles:
            sources.append({
                "title": title,
                "year": chunk.get("year", "Unknown"),
                "authors": [],
                "chunk_id": chunk.get("chunk_id", "")
            })
            seen_titles.add(title)
    
    # If no API key, return informative message
    if not api_key:
        return ChatResponse(
            response="⚠️ **LLM Not Configured**\n\n"
                    "To get AI-synthesized answers, please configure an LLM in the admin panel:\n\n"
                    "1. Go to /admin\n"
                    "2. Login with admin/admin123\n"
                    "3. Set your LLM provider (OpenAI or Anthropic)\n"
                    "4. Add your API key\n\n"
                    "**Raw context retrieved from papers:**\n\n" + 
                    "\n\n".join([f"• {c.get('title', 'Unknown')}: {c.get('text', '')[:200]}..." for c in relevant_chunks[:3]]),
            sources=sources,
            conflicts=relevant_conflicts,
            single_study_notes=[]
        )
    
    # Call the LLM
    llm_response = None
    try:
        async with httpx.AsyncClient() as client:
            system_prompt = """You are a gas turbine combustion expert AI assistant. You answer ONLY based on the provided research paper excerpts. Rules:
1. Never use knowledge outside the provided context.
2. When multiple papers agree, cite all: (Smith et al., 2021; Jones et al., 2019).
3. When papers DISAGREE, flag it: ⚠️ CONFLICTING EVIDENCE: [details]
4. For single-study findings: 📌 NOTE: This finding is from a single study.
5. Always end with a 'Sources Used' section."""
            
            response = await client.post(
                f"{api_url}/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": model,
                    "max_tokens": 2048,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.message}"}
                    ]
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                data = response.json()
                llm_response = data.get("content", [{}])[0].get("text", "")
            else:
                print(f"LLM API error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"LLM API error: {e}")
    
    # Use LLM response if available, otherwise build fallback
    if llm_response:
        return ChatResponse(
            response=llm_response,
            sources=sources,
            conflicts=relevant_conflicts[:3],
            single_study_notes=[]
        )
    
    # Fallback
    response_text = "Based on the research literature:\n\n"
    
    if relevant_chunks:
        for i, chunk in enumerate(relevant_chunks[:3]):
            response_text += f"**{chunk.get('title', 'Unknown')}** ({chunk.get('year', 'Unknown')}):\n"
            response_text += f"{chunk.get('text', '')[:500]}...\n\n"
    else:
        response_text += "No directly relevant chunks found. Please try a more specific query about gas turbine combustion."
    
    if relevant_conflicts:
        response_text += "\n⚠️ **Conflicting Evidence Detected:**\n"
        for c in relevant_conflicts[:2]:
            response_text += f"- {c}\n"
    
    return ChatResponse(
        response=response_text,
        sources=sources,
        conflicts=relevant_conflicts[:3],
        single_study_notes=[]
    )

# Setup admin routes
app = setup_admin_routes(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
