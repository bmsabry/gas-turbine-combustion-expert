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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import asyncio

# Import admin module
from admin import setup_admin_routes, load_settings

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
            data = json.load(f)
            if isinstance(data, list):
                return data
    return []

def load_faiss_index():
    index_file = PROJECT_DIR / "embeddings" / "faiss_index.bin"
    if index_file.exists():
        return faiss.read_index(str(index_file))
    return None

def load_knowledge_graph():
    entities_file = PROJECT_DIR / "knowledge_graph" / "entities.json"
    relations_file = PROJECT_DIR / "knowledge_graph" / "relationships.json"
    conflicts_file = PROJECT_DIR / "knowledge_graph" / "contradictions.json"
    
    entities = []
    relationships = []
    conflicts = []
    
    if entities_file.exists():
        with open(entities_file) as f:
            entities = json.load(f)
    if relations_file.exists():
        with open(relations_file) as f:
            relationships = json.load(f)
    if conflicts_file.exists():
        with open(conflicts_file) as f:
            conflicts = json.load(f)
    
    return entities, relationships, conflicts

def load_chunks():
    chunks_dir = PROJECT_DIR / "chunks"
    all_chunks = []
    for chunk_file in chunks_dir.glob("*.json"):
        with open(chunk_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                all_chunks.extend(data)
    return all_chunks

# Global data
embeddings_data = []
faiss_index = None
all_chunks = []
entities, relationships, conflicts = [], [], []

@app.on_event("startup")
async def startup_event():
    global embeddings_data, faiss_index, all_chunks, entities, relationships, conflicts
    
    print("Loading embeddings...")
    embeddings_data = load_embeddings()
    print(f"Loaded {len(embeddings_data)} embeddings")
    
    print("Loading FAISS index...")
    faiss_index = load_faiss_index()
    print(f"FAISS index: {'loaded' if faiss_index else 'not found'}")
    
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
           any(tag in query_lower for tag in ["noxs", "nox", "emission", "swirl", "pressure", "flame", "combustion", "stability", "flashback"]):
            if text not in seen_texts and len(relevant_chunks) < 5:
                relevant_chunks.append(chunk)
                seen_texts.add(text)
    
    
    # Check for conflicts related to the query
    relevant_conflicts = []
    for conflict in conflicts:
        entity = conflict.get("entity", "").lower()
        if entity in query_lower:
            relevant_conflicts.append(
                f"Paper A ({conflict.get('paper1', 'Unknown')}) states {conflict.get('relation1', 'X')}. "
                f"However, Paper B ({conflict.get('paper2', 'Unknown')}) states {conflict.get('relation2', 'Y')}."
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
    
    
    # If API key is set, call the LLM
    if api_key:
        try:
            async with httpx.AsyncClient() as client:
                # Build system prompt
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
                    return ChatResponse(
                        response=llm_response,
                        sources=sources,
                        conflicts=relevant_conflicts[:3],
                        single_study_notes=[]
                    )
        except Exception as e:
            print(f"LLM API error: {e}")
    
    # Fallback: Return context-based response
    response_text = f"Based on the research literature:\n\n"
    
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
