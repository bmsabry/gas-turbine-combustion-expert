import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import re
import asyncio
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

from api.admin import setup_admin_routes, load_settings

app = FastAPI(title="Gas Turbine Combustion Expert API v2")
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
    sub_queries: List[str] = []


_chunks = []
_tfidf_matrix = None
_vectorizer = None
_neural_embeddings = None
_neural_chunk_ids = []
_entities = []
_relationships = []
_contradictions = []
_data_loaded = False


def strip_all_references(text: str) -> str:
    """Remove ALL forms of citations and references from LLM output."""
    # Step 1: Remove trailing reference/sources sections (everything after these headers)
    section_patterns = [
        r'\n+\*{0,3}\s*sources\s*used[\s\S]*$',
        r'\n+\*{0,3}\s*references[\s\S]*$',
        r'\n+\*{0,3}\s*bibliography[\s\S]*$',
        r'\n+\*{0,3}\s*citations[\s\S]*$',
        r'\n+\*{0,3}\s*works\s*cited[\s\S]*$',
        r'\n+---+\s*\n\s*\*{0,3}\s*(sources|references|bibliography)[\s\S]*$',
        # Perplexity numbered URL blocks at end
        r'\n+\[\d+\]\s*https?://[^\n]+(?:\n\[\d+\]\s*https?://[^\n]+)*\s*$',
        r'\n+\d+\.\s*https?://[^\n]+(?:\n\d+\.\s*https?://[^\n]+)*\s*$',
    ]
    for pat in section_patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)

    # Step 2: Remove inline citation numbers like [1], [2], [1][2], [3][4][5]
    # Perplexity style: [1], [2][3], etc.
    text = re.sub(r'(\[\d+\])+', '', text)

    # Step 3: Remove inline author-year citations like (Smith et al., 2021) or (Smith, 2021)
    text = re.sub(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?[,\s]+\d{4}[^)]*\)', '', text)

    # Step 4: Remove superscript-style refs like ^1 ^2
    text = re.sub(r'\^\d+', '', text)

    # Step 5: Clean up any double spaces left by removals
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r' ([.,;:?!])', r'\1', text)

    return text.strip()


def load_all_data():
    global _chunks, _tfidf_matrix, _vectorizer, _neural_embeddings, _neural_chunk_ids
    global _entities, _relationships, _contradictions, _data_loaded
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

    neural_path = PROJECT_DIR / "embeddings" / "neural_embeddings.npy"
    neural_ids_path = PROJECT_DIR / "embeddings" / "neural_chunk_ids.json"
    if neural_path.exists() and neural_ids_path.exists():
        try:
            _neural_embeddings = np.load(str(neural_path))
            with open(neural_ids_path) as f:
                _neural_chunk_ids = json.load(f)
            print(f"Neural embeddings loaded: {_neural_embeddings.shape}")
        except Exception as e:
            print(f"Neural embeddings load failed: {e}")

    kg_dir = PROJECT_DIR / "knowledge_graph"
    for fname, target in [("entities.json", "_entities"), ("relationships.json", "_relationships"), ("contradictions.json", "_contradictions")]:
        try:
            with open(kg_dir / fname) as f:
                globals()[target] = json.load(f)
        except Exception:
            pass

    print(f"KG: {len(_entities)} entities, {len(_contradictions)} conflicts")
    _data_loaded = True


async def decompose_query(query: str, client: httpx.AsyncClient, settings: dict) -> List[str]:
    """Use Gemini Flash to decompose complex queries into focused sub-queries."""
    api_key = settings.get("llm_api_key", "").strip()
    if not api_key:
        return [query]
    try:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://gas-turbine-combustion-expert.onrender.com",
                "X-Title": "Gas Turbine Combustion Expert"
            },
            json={
                "model": "google/gemini-2.0-flash-001",
                "max_tokens": 200,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Decompose this gas turbine combustion question into 2-4 specific search sub-queries.\n"
                        "Return ONLY a JSON array of strings, nothing else.\n"
                        'Example: ["NOx emissions swirl number effect", "lean premixed combustion stability"]\n'
                        f"Question: {query}"
                    )
                }]
            },
            timeout=15.0
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                sub_queries = json.loads(match.group())
                if isinstance(sub_queries, list) and sub_queries:
                    if query not in sub_queries:
                        sub_queries.insert(0, query)
                    return sub_queries[:4]
    except Exception as e:
        print(f"Query decomposition failed: {e}")
    return [query]


async def rerank_with_gemini(query: str, chunks: List[Dict], client: httpx.AsyncClient, settings: dict, top_k: int = 100) -> List[Dict]:
    """Use Gemini Flash to rerank chunks - returns top_k most relevant."""
    if len(chunks) <= top_k:
        return chunks
    api_key = settings.get("llm_api_key", "").strip()
    if not api_key:
        return chunks[:top_k]
    try:
        candidates = []
        for i, chunk in enumerate(chunks[:80]):
            text_preview = chunk.get("text", "")[:200].replace("\n", " ")
            candidates.append(f"{i}: {text_preview}")
        candidates_text = "\n".join(candidates)
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://gas-turbine-combustion-expert.onrender.com",
                "X-Title": "Gas Turbine Combustion Expert"
            },
            json={
                "model": "google/gemini-2.0-flash-001",
                "max_tokens": 400,
                "messages": [{
                    "role": "user",
                    "content": (
                        f"Gas turbine combustion expert: rank these text chunks by relevance to the question.\n"
                        f"Question: {query}\n\nChunks:\n{candidates_text}\n\n"
                        f"Return ONLY a JSON array of the top {top_k} chunk indices in order of relevance.\n"
                        "Example: [3, 0, 7, 12, 1]"
                    )
                }]
            },
            timeout=25.0
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                indices = json.loads(match.group())
                if isinstance(indices, list):
                    reranked = []
                    seen = set()
                    for idx in indices:
                        if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in seen:
                            reranked.append(chunks[idx])
                            seen.add(idx)
                    for i in range(len(chunks)):
                        if i not in seen and len(reranked) < top_k:
                            reranked.append(chunks[i])
                    return reranked[:top_k]
    except Exception as e:
        print(f"Reranking failed: {e}")
    return chunks[:top_k]


def search_chunks_tfidf(query: str, top_k: int = 300) -> List[Dict]:
    """TF-IDF retrieval returning top_k candidates."""
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

async def search_chunks_neural(query: str, client: httpx.AsyncClient, top_k: int = 300) -> List[Dict]:
    """Neural semantic search using Deep-Infra embeddings (all-MiniLM-L6-v2, 384-dim)."""
    if _neural_embeddings is None or not _neural_chunk_ids:
        print("Neural search: embeddings not loaded, skipping")
        return []
    deep_infra_key = os.environ.get("DEEP_INFRA_API_KEY", os.environ.get("DEEP-INFRA_API_KEY", "")).strip()
    if not deep_infra_key:
        print("Neural search: no Deep-Infra API key, skipping")
        return []
    try:
        resp = await client.post(
            "https://api.deepinfra.com/v1/openai/embeddings",
            headers={"Authorization": f"Bearer {deep_infra_key}", "Content-Type": "application/json"},
            json={"model": "sentence-transformers/all-MiniLM-L6-v2", "input": query},
            timeout=10.0
        )
        if resp.status_code != 200:
            print(f"Neural embed API error: {resp.status_code} {resp.text[:200]}")
            return []
        query_vec = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
        # Cosine similarity against all stored embeddings
        norms = np.linalg.norm(_neural_embeddings, axis=1)
        q_norm = np.linalg.norm(query_vec)
        sims = (_neural_embeddings @ query_vec) / (norms * q_norm + 1e-8)
        top_idx = np.argsort(sims)[::-1][:top_k]
        # Build chunk_id → chunk lookup
        chunk_lookup = {c.get("chunk_id", ""): c for c in _chunks if c.get("chunk_id")}
        results = []
        for idx in top_idx:
            if idx < len(_neural_chunk_ids):
                cid = _neural_chunk_ids[idx]
                chunk = chunk_lookup.get(cid)
                if chunk and sims[idx] > 0.1:
                    c = chunk.copy()
                    c["score"] = float(sims[idx])
                    c["retrieval_type"] = "neural"
                    results.append(c)
        print(f"Neural search: {len(results)} results for query '{query[:50]}'")
        return results
    except Exception as e:
        print(f"Neural search error: {e}")
        return []


def find_conflicts(query: str) -> List[str]:
    """Find knowledge graph contradictions relevant to query."""
    query_lower = query.lower()
    results = []
    for conflict in _contradictions[:100]:
        entity = conflict.get("entity", "").lower()
        if entity and entity in query_lower:
            pa = conflict.get("paper_a", "Study A")
            pb = conflict.get("paper_b", "Study B")
            fa = conflict.get("paper_a_finding", "makes a claim")
            fb = conflict.get("paper_b_finding", "contradicts")
            results.append(f"CONFLICTING EVIDENCE on '{entity}': {pa} states: {fa}. However, {pb} states: {fb}.")
    return results[:5]


@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_all_data)


@app.get("/")
async def root():
    for p in [STATIC_DIR / "index.html", PROJECT_DIR / "static" / "index.html"]:
        if p.exists():
            return FileResponse(str(p), media_type="text/html")
    return {"name": "Gas Turbine Combustion Expert API", "version": "2.0.0", "status": "running"}


@app.get("/api")
async def api_info():
    return {
        "name": "Gas Turbine Combustion Expert API",
        "version": "2.0.0",
        "status": "running",
        "neural_embeddings": _neural_embeddings is not None,
        "upgrades": ["query_decomposition", "gemini_reranking", "hybrid_neural_tfidf", "conversation_history", "600_chunk_context"]
    }


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
        "tfidf_loaded": _tfidf_matrix is not None,
        "neural_embeddings_loaded": _neural_embeddings is not None,
        "neural_embedding_count": len(_neural_chunk_ids) if _neural_chunk_ids else 0
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not _data_loaded:
        load_all_data()

    settings = load_settings()
    api_key = settings.get("llm_api_key", "").strip()
    api_url = settings.get("llm_api_url", "").strip().rstrip("/")
    model = settings.get("llm_model", "google/gemini-2.0-flash-001")

    async with httpx.AsyncClient(timeout=120.0) as client:

        # STEP 1: Query Decomposition
        sub_queries = await decompose_query(request.message, client, settings)
        print(f"Sub-queries: {sub_queries}")

        # STEP 2: Hybrid Retrieval — TF-IDF (per sub-query) + Neural (original query)
        all_retrieved = {}

        # TF-IDF: search each sub-query for keyword precision
        for sq in sub_queries:
            results = search_chunks_tfidf(sq, top_k=300)
            for r in results:
                cid = r.get("chunk_id", r.get("text", "")[:50])
                if cid not in all_retrieved or r["score"] > all_retrieved[cid]["score"]:
                    r["retrieval_type"] = "tfidf"
                    all_retrieved[cid] = r

        tfidf_count = len(all_retrieved)

        # Neural: search original query for semantic recall
        neural_results = await search_chunks_neural(request.message, client, top_k=300)
        neural_added = 0
        for r in neural_results:
            cid = r.get("chunk_id", r.get("text", "")[:50])
            if cid not in all_retrieved:
                all_retrieved[cid] = r
                neural_added += 1
            # If neural score is much higher, prefer neural score
            elif r["score"] > all_retrieved[cid]["score"] * 1.2:
                all_retrieved[cid]["score"] = r["score"]

        print(f"Hybrid retrieval: {tfidf_count} TF-IDF + {neural_added} neural-only = {len(all_retrieved)} total candidates")
        candidates = sorted(all_retrieved.values(), key=lambda x: x["score"], reverse=True)[:1200]


        if not candidates:
            return ChatResponse(
                response="No relevant information found in the research papers. Please rephrase your question.",
                sources=[], conflicts=[], single_study_notes=[], sub_queries=sub_queries
            )

        # STEP 3: Gemini Flash Reranking → top 600
        reranked = await rerank_with_gemini(request.message, candidates, client, settings, top_k=600)


        # STEP 4: Build sources list
        sources = []
        seen_titles = set()
        for chunk in reranked:
            title = chunk.get("title", chunk.get("paper_id", "Unknown"))
            if title not in seen_titles:
                seen_titles.add(title)
                sources.append({
                    "title": title,
                    "year": chunk.get("year", ""),
                    "credibility_score": chunk.get("credibility_score", 3),
                    "section": chunk.get("section", ""),
                    "relevance_score": round(chunk.get("score", 0), 3)
                })

        # STEP 5: Knowledge graph conflict detection
        conflicts = find_conflicts(request.message)

        # STEP 6: Build context from top 600 chunks (hybrid retrieval)
        context_parts = []
        for i, chunk in enumerate(reranked[:600]):
            title = chunk.get("title", "Unknown")
            year = chunk.get("year", "")
            text = chunk.get("text", "")
            section = chunk.get("section", "")
            chunk_type = chunk.get("chunk_type", "text")
            type_label = "[VISUAL/EQUATION] " if chunk_type == "vision" else ""
            context_parts.append(f"[{i+1}] {type_label}{title} ({year}) - {section}:\n{text}\n")
        context = "\n".join(context_parts)

        conflict_ctx = ""
        if conflicts:
            conflict_ctx = "\n\nLITERATURE CONFLICTS DETECTED:\n" + "\n".join(conflicts)

        # STEP 7: Conversation history (last 3 turns)
        history_messages = []
        for h in request.history[-6:]:
            if h.get("role") in ["user", "assistant"]:
                history_messages.append({"role": h["role"], "content": h["content"]})

        # STEP 8: Check API key
        if not api_key:
            preview = "\n\n".join(
                f"**[{i+1}] {c.get('title', 'Unknown')}**\n{c.get('text', '')[:300]}..."
                for i, c in enumerate(reranked[:5])
            )
            return ChatResponse(
                response="## LLM Not Configured\n\nPlease configure an LLM API key in the Admin Panel `/admin`.\n\n---\n\n**Top retrieved context:**\n\n" + preview,
                sources=sources, conflicts=conflicts, single_study_notes=[], sub_queries=sub_queries
            )

        # STEP 9: System prompt
        system_prompt = (
            "ABSOLUTE RULE — NO CITATIONS EVER: You are STRICTLY FORBIDDEN from including ANY citations, "
            "references, footnotes, or source attributions ANYWHERE in your answer. "
            "NO [1], NO [2], NO (Author, Year), NO 'according to', NO 'as shown by', NO 'Sources:', "
            "NO 'References:', NO numbered URL lists, NO footnote markers of any kind. "
            "Write ONLY the direct technical content. Zero attribution anywhere.\n\n"
            "You are a world-class gas turbine combustion expert with deep mastery of combustion physics, "
            "thermodynamics, fluid dynamics, chemical kinetics, emissions, turbomachinery, and combustor design. "
            "You think like a senior combustion engineer who has read everything in the field.\n\n"
            "KNOWLEDGE POLICY:\n"
            "- The provided research context is your PRIMARY source. Prioritize specific findings, "
            "data, numbers, and experimental results from those chunks.\n"
            "- You MAY and SHOULD supplement with your deep engineering training knowledge for "
            "well-established principles, mechanisms, governing equations, and frameworks — "
            "especially to fill gaps between what the retrieved chunks explicitly state.\n"
            "- NEVER invent specific experimental measurements or test results not supported by the context.\n"
            "- Use confidence language naturally: 'It is well established that...', "
            "'The research strongly supports...', 'Evidence suggests...', 'Limited data indicates...'\n\n"
            "HOW TO ANSWER EVERY QUESTION:\n"
            "1. WRITE AS A CONNECTED ENGINEERING NARRATIVE: Your answer must read like an explanation from a "
            "senior engineer, not a textbook index. Each paragraph must connect to the next. "
            "Use transitional language: 'This is why...', 'As a result...', 'Because of this...', "
            "'This directly affects...', 'The trade-off here is...'. "
            "The reader must feel they are being walked through the topic, not handed a categorized list.\n"
            "2. STRUCTURE WITH HEADERS BUT WRITE IN PROSE: Use ### headers for major topic sections. "
            "WITHIN each section, write 2-4 connected prose paragraphs that explain the concept, "
            "its governing physics, how it relates to neighboring concepts, and its trade-offs. "
            "Use bullet points ONLY when listing 5 or more parallel items of equal weight (e.g., a list of "
            "mechanisms, a list of parameters). NEVER use bullets as the primary content format.\n"
            "3. BUILD THE STORY: Structure your answer so it flows logically — from fundamentals to specifics, "
            "from physics to engineering practice, from problem to solution. "
            "Every section should make the next section make more sense.\n"
            "4. EQUATIONS — STRICT RULES: Write equations in clean plain text using Unicode symbols "
            "(∂, ∇, ∫, α, β, γ, ρ, φ, τ, ε, η, λ, ω, Σ, ∝, ≈, ≤, ≥, →). "
            "NEVER use LaTeX notation (no \\(, \\[, \\frac, \\nabla, \\int, $ signs). "
            "Write equations like: NOx ∝ τ·exp(-Ea/RT). "
            "Only include equations that are ESSENTIAL — maximum 3 key equations per response. "
            "Immediately after each equation, explain in one sentence what it physically means.\n"
            "5. TABLES — STRICT RULES: Use tables ONLY as a final summary comparison. "
            "Maximum 4 columns. Maximum 6 rows. Keep each cell to 5-8 words maximum — never full sentences in cells. "
            "If a table would need more than 4 columns or dense text in cells, write it as prose instead.\n"
            "6. BE QUANTITATIVE IN PROSE: Weave specific numbers into your narrative paragraphs naturally — "
            "'...which raises NOx by 15-20% per 100K increase above 1800K' rather than isolated bullet points of numbers.\n"
            "7. FLAG CONFLICTS NATURALLY: When evidence conflicts, address it within the relevant paragraph: "
            "⚠️ CONFLICTING EVIDENCE: [what one body of research shows]. However, [what other evidence shows]. "
            "The difference comes down to [explain the key condition that determines which applies].\n"
            "8. CLOSE WITH PRACTICAL GUIDANCE: End with a ### Practical Engineering Guidance section. "
            "Write 2-3 prose sentences giving the overall recommendation, then a compact table (3 columns max: "
            "Scenario | Recommended Approach | Key Reason) summarizing when to use what.\n"
            "9. AUDIENCE AND TONE: Write for experienced combustion engineers. Be precise and rigorous. "
            "Do not over-explain basic concepts. Do not hedge unnecessarily. "
            "Aim for the tone of a confident expert explaining to a peer, not a textbook chapter."
        )

        # STEP 10: Assemble user message
        user_msg = (
            f"Research Context (top 600 most relevant chunks from 386 PDFs — hybrid neural+TF-IDF retrieval — papers, textbooks, "
            f"equations, figures):\n"
            f"{context}{conflict_ctx}\n\n"
            f"Question: {request.message}\n\n"
            "Write a comprehensive, deeply technical answer as a CONNECTED ENGINEERING NARRATIVE — "
            "not a list of categories. Each section should flow into the next. "
            "Use prose paragraphs as the primary format. Use ### headers for major sections. "
            "Use bullets only for 5+ parallel items. "
            "Use ONE compact summary table at the end (max 4 cols, 6 rows, brief cell text). "
            "Write equations in plain text Unicode only (no LaTeX). "
            "End with ### Practical Engineering Guidance. "
            "DO NOT include any citations, references, or source lists anywhere."
        )





        messages_with_history = history_messages + [{"role": "user", "content": user_msg}]

        # STEP 11: Call LLM
        if "anthropic" in api_url.lower():
            endpoint = f"{api_url}/v1/messages"
            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            payload = {
                "model": model,
                "max_tokens": 8000,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_msg}]
            }
        else:
            # OpenAI-compatible (OpenRouter, Perplexity, OpenAI, etc.)
            endpoint = f"{api_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://gas-turbine-combustion-expert.onrender.com",
                "X-Title": "Gas Turbine Combustion Expert"
            }
            payload = {
                "model": model,
                "max_tokens": 8000,
                "messages": [{"role": "system", "content": system_prompt}] + messages_with_history
            }

        try:
            resp = await client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if "anthropic" in api_url.lower():
                raw_text = data["content"][0]["text"]
            else:
                raw_text = data["choices"][0]["message"]["content"]

            # Strip ALL forms of references/citations from the output
            llm_text = strip_all_references(raw_text)

            return ChatResponse(
                response=llm_text,
                sources=sources,
                conflicts=conflicts,
                single_study_notes=[],
                sub_queries=sub_queries
            )
        except httpx.HTTPStatusError as e:
            err = e.response.text[:500] if e.response else str(e)
            return ChatResponse(
                response=f"## API Error\n\nStatus: {e.response.status_code}\n\n{err}\n\nCheck your API key in Admin Panel.",
                sources=sources, conflicts=conflicts, single_study_notes=[], sub_queries=sub_queries
            )
        except Exception as e:
            return ChatResponse(
                response=f"## Error\n\n{str(e)}\n\nCheck API settings in Admin Panel.",
                sources=sources, conflicts=conflicts, single_study_notes=[], sub_queries=sub_queries
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
