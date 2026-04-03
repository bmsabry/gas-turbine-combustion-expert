#!/usr/bin/env python3
"""Phase 4: Vector Embeddings for Gas Turbine Combustion Expert

Embeds all child chunks using sentence-transformers and stores in Qdrant.
Uses local embeddings (no API needed) with all-MiniLM-L6-v2 model.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CHUNKS_DIR = PROJECT_ROOT / "chunks"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Global variables
model = None

def get_model():
    """Load embedding model (lazy loading)"""
    global model
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers model...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return model

def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Embed a list of texts using sentence-transformers"""
    model = get_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings.tolist()

def load_all_chunks() -> List[Dict[str, Any]]:
    """Load all chunk files and extract child chunks"""
    all_chunks = []
    chunk_files = list(CHUNKS_DIR.glob("*.json"))
    
    logger.info(f"Loading {len(chunk_files)} chunk files...")
    
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r') as f:
                chunks = json.load(f)
            
            # Only embed child chunks (level 2)
            for chunk in chunks:
                if chunk.get('chunk_level') == 2 or 'child' in chunk.get('chunk_id', ''):
                    all_chunks.append({
                        'chunk_id': chunk['chunk_id'],
                        'parent_id': chunk.get('parent_id', ''),
                        'paper_id': chunk.get('paper_id', ''),
                        'text': chunk.get('text', ''),
                        'section': chunk.get('section', ''),
                        'section_type': chunk.get('section_type', ''),
                        'topic_tags': chunk.get('topic_tags', []),
                        'credibility_score': chunk.get('credibility_score', 3),
                        'year': chunk.get('year'),
                        'title': chunk.get('title', '')
                    })
        except Exception as e:
            logger.error(f"Error loading {chunk_file}: {e}")
    
    logger.info(f"Loaded {len(all_chunks)} child chunks")
    return all_chunks

def create_embeddings_file(chunks: List[Dict], output_path: Path):
    """Create embeddings file with vectors"""
    texts = [c['text'] for c in chunks]
    
    logger.info(f"Embedding {len(texts)} texts...")
    start_time = time.time()
    
    embeddings = embed_texts(texts, batch_size=64)
    
    elapsed = time.time() - start_time
    logger.info(f"Embedding completed in {elapsed:.1f}s ({len(texts)/elapsed:.1f} texts/sec)")
    
    # Create embedding records
    records = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        records.append({
            'chunk_id': chunk['chunk_id'],
            'parent_id': chunk['parent_id'],
            'paper_id': chunk['paper_id'],
            'text': chunk['text'],
            'embedding': embedding,
            'metadata': {
                'section': chunk['section'],
                'section_type': chunk['section_type'],
                'topic_tags': chunk['topic_tags'],
                'credibility_score': chunk['credibility_score'],
                'year': chunk['year'],
                'title': chunk['title']
            }
        })
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i+1}/{len(chunks)} chunks")
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump({
            'embeddings': records,
            'model': 'all-MiniLM-L6-v2',
            'dimension': 384,
            'total_chunks': len(records),
            'created_at': datetime.now().isoformat()
        }, f)
    
    logger.info(f"Saved {len(records)} embeddings to {output_path}")
    return records

def create_faiss_index(embeddings_path: Path, index_path: Path):
    """Create FAISS index for fast similarity search"""
    try:
        import faiss
        import numpy as np
        
        with open(embeddings_path, 'r') as f:
            data = json.load(f)
        
        embeddings = [r['embedding'] for r in data['embeddings']]
        vectors = np.array(embeddings, dtype='float32')
        
        # Create FAISS index
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        
        # Save index
        faiss.write_index(index, str(index_path))
        
        # Save ID mapping
        id_mapping = [r['chunk_id'] for r in data['embeddings']]
        with open(index_path.with_suffix('.ids'), 'w') as f:
            json.dump(id_mapping, f)
        
        logger.info(f"Created FAISS index with {len(id_mapping)} vectors")
        return True
    except ImportError:
        logger.warning("FAISS not installed, skipping index creation")
        return False

def main():
    """Main embedding pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Embed chunks for semantic search")
    parser.add_argument('--batch-size', type=int, default=64, help='Embedding batch size')
    parser.add_argument('--output', type=str, default='embeddings.json', help='Output filename')
    args = parser.parse_args()
    
    print("="*60)
    print("PHASE 4: VECTOR EMBEDDINGS")
    print("="*60)
    
    # Load chunks
    chunks = load_all_chunks()
    
    if not chunks:
        logger.error("No chunks found!")
        return
    
    # Create embeddings
    output_path = EMBEDDINGS_DIR / args.output
    records = create_embeddings_file(chunks, output_path)
    
    # Create FAISS index
    create_faiss_index(output_path, EMBEDDINGS_DIR / 'faiss_index.bin')
    
    print("\n" + "="*60)
    print("EMBEDDING COMPLETE")
    print(f"  Total chunks embedded: {len(records)}")
    print(f"  Embedding dimension: 384")
    print(f"  Model: all-MiniLM-L6-v2")
    print(f"  Output: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
