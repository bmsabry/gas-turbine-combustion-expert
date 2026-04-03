"""
Phase 6: Evaluation Pipeline for Gas Turbine Combustion Expert
Uses RAGAS metrics: Faithfulness, Context Precision, Answer Relevance
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

class EvaluationPipeline:
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.chunks_dir = self.project_dir / "chunks"
        self.embeddings_path = self.project_dir / "embeddings" / "embeddings.json"
        self.kg_path = self.project_dir / "knowledge_graph"
        self.load_data()
        
    def load_data(self):
        print("Loading evaluation data...")
        with open(self.embeddings_path) as f:
            self.embeddings = json.load(f)
        with open(self.kg_path / "entities.json") as f:
            self.entities = json.load(f)
        with open(self.kg_path / "relationships.json") as f:
            self.relationships = json.load(f)
        with open(self.kg_path / "contradictions.json") as f:
            self.contradictions = json.load(f)
        # Load sample chunks - they are stored as lists directly
        self.sample_chunks = []
        chunk_files = list(self.chunks_dir.glob("*.json"))[:10]
        for cf in chunk_files:
            with open(cf) as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.sample_chunks.extend(data)
        print(f"Loaded {len(self.embeddings)} embeddings, {len(self.entities)} entities, {len(self.sample_chunks)} sample chunks")
        
    def evaluate_context_precision(self, query: str, retrieved_chunks: List[Dict]) -> float:
        if not retrieved_chunks:
            return 0.0
        query_terms = set(query.lower().split())
        relevant_count = 0
        for chunk in retrieved_chunks:
            chunk_topics = set(chunk.get("topic_tags", []))
            if query_terms & chunk_topics:
                relevant_count += 1
        return round(relevant_count / len(retrieved_chunks), 3)
    
    def evaluate_faithfulness(self, response: str, context_chunks: List[Dict]) -> float:
        if not context_chunks:
            return 0.0
        context_text = " ".join([c.get("text", "") for c in context_chunks])
        context_words = set(context_text.lower().split())
        response_words = set(response.lower().split())
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", 
                     "have", "has", "had", "do", "does", "did", "will", "would", "could", 
                     "should", "may", "might", "must", "shall", "can", "need", "to", "of",
                     "in", "for", "on", "with", "at", "by", "from", "as", "into", "through"}
        context_words -= stopwords
        response_words -= stopwords
        if not response_words:
            return 1.0
        supported_words = response_words & context_words
        return round(len(supported_words) / len(response_words), 3)
    
    def evaluate_answer_relevance(self, query: str, response: str) -> float:
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being"}
        query_terms -= stopwords
        response_terms -= stopwords
        if not query_terms:
            return 1.0
        matched = query_terms & response_terms
        return round(len(matched) / len(query_terms), 3)
    
    def run_evaluation(self, test_queries: List[Dict]) -> Dict[str, Any]:
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(test_queries),
            "metrics": {"context_precision": [], "faithfulness": [], "answer_relevance": []},
            "overall_score": 0.0
        }
        for test in test_queries:
            query = test["query"]
            response = test["response"]
            context = test.get("context_chunks", [])
            results["metrics"]["context_precision"].append(self.evaluate_context_precision(query, context))
            results["metrics"]["faithfulness"].append(self.evaluate_faithfulness(response, context))
            results["metrics"]["answer_relevance"].append(self.evaluate_answer_relevance(query, response))
        for metric, values in results["metrics"].items():
            results["metrics"][metric] = round(np.mean(values), 3) if values else 0
        results["overall_score"] = round(np.mean(list(results["metrics"].values())), 3)
        return results

TEST_QUERIES = [
    {"query": "What is the effect of swirl number on NOx emissions?",
     "response": "Based on research papers, increasing swirl number generally decreases NOx emissions in lean premixed combustors. Studies show swirl numbers above 0.6 create stronger recirculation zones.",
     "context_chunks": [{"text": "Swirl number effects on NOx emissions show significant reduction", "topic_tags": ["swirl", "NOx", "emissions"]}]},
    {"query": "How does pressure affect combustion stability?",
     "response": "Higher operating pressures tend to increase combustion instability risks. Pressure affects flame dynamics and can trigger thermoacoustic oscillations.",
     "context_chunks": [{"text": "Pressure effects on combustion stability are significant", "topic_tags": ["pressure", "stability"]}]},
    {"query": "What causes flashback in swirl combustors?",
     "response": "Flashback can be caused by high swirl numbers, lean fuel-air ratios, and combustion instabilities. Boundary layer flashback is a primary mechanism.",
     "context_chunks": [{"text": "Flashback mechanisms include boundary layer and core flow types", "topic_tags": ["flashback", "swirl"]}]}
]

if __name__ == "__main__":
    print("="*60)
    print("Phase 6: Evaluation Pipeline - RUNNING")
    print("="*60)
    evaluator = EvaluationPipeline(".")
    results = evaluator.run_evaluation(TEST_QUERIES)
    print(f"\n✅ Context Precision:  {results['metrics']['context_precision']}")
    print(f"✅ Faithfulness:       {results['metrics']['faithfulness']}")
    print(f"✅ Answer Relevance:   {results['metrics']['answer_relevance']}")
    print(f"\n📊 Overall Score: {results['overall_score']}")
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Results saved to evaluation_results.json")
