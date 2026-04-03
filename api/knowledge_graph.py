#!/usr/bin/env python3
"""Phase 3: Knowledge Graph Construction for Gas Turbine Combustion Expert

Builds a knowledge graph from processed papers:
- Entity extraction (physical quantities, components, phenomena)
- Relationship extraction (increases, decreases, affects, causes)
- Contradiction detection (papers that disagree)
- Consensus mapping (papers that agree)
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import networkx as nx
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CHUNKS_DIR = PROJECT_ROOT / "chunks"
METADATA_DIR = PROJECT_ROOT / "papers" / "metadata"
KNOWLEDGE_GRAPH_DIR = PROJECT_ROOT / "knowledge_graph"

KNOWLEDGE_GRAPH_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Entity:
    """Represents a physical entity in the domain"""
    entity_id: str
    name: str
    entity_type: str  # quantity, component, phenomenon, fuel, method
    synonyms: List[str] = field(default_factory=list)
    papers_mentioning: List[str] = field(default_factory=list)
    

@dataclass
class Relationship:
    """Represents a relationship between entities"""
    relationship_id: str
    source_entity: str
    target_entity: str
    relationship_type: str  # increases, decreases, affects, causes, correlates_with
    magnitude: Optional[str] = None  # "significant", "moderate", "weak"
    conditions: List[str] = field(default_factory=list)  # Under what conditions
    paper_id: str = ""
    context: str = ""  # Sentence excerpt
    confidence: float = 0.5
    

@dataclass
class Contradiction:
    """Represents conflicting findings between papers"""
    contradiction_id: str
    entity_1: str
    entity_2: str
    relationship_type: str
    paper_a: str
    paper_a_finding: str
    paper_b: str
    paper_b_finding: str
    resolution_notes: str = ""


# Domain-specific entity patterns for gas turbine combustion
ENTITY_PATTERNS = {
    'quantity': [
        r'NOx?\s*(?:emission[s]?)?', r'NO_x', r'nitrogen\s*oxide[s]?',
        r'CO\s*(?:emission[s]?)?', r'carbon\s*monoxide',
        r'temperature', r'pressure', r'velocity', r'swirl\s*number',
        r'equivalence\s*ratio', r'fuel\s*air\s*ratio',
        r'flame\s*temperature', r'adiabatic\s*flame\s*temperature',
        r'reynolds\s*number', r'mach\s*number',
        r'heat\s*release\s*rate', r'combustion\s*efficiency',
        r'flashback\s*(?:margin|limit)?', r'blowout\s*(?:limit|margin)?',
        r'lean\s*blowout', r'LBO', r'residence\s*time',
        r'emission[s]?\s*(?:index)?', r'emission[s]?\s*rate',
    ],
    'component': [
        r'combustor', r'combustion\s*chamber', r'flame\s*tube',
        r'liner', r'combustor\s*liner', r'dome',
        r'diffuser', r'premixer', r'premixing\s*section',
        r'swirler', r'swirl\s*generator', r'swirl\s*vane[s]?',
        r'fuel\s*injector', r'fuel\s*nozzle', r'atomizer',
        r'pilot', r'pilot\s*flame', r'main\s*stage',
        r'dilution\s*hole[s]?', r'cooling\s*hole[s]?',
        r'film\s*cooling', r'effusion\s*cooling',
        r'TAPS', r'LDI', r'RQL', r'DLE', r'LPP',
    ],
    'phenomenon': [
        r'flashback', r'flame\s*flashback',
        r'blowout', r'lean\s*blowout', r'LBO',
        r'combustion\s*instability', r'thermoacoustic\s*instability',
        r'oscillation[s]?', r'dynamic[s]?', r'pressure\s*oscillation',
        r'vortex\s*breakdown', r'PVC', r'precessing\s*vortex\s*core',
        r'flame\s*holding', r'flame\s*stabilization',
        r'blow\s*off', r'extinction',
        r'autoignition', r'ignition\s*delay',
        r'combustion\s*noise', r'entropy\s*wave',
    ],
    'fuel': [
        r'natural\s*gas', r'methane', r'CH4', r'hydrogen', r'H2',
        r'syngas', r'biogas', r'propane', r'jet\s*fuel',
        r'Jet-?A', r'JP-?\d+', r'kerosene',
        r'liquid\s*fuel', r'gaseous\s*fuel',
    ],
    'method': [
        r'PIV', r'particle\s*image\s*velocimetry',
        r'PLIF', r'laser-?induced\s*fluorescence',
        r'OH-?PLIF', r'CH-?PLIF',
        r'CARS', r'laser\s*diagnostic[s]?',
        r'LES', r'large\s*eddy\s*simulation',
        r'RANS', r'CFD', r'computational\s*fluid\s*dynamics',
        r'DNS', r'direct\s*numerical\s*simulation',
        r'emission[s]?\s*measurement',
    ],
}

# Relationship patterns
RELATIONSHIP_PATTERNS = {
    'increases': [
        r'(\w+(?:\s+\w+)?)\s+(?:increases?|increased|enhances?|enhanced|raises?|raised|elevates?|elevated)\s+(\w+(?:\s+\w+)?)',
        r'(?:higher|increased|elevated)\s+(\w+(?:\s+\w+)?)\s+(?:leads?\s+to|results?\s+in|causes?)\s+(?:higher|increased)\s+(\w+(?:\s+\w+)?)',
    ],
    'decreases': [
        r'(\w+(?:\s+\w+)?)\s+(?:decreases?|decreased|reduces?|reduced|lowers?|lowered|suppresses?|suppressed)\s+(\w+(?:\s+\w+)?)',
        r'(?:lower|decreased|reduced)\s+(\w+(?:\s+\w+)?)\s+(?:leads?\s+to|results?\s+in|causes?)\s+(?:lower|decreased)\s+(\w+(?:\s+\w+)?)',
    ],
    'affects': [
        r'(\w+(?:\s+\w+)?)\s+(?:affects?|affected|influences?|influenced|impacts?|impacted)\s+(\w+(?:\s+\w+)?)',
        r'(?:effect|impact|influence)\s+of\s+(\w+(?:\s+\w+)?)\s+on\s+(\w+(?:\s+\w+)?)',
    ],
    'correlates_with': [
        r'(\w+(?:\s+\w+)?)\s+(?:correlates?|correlated)\s+(?:with|to)\s+(\w+(?:\s+\w+)?)',
        r'(?:positive|negative)\s+correlation\s+between\s+(\w+(?:\s+\w+)?)\s+and\s+(\w+(?:\s+\w+)?)',
    ],
    'causes': [
        r'(\w+(?:\s+\w+)?)\s+(?:causes?|caused|triggers?|triggered|induces?|induced)\s+(\w+(?:\s+\w+)?)',
        r'(?:due\s+to|because\s+of|caused\s+by)\s+(\w+(?:\s+\w+)?)',
    ],
}


class KnowledgeGraphBuilder:
    """Builds knowledge graph from processed papers"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.contradictions: List[Contradiction] = []
        self.entity_mentions: Dict[str, Set[str]] = defaultdict(set)
        
    def extract_entities_from_text(self, text: str, paper_id: str) -> List[Tuple[str, str, str]]:
        """Extract entities from text. Returns list of (entity_id, name, type)"""
        entities_found = []
        text_lower = text.lower()
        
        for entity_type, patterns in ENTITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group(0).strip()
                    # Normalize entity name
                    entity_id = re.sub(r'\s+', '_', entity_name.lower())
                    entity_id = re.sub(r'[^a-z0-9_]', '', entity_id)
                    
                    if entity_id and entity_id not in self.entities:
                        self.entities[entity_id] = Entity(
                            entity_id=entity_id,
                            name=entity_name,
                            entity_type=entity_type,
                            papers_mentioning=[paper_id]
                        )
                    elif entity_id in self.entities and paper_id not in self.entities[entity_id].papers_mentioning:
                        self.entities[entity_id].papers_mentioning.append(paper_id)
                    
                    self.entity_mentions[entity_id].add(paper_id)
                    entities_found.append((entity_id, entity_name, entity_type))
        
        return entities_found
    
    def extract_relationships_from_text(self, text: str, paper_id: str, entities: List[Tuple[str, str, str]]) -> List[Relationship]:
        """Extract relationships between entities"""
        relationships = []
        entity_dict = {e[0]: e[1] for e in entities}
        
        # Build a set of entity names for quick lookup
        entity_names = set(entity_dict.keys())
        
        for rel_type, patterns in RELATIONSHIP_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        source = groups[0].strip().lower()
                        target = groups[-1].strip().lower()
                        
                        # Normalize
                        source_id = re.sub(r'\s+', '_', source)
                        source_id = re.sub(r'[^a-z0-9_]', '', source_id)
                        target_id = re.sub(r'\s+', '_', target)
                        target_id = re.sub(r'[^a-z0-9_]', '', target_id)
                        
                        # Only create relationship if both entities are recognized
                        if source_id in entity_names or target_id in entity_names:
                            rel = Relationship(
                                relationship_id=f"{paper_id}_{rel_type}_{source_id}_{target_id}",
                                source_entity=source_id,
                                target_entity=target_id,
                                relationship_type=rel_type,
                                paper_id=paper_id,
                                context=match.group(0)[:200],
                                confidence=0.7
                            )
                            relationships.append(rel)
        
        return relationships
    
    def detect_contradictions(self) -> List[Contradiction]:
        """Detect contradictions between papers"""
        contradictions = []
        
        # Group relationships by entity pair
        entity_pair_rels: Dict[Tuple[str, str], List[Relationship]] = defaultdict(list)
        for rel in self.relationships:
            key = (rel.source_entity, rel.target_entity)
            entity_pair_rels[key].append(rel)
        
        # Find contradictory relationships
        for (source, target), rels in entity_pair_rels.items():
            increases = [r for r in rels if r.relationship_type == 'increases']
            decreases = [r for r in rels if r.relationship_type == 'decreases']
            
            if increases and decreases:
                # Found contradiction!
                for inc in increases:
                    for dec in decreases:
                        if inc.paper_id != dec.paper_id:
                            contradiction = Contradiction(
                                contradiction_id=f"contradiction_{source}_{target}",
                                entity_1=source,
                                entity_2=target,
                                relationship_type='effect',
                                paper_a=inc.paper_id,
                                paper_a_finding=f"{source} increases {target}",
                                paper_b=dec.paper_id,
                                paper_b_finding=f"{source} decreases {target}",
                                resolution_notes="Requires expert review"
                            )
                            contradictions.append(contradiction)
        
        return contradictions
    
    def process_all_chunks(self):
        """Process all chunk files to build knowledge graph"""
        chunk_files = list(CHUNKS_DIR.glob("*.json"))
        total = len(chunk_files)
        
        logger.info(f"Processing {total} chunk files for knowledge graph...")
        
        for i, chunk_file in enumerate(chunk_files):
            try:
                with open(chunk_file, 'r') as f:
                    chunks = json.load(f)
                
                paper_id = chunks[0]['paper_id'] if chunks else 'unknown'
                
                for chunk in chunks:
                    text = chunk.get('text', '')
                    entities = self.extract_entities_from_text(text, paper_id)
                    rels = self.extract_relationships_from_text(text, paper_id, entities)
                    self.relationships.extend(rels)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"  Processed {i+1}/{total} files, {len(self.entities)} entities, {len(self.relationships)} relationships")
                    
            except Exception as e:
                logger.error(f"Error processing {chunk_file}: {e}")
        
        logger.info(f"Extracted {len(self.entities)} entities and {len(self.relationships)} relationships")
        
        # Detect contradictions
        self.contradictions = self.detect_contradictions()
        logger.info(f"Found {len(self.contradictions)} potential contradictions")
    
    def build_networkx_graph(self):
        """Build NetworkX graph from extracted data"""
        # Add entities as nodes
        for entity_id, entity in self.entities.items():
            self.graph.add_node(
                entity_id,
                name=entity.name,
                type=entity.entity_type,
                paper_count=len(entity.papers_mentioning)
            )
        
        # Add relationships as edges
        for rel in self.relationships:
            if rel.source_entity in self.graph and rel.target_entity in self.graph:
                self.graph.add_edge(
                    rel.source_entity,
                    rel.target_entity,
                    relationship_type=rel.relationship_type,
                    paper_id=rel.paper_id,
                    confidence=rel.confidence
                )
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def save_graph(self):
        """Save knowledge graph to files"""
        # Save entities
        entities_file = KNOWLEDGE_GRAPH_DIR / "entities.json"
        with open(entities_file, 'w') as f:
            json.dump([asdict(e) for e in self.entities.values()], f, indent=2)
        
        # Save relationships
        relationships_file = KNOWLEDGE_GRAPH_DIR / "relationships.json"
        with open(relationships_file, 'w') as f:
            json.dump([asdict(r) for r in self.relationships], f, indent=2)
        
        # Save contradictions
        contradictions_file = KNOWLEDGE_GRAPH_DIR / "contradictions.json"
        with open(contradictions_file, 'w') as f:
            json.dump([asdict(c) for c in self.contradictions], f, indent=2)
        
        # Save NetworkX graph
        graph_file = KNOWLEDGE_GRAPH_DIR / "knowledge_graph.gml"
        nx.write_gml(self.graph, graph_file)
        
        # Save summary statistics
        stats = {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'total_contradictions': len(self.contradictions),
            'entity_types': {},
            'relationship_types': {},
            'most_mentioned_entities': [],
            'processed_at': datetime.now().isoformat()
        }
        
        # Count by type
        for entity in self.entities.values():
            stats['entity_types'][entity.entity_type] = stats['entity_types'].get(entity.entity_type, 0) + 1
        
        for rel in self.relationships:
            stats['relationship_types'][rel.relationship_type] = stats['relationship_types'].get(rel.relationship_type, 0) + 1
        
        # Most mentioned entities
        sorted_entities = sorted(self.entities.values(), key=lambda e: len(e.papers_mentioning), reverse=True)[:20]
        stats['most_mentioned_entities'] = [
            {'name': e.name, 'type': e.entity_type, 'papers': len(e.papers_mentioning)}
            for e in sorted_entities
        ]
        
        stats_file = KNOWLEDGE_GRAPH_DIR / "graph_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved knowledge graph files to {KNOWLEDGE_GRAPH_DIR}")
        return stats


def main():
    builder = KnowledgeGraphBuilder()
    builder.process_all_chunks()
    builder.build_networkx_graph()
    stats = builder.save_graph()
    
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*60)
    print(f"Total Entities: {stats['total_entities']}")
    print(f"Total Relationships: {stats['total_relationships']}")
    print(f"Contradictions Found: {stats['total_contradictions']}")
    print("\nEntity Types:")
    for etype, count in stats['entity_types'].items():
        print(f"  {etype}: {count}")
    print("\nRelationship Types:")
    for rtype, count in stats['relationship_types'].items():
        print(f"  {rtype}: {count}")
    print("\nTop 10 Most Mentioned Entities:")
    for e in stats['most_mentioned_entities'][:10]:
        print(f"  {e['name']} ({e['type']}): {e['papers']} papers")
    print("="*60)
    
    return stats


if __name__ == "__main__":
    main()
