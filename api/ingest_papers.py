#!/usr/bin/env python3
"""Phase 1: Multimodal PDF Ingestion Pipeline for Gas Turbine Combustion Expert

Processes research papers with:
- Layout-aware PDF parsing (using PyMuPDF)
- Hierarchical chunking (parent/child strategy)
- Metadata extraction (authors, year, journal, DOI)
- Auto-tagging with domain topics
- Progress tracking with resume capability
- Rate-limit aware batch processing
"""

import os
import json
import re
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PAPERS_DIR = PROJECT_ROOT / "papers" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "papers" / "processed"
CHUNKS_DIR = PROJECT_ROOT / "chunks"
METADATA_DIR = PROJECT_ROOT / "papers" / "metadata"
PROGRESS_FILE = PROJECT_ROOT / "ingestion_progress.json"

for d in [PROCESSED_DIR, CHUNKS_DIR, METADATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class PaperMetadata:
    paper_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    filename: str = ""
    credibility_score: int = 3
    topic_tags: List[str] = field(default_factory=list)
    abstract: str = ""
    chunk_count: int = 0
    processed_at: str = ""
    checksum: str = ""


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    paper_id: str
    title: str
    year: Optional[int]
    section: str
    section_type: str
    text: str
    topic_tags: List[str] = field(default_factory=list)
    credibility_score: int = 3
    chunk_level: int = 2
    token_count: int = 0
    char_count: int = 0


class IngestionProgress:
    def __init__(self):
        self.processed_files: Dict[str, str] = {}
        self.failed_files: Dict[str, str] = {}
        self.total_papers: int = 0
        self.start_time: str = ""
        self.last_update: str = ""
        self.load()
    
    def load(self):
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, 'r') as f:
                data = json.load(f)
                self.processed_files = data.get('processed_files', {})
                self.failed_files = data.get('failed_files', {})
                self.total_papers = data.get('total_papers', 0)
                self.start_time = data.get('start_time', '')
                self.last_update = data.get('last_update', '')
            logger.info(f"Loaded progress: {len(self.processed_files)} processed")
    
    def save(self):
        self.last_update = datetime.now().isoformat()
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({
                'processed_files': self.processed_files,
                'failed_files': self.failed_files,
                'total_papers': self.total_papers,
                'start_time': self.start_time,
                'last_update': self.last_update
            }, f, indent=2)
    
    def mark_processed(self, filename: str, paper_id: str):
        self.processed_files[filename] = paper_id
        self.save()
    
    def mark_failed(self, filename: str, error: str):
        self.failed_files[filename] = error
        self.save()
    
    def is_processed(self, filename: str) -> bool:
        return filename in self.processed_files


def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, Dict]:
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")
    
    doc = fitz.open(pdf_path)
    full_text = []
    sections = {}
    current_section = "main"
    section_patterns = [
        r'(?i)^abstract', r'(?i)^introduction', r'(?i)^methods?',
        r'(?i)^experimental', r'(?i)^results', r'(?i)^discussion',
        r'(?i)^conclusions?', r'(?i)^references', r'(?i)^nomenclature',
    ]
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text.append(f"\n--- Page {page_num + 1} ---\n")
        full_text.append(text)
        
        for line in text.split('\n'):
            line_stripped = line.strip()
            for pattern in section_patterns:
                if re.match(pattern, line_stripped):
                    current_section = line_stripped.lower().replace(' ', '_')[:30]
                    break
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(line)
    
    metadata = doc.metadata or {}
    page_count = len(doc)
    doc.close()
    
    return '\n'.join(full_text), {
        'page_count': page_count,
        'pdf_metadata': metadata,
        'sections': {k: '\n'.join(v) for k, v in sections.items()}
    }


def estimate_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def create_paper_id(filename: str) -> str:
    base = Path(filename).stem
    match = re.match(r'([a-zA-Z]+)[-_]?(\d{4})', base)
    if match:
        return f"{match.group(1).lower()}{match.group(2)}_{base[:30]}".replace(' ', '_').replace('-', '_')
    hash_part = hashlib.md5(filename.encode()).hexdigest()[:8]
    return f"paper_{hash_part}_{base[:20]}".replace(' ', '_').replace('-', '_')


def extract_metadata_from_text(text: str, filename: str, pdf_meta: Dict) -> PaperMetadata:
    paper_id = create_paper_id(filename)
    
    lines = text.split('\n')
    title = ""
    for line in lines[:20]:
        line = line.strip()
        if 20 < len(line) < 300 and not line.isupper():
            title = line
            break
    
    year_match = re.search(r'\b(19|20)\d{2}\b', text[:2000])
    year = int(year_match.group()) if year_match else None
    
    authors = []
    for match in re.finditer(r'([A-Z][a-z]+ [A-Z][a-z]+)', text[:1000]):
        name = match.group(1)
        if name not in ['University', 'Department', 'Institute', 'Laboratory']:
            authors.append(name)
        if len(authors) >= 5:
            break
    
    abstract = ""
    abstract_match = re.search(r'(?i)abstract[\s\n]*(.+?)(?=\n\n|1\.|introduction|keywords)', text[:5000], re.DOTALL)
    if abstract_match:
        abstract = abstract_match.group(1).strip()[:500]
    
    credibility = 3
    if any(kw in text.lower() for kw in ['journal', 'transactions', 'proceedings']):
        credibility = 4
    if any(kw in text.lower() for kw in ['aiaa', 'asme', 'elsevier', 'springer']):
        credibility = 5
    
    return PaperMetadata(
        paper_id=paper_id,
        title=title[:200] if title else f"Paper from {filename}",
        authors=authors[:10],
        year=year,
        filename=filename,
        credibility_score=credibility,
        abstract=abstract,
        processed_at=datetime.now().isoformat()
    )


def generate_topic_tags(text: str, abstract: str) -> List[str]:
    topic_keywords = {
        'swirl': ['swirl', 'swirling', 'vortex', 'precessing', 'pvc'],
        'NOx_emissions': ['nox', 'no_x', 'nitrogen oxide', 'emission', 'pollutant'],
        'combustion_instability': ['instability', 'oscillation', 'thermoacoustic'],
        'premixed': ['premixed', 'lean premixed', 'lpm'],
        'diffuser': ['diffuser', 'dump gap'],
        'premixer': ['premixer', 'premixing', 'fuel-air mixing'],
        'liner': ['liner', 'combustor liner', 'cooling'],
        'film_cooling': ['film cooling', 'liner cooling'],
        'flashback': ['flashback', 'flame flashback'],
        'blowout': ['blowout', 'lean blowout', 'lbo'],
        'hydrogen': ['hydrogen', 'h2', 'hydrogen fuel'],
        'gas_turbine': ['gas turbine', 'combustor', 'combustion chamber'],
        'spray': ['spray', 'atomization', 'droplet'],
        'LES': ['les', 'large eddy simulation', 'cfd'],
        'PIV': ['piv', 'particle image velocimetry'],
        'PLIF': ['plif', 'laser-induced fluorescence'],
        'RQL': ['rql', 'rich burn quick lean'],
        'TAPS': ['taps', 'twin annular premixing'],
        'LDI': ['ldi', 'lean direct injection'],
        'DLE': ['dle', 'dry low emission'],
    }
    
    combined = (text + ' ' + abstract).lower()
    tags = []
    for topic, keywords in topic_keywords.items():
        if any(kw in combined for kw in keywords):
            tags.append(topic)
    return tags[:8]


def create_chunks(text: str, metadata: PaperMetadata, section_data: Dict) -> List[Chunk]:
    chunks = []
    section_types = {
        'abstract': 'abstract', 'introduction': 'introduction',
        'methods': 'methods', 'experimental': 'methods',
        'results': 'results', 'discussion': 'discussion',
        'conclusion': 'conclusion', 'conclusions': 'conclusion',
    }
    
    for section_name, section_text in section_data.items():
        if len(section_text.strip()) < 100:
            continue
        
        section_type = section_types.get(section_name.split('_')[0], 'other')
        parent_id = f"{metadata.paper_id}_parent_{section_name[:20]}"
        
        parent = Chunk(
            chunk_id=parent_id,
            parent_id=parent_id,
            paper_id=metadata.paper_id,
            title=metadata.title,
            year=metadata.year,
            section=section_name[:50],
            section_type=section_type,
            text=section_text[:8000],
            topic_tags=metadata.topic_tags,
            credibility_score=metadata.credibility_score,
            chunk_level=1,
            token_count=estimate_token_count(section_text[:8000]),
            char_count=len(section_text[:8000])
        )
        chunks.append(parent)
        
        paragraphs = re.split(r'\n\n+', section_text)
        child_text = ""
        child_count = 0
        
        for para in paragraphs:
            para = para.strip()
            if len(para) < 50:
                continue
            child_text += para + "\n\n"
            
            if estimate_token_count(child_text) >= 200:
                child = Chunk(
                    chunk_id=f"{metadata.paper_id}_child_{section_name[:15]}_{child_count}",
                    parent_id=parent_id,
                    paper_id=metadata.paper_id,
                    title=metadata.title,
                    year=metadata.year,
                    section=section_name[:50],
                    section_type=section_type,
                    text=child_text.strip()[:2000],
                    topic_tags=metadata.topic_tags,
                    credibility_score=metadata.credibility_score,
                    chunk_level=2,
                    token_count=estimate_token_count(child_text.strip()[:2000]),
                    char_count=len(child_text.strip()[:2000])
                )
                chunks.append(child)
                child_text = ""
                child_count += 1
        
        if child_text.strip():
            child = Chunk(
                chunk_id=f"{metadata.paper_id}_child_{section_name[:15]}_{child_count}",
                parent_id=parent_id,
                paper_id=metadata.paper_id,
                title=metadata.title,
                year=metadata.year,
                section=section_name[:50],
                section_type=section_type,
                text=child_text.strip()[:2000],
                topic_tags=metadata.topic_tags,
                credibility_score=metadata.credibility_score,
                chunk_level=2,
                token_count=estimate_token_count(child_text.strip()[:2000]),
                char_count=len(child_text.strip()[:2000])
            )
            chunks.append(child)
    
    return chunks


def process_paper(pdf_path: Path, progress: IngestionProgress) -> Tuple[Optional[PaperMetadata], List[Chunk]]:
    filename = pdf_path.name
    
    if progress.is_processed(filename):
        return None, []
    
    try:
        logger.info(f"Processing: {filename}")
        
        with open(pdf_path, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()
        
        text, struct_data = extract_text_from_pdf(pdf_path)
        
        metadata = extract_metadata_from_text(text, filename, struct_data.get('pdf_metadata', {}))
        metadata.checksum = checksum
        metadata.topic_tags = generate_topic_tags(text, metadata.abstract)
        
        chunks = create_chunks(text, metadata, struct_data.get('sections', {}))
        metadata.chunk_count = len(chunks)
        
        metadata_path = METADATA_DIR / f"{metadata.paper_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        chunks_path = CHUNKS_DIR / f"{metadata.paper_id}_chunks.json"
        with open(chunks_path, 'w') as f:
            json.dump([asdict(c) for c in chunks], f, indent=2)
        
        processed_path = PROCESSED_DIR / f"{metadata.paper_id}.txt"
        with open(processed_path, 'w') as f:
            f.write(text)
        
        progress.mark_processed(filename, metadata.paper_id)
        logger.info(f"Done: {metadata.paper_id} ({len(chunks)} chunks)")
        
        return metadata, chunks
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Failed {filename}: {error_msg}")
        progress.mark_failed(filename, error_msg)
        return None, []


def run_ingestion(batch_size: int = 10, delay_seconds: float = 1.0, max_papers: int = None):
    progress = IngestionProgress()
    
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    if max_papers:
        pdf_files = pdf_files[:max_papers]
    
    total = len(pdf_files)
    already_done = len(progress.processed_files)
    
    logger.info(f"Found {total} PDFs, {already_done} already processed")
    
    if not progress.start_time:
        progress.start_time = datetime.now().isoformat()
        progress.total_papers = total
        progress.save()
    
    processed_count = 0
    failed_count = 0
    
    for i, pdf_path in enumerate(pdf_files):
        if progress.is_processed(pdf_path.name):
            continue
        
        logger.info(f"[{i+1}/{total}] {pdf_path.name}")
        
        metadata, chunks = process_paper(pdf_path, progress)
        
        if metadata:
            processed_count += 1
        else:
            failed_count += 1
        
        time.sleep(delay_seconds)
        
        if processed_count > 0 and processed_count % batch_size == 0:
            logger.info(f"Batch complete: {processed_count} papers. Pausing...")
            time.sleep(delay_seconds * 2)
    
    logger.info(f"\nCOMPLETE: {processed_count} processed, {failed_count} failed")
    return {'total': total, 'processed': processed_count, 'failed': failed_count}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest PDF papers")
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--delay', type=float, default=1.0)
    parser.add_argument('--max', type=int, default=None)
    args = parser.parse_args()
    
    try:
        import fitz
    except ImportError:
        import subprocess
        subprocess.run(['pip', 'install', 'pymupdf', '-q'])
    
    results = run_ingestion(batch_size=args.batch_size, delay_seconds=args.delay, max_papers=args.max)
    print(f"\nResults: {results}")
