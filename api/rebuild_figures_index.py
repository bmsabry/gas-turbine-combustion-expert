
#!/usr/bin/env python3
"""
Rebuild figures index by re-processing PDFs and matching existing images.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF

PROJECT_DIR = Path("/a0/usr/projects/gas_turbine_combustion_expert_v1")
PAPERS_DIR = PROJECT_DIR / "papers" / "raw"
VISION_CHUNKS_DIR = PROJECT_DIR / "chunks_vision"
FIGURES_DIR = PROJECT_DIR / "figures"
METADATA_DIR = PROJECT_DIR / "figures_metadata"

def load_vision_chunks() -> Dict[str, List[Dict]]:
    """Load all vision chunks indexed by source_pdf."""
    chunks_by_pdf = {}
    
    for chunk_file in VISION_CHUNKS_DIR.glob("*.json"):
        try:
            with open(chunk_file, 'r') as f:
                chunks = json.load(f)
                for chunk in chunks:
                    source_pdf = chunk.get('source_pdf', '')
                    if source_pdf:
                        if source_pdf not in chunks_by_pdf:
                            chunks_by_pdf[source_pdf] = []
                        chunks_by_pdf[source_pdf].append(chunk)
        except Exception as e:
            print(f"Error loading {chunk_file}: {e}")
            
    print(f"Loaded vision chunks for {len(chunks_by_pdf)} PDFs")
    return chunks_by_pdf

def get_paper_id(pdf_name: str) -> str:
    return pdf_name.replace('.pdf', '')

def find_vision_chunk_for_page(pdf_name: str, page_num: int, vision_chunks: Dict) -> Optional[Dict]:
    paper_id = get_paper_id(pdf_name)
    
    if pdf_name in vision_chunks:
        for chunk in vision_chunks[pdf_name]:
            if chunk.get('page_number') == page_num:
                return chunk
    
    for source_pdf, chunks in vision_chunks.items():
        if paper_id in source_pdf or source_pdf in paper_id:
            for chunk in chunks:
                if chunk.get('page_number') == page_num:
                    return chunk
    
    return None

def extract_figure_description(vision_text: str) -> Optional[str]:
    import re
    description_parts = []
    
    figure_pattern = r'(Figure\s*\d+[a-z]?[\s:.-][^\n]+(?:\n(?![A-Z]\.|\d\.\s)[^\n]*)*)'
    matches = re.findall(figure_pattern, vision_text, re.IGNORECASE)
    
    if matches:
        for match in matches[:3]:
            description_parts.append(match.strip())
    
    if 'Diagrams' in vision_text or 'Schematics' in vision_text:
        diagrams_section = re.search(
            r'\*\*Diagrams[^\*]+\*\*([^\n]+(?:\n(?![A-Z]\*\*)[^\n]*)*)',
            vision_text,
            re.IGNORECASE
        )
        if diagrams_section:
            description_parts.append(diagrams_section.group(1).strip())
    
    return ' '.join(description_parts) if description_parts else None

def generate_figure_id(pdf_name: str, page_num: int, fig_index: int) -> str:
    unique_str = f"{pdf_name}_page{page_num}_fig{fig_index}"
    hash_suffix = hashlib.md5(unique_str.encode()).hexdigest()[:8]
    return f"fig_{hash_suffix}"

def process_all_pdfs():
    """Process all PDFs and build figures index."""
    vision_chunks = load_vision_chunks()
    all_figures = []
    all_pdfs = list(PAPERS_DIR.glob("*.pdf"))
    
    print(f"Processing {len(all_pdfs)} PDFs...")
    
    existing_figures = {}
    for fig_path in FIGURES_DIR.glob("*.png"):
        existing_figures[fig_path.name] = fig_path
    
    print(f"Found {len(existing_figures)} existing figure images")
    
    for i, pdf_path in enumerate(all_pdfs):
        if (i + 1) % 50 == 0:
            print(f"Processing [{i+1}/{len(all_pdfs)}] {pdf_path.name}...")
        
        pdf_name = pdf_path.name
        try:
            pdf_doc = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                images = page.get_images(full=True)
                
                if not images:
                    continue
                    
                page_rect = page.rect
                page_width = page_rect.width
                page_height = page_rect.height
                
                fig_index = 0
                for img_index, img_info in enumerate(images):
                    try:
                        xref = img_info[0]
                        img_rects = page.get_image_rects(xref)
                        
                        if not img_rects:
                            continue
                            
                        for rect in img_rects:
                            img_width = rect.width
                            img_height = rect.height
                            
                            if img_width < page_width * 0.05 and img_height < page_height * 0.05:
                                continue
                            if img_width < 50 or img_height < 50:
                                continue
                            
                            fig_id = generate_figure_id(pdf_name, page_num, fig_index)
                            fig_filename = f"{fig_id}.png"
                            
                            if fig_filename not in existing_figures:
                                continue
                            
                            fig_path = existing_figures[fig_filename]
                            
                            vision_chunk = find_vision_chunk_for_page(pdf_name, page_num, vision_chunks)
                            
                            figure_metadata = {
                                'figure_id': fig_id,
                                'source_pdf': pdf_name,
                                'page_number': page_num,
                                'figure_index': fig_index,
                                'image_path': str(fig_path),
                                'position': {
                                    'x_pct': float(rect.x0 / page_width),
                                    'y_pct': float(rect.y0 / page_height),
                                    'width_pct': float(img_width / page_width),
                                    'height_pct': float(img_height / page_height)
                                },
                                'vision_chunk': vision_chunk,
                                'description': None
                            }
                            
                            if vision_chunk:
                                figure_metadata['description'] = extract_figure_description(
                                    vision_chunk.get('text', '')
                                )
                            
                            all_figures.append(figure_metadata)
                            fig_index += 1
                            
                    except Exception as e:
                        continue
            
            pdf_doc.close()
            
        except Exception as e:
            print(f"Error processing {pdf_name}: {e}")
    
    metadata = {
        'total_figures': len(all_figures),
        'total_pdfs_processed': len(all_pdfs),
        'figures': all_figures
    }
    
    output_file = METADATA_DIR / "figures_index.json"
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n=== COMPLETE ===")
    print(f"Total figures indexed: {len(all_figures)}")
    print(f"Saved to: {output_file}")
    
    return metadata

if __name__ == "__main__":
    process_all_pdfs()
