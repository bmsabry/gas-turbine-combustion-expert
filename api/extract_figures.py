
#!/usr/bin/env python3
"""
Figure Extraction System for Gas Turbine Combustion Expert
Extracts figures from PDFs and links them to vision chunks for semantic search.
"""

import os
import sys
import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import fitz  # PyMuPDF
from PIL import Image
import io

# Project paths
PROJECT_DIR = Path("/a0/usr/projects/gas_turbine_combustion_expert_v1")
PAPERS_DIR = PROJECT_DIR / "papers" / "raw"
VISION_CHUNKS_DIR = PROJECT_DIR / "chunks_vision"
FIGURES_DIR = PROJECT_DIR / "figures"
METADATA_DIR = PROJECT_DIR / "figures_metadata"

class FigureExtractor:
    """Extract figures from PDFs and link to vision chunks."""
    
    def __init__(self):
        self.figures_dir = FIGURES_DIR
        self.metadata_dir = METADATA_DIR
        self.figures_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Load vision chunks for linking
        self.vision_chunks = self._load_vision_chunks()
        
    def _load_vision_chunks(self) -> Dict[str, List[Dict]]:
        """Load all vision chunks indexed by source_pdf."""
        chunks_by_pdf = {}
        
        if not VISION_CHUNKS_DIR.exists():
            print(f"Warning: Vision chunks directory not found: {VISION_CHUNKS_DIR}")
            return chunks_by_pdf
            
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
    
    def _get_paper_id(self, pdf_name: str) -> str:
        """Extract paper_id from PDF filename."""
        return pdf_name.replace('.pdf', '')
    
    def _find_vision_chunk_for_page(self, pdf_name: str, page_num: int) -> Optional[Dict]:
        """Find the vision chunk for a specific page."""
        paper_id = self._get_paper_id(pdf_name)
        
        if pdf_name in self.vision_chunks:
            for chunk in self.vision_chunks[pdf_name]:
                if chunk.get('page_number') == page_num:
                    return chunk
        
        for source_pdf, chunks in self.vision_chunks.items():
            if paper_id in source_pdf or source_pdf in paper_id:
                for chunk in chunks:
                    if chunk.get('page_number') == page_num:
                        return chunk
        
        return None
    
    def _extract_figure_regions(self, page, page_num: int, pdf_name: str) -> List[Dict]:
        """Extract figure regions from a PDF page."""
        figures = []
        
        images = page.get_images(full=True)
        
        if not images:
            return figures
            
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
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
                    
                    x_pct = rect.x0 / page_width
                    y_pct = rect.y0 / page_height
                    
                    figures.append({
                        'xref': xref,
                        'rect': rect,
                        'page_num': page_num,
                        'position': {
                            'x_pct': float(x_pct),
                            'y_pct': float(y_pct),
                            'width_pct': float(img_width / page_width),
                            'height_pct': float(img_height / page_height)
                        }
                    })
                    
            except Exception as e:
                print(f"  Error processing image {img_index}: {e}")
                continue
                
        return figures
    
    def _extract_image(self, pdf_doc, xref: int) -> Optional[bytes]:
        """Extract image data from PDF."""
        try:
            base_image = pdf_doc.extract_image(xref)
            if base_image:
                return base_image["image"]
        except Exception as e:
            print(f"  Error extracting image: {e}")
        return None
    
    def _generate_figure_id(self, pdf_name: str, page_num: int, fig_index: int) -> str:
        """Generate unique figure ID."""
        unique_str = f"{pdf_name}_page{page_num}_fig{fig_index}"
        hash_suffix = hashlib.md5(unique_str.encode()).hexdigest()[:8]
        return f"fig_{hash_suffix}"
    
    def extract_figures_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """Extract all figures from a single PDF."""
        pdf_name = pdf_path.name
        figures = []
        
        try:
            pdf_doc = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                
                fig_regions = self._extract_figure_regions(page, page_num, pdf_name)
                
                for fig_index, fig_region in enumerate(fig_regions):
                    img_data = self._extract_image(pdf_doc, fig_region['xref'])
                    
                    if not img_data:
                        continue
                    
                    fig_id = self._generate_figure_id(pdf_name, page_num, fig_index)
                    
                    fig_filename = f"{fig_id}.png"
                    fig_path = self.figures_dir / fig_filename
                    
                    try:
                        img = Image.open(io.BytesIO(img_data))
                        if img.mode in ('CMYK', 'RGBA', 'P'):
                            img = img.convert('RGB')
                        img.save(fig_path, 'PNG', optimize=True)
                    except Exception as e:
                        print(f"  Error saving image {fig_id}: {e}")
                        continue
                    
                    vision_chunk = self._find_vision_chunk_for_page(pdf_name, page_num)
                    
                    figure_metadata = {
                        'figure_id': fig_id,
                        'source_pdf': pdf_name,
                        'page_number': page_num,
                        'figure_index': fig_index,
                        'image_path': str(fig_path),
                        'position': fig_region['position'],
                        'vision_chunk': vision_chunk,
                        'description': None
                    }
                    
                    if vision_chunk:
                        figure_metadata['description'] = self._extract_figure_description(
                            vision_chunk.get('text', ''),
                            fig_region['position']
                        )
                    
                    figures.append(figure_metadata)
                    
            pdf_doc.close()
            
        except Exception as e:
            print(f"Error processing {pdf_name}: {e}")
            
        return figures
    
    def _extract_figure_description(self, vision_text: str, position: Dict) -> Optional[str]:
        """Extract relevant figure description from vision chunk text."""
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
    
    def process_all_pdfs(self, limit: Optional[int] = None) -> Dict:
        """Process all PDFs and extract figures."""
        all_figures = []
        all_pdfs = list(PAPERS_DIR.glob("*.pdf"))
        
        if limit:
            all_pdfs = all_pdfs[:limit]
            
        print(f"Processing {len(all_pdfs)} PDFs...")
        
        for i, pdf_path in enumerate(all_pdfs):
            print(f"\n[{i+1}/{len(all_pdfs)}] Processing {pdf_path.name}...")
            
            figures = self.extract_figures_from_pdf(pdf_path)
            all_figures.extend(figures)
            
            print(f"  Extracted {len(figures)} figures")
        
        metadata_file = self.metadata_dir / "figures_index.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'total_figures': len(all_figures),
                'total_pdfs_processed': len(all_pdfs),
                'figures': all_figures
            }, f, indent=2)
            
        print(f"\n=== EXTRACTION COMPLETE ===")
        print(f"Total figures extracted: {len(all_figures)}")
        print(f"Metadata saved to: {metadata_file}")
        
        return {
            'total_figures': len(all_figures),
            'figures': all_figures
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract figures from PDF papers')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of PDFs to process')
    args = parser.parse_args()
    
    extractor = FigureExtractor()
    extractor.process_all_pdfs(limit=args.limit)


if __name__ == "__main__":
    main()
