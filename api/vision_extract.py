#!/usr/bin/env python3
"""
Gemini Flash Vision Extraction Pipeline
Extracts equations and figure descriptions from PDF pages using Gemini Flash via OpenRouter.
This script uses ZERO Claude tokens - only Gemini Flash.
"""

import json
import base64
import asyncio
import hashlib
import time
from pathlib import Path
from datetime import datetime
import httpx
import fitz  # PyMuPDF

PROJECT_DIR = Path('/a0/usr/projects/gas_turbine_combustion_expert_v1')
VISION_CHUNKS_DIR = PROJECT_DIR / 'chunks_vision'
VISION_CHUNKS_DIR.mkdir(exist_ok=True)

PROGRESS_FILE = PROJECT_DIR / 'vision_progress.json'

# Load API settings
with open(PROJECT_DIR / 'admin_settings.json') as f:
    settings = json.load(f)

API_KEY = settings['llm_api_key']
MODEL = 'google/gemini-2.0-flash-001'
API_URL = 'https://openrouter.ai/api/v1/chat/completions'

SEMAPHORE = asyncio.Semaphore(3)  # 3 concurrent requests max
RETRY_DELAY = 2
MAX_RETRIES = 3

PROMPT = """You are a scientific document analyzer. Analyze this page from a gas turbine combustion research paper.

Extract and describe:
1. All mathematical equations - write them in plain text notation (e.g. NOx = k * T^n * exp(-E/RT))
2. All graphs and charts - describe axes, trends, key data points and conclusions
3. All tables - summarize the data and what it shows
4. Any diagrams or schematics - describe what is shown

Be precise and technical. Focus on combustion engineering content.
If the page is mostly text with no figures/equations, respond with: NO_VISUAL_CONTENT
If there is content, start with: VISUAL_CONTENT_FOUND"""


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {'completed': [], 'failed': [], 'total_chunks': 0}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def page_to_base64(page, dpi=120):
    """Render a PDF page to base64 PNG image."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes('png')
    return base64.b64encode(img_bytes).decode('utf-8')


async def call_gemini_vision(client, image_b64, page_num, pdf_name):
    """Call Gemini Flash via OpenRouter with an image."""
    async with SEMAPHORE:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(
                    API_URL,
                    headers={
                        'Authorization': f'Bearer {API_KEY}',
                        'Content-Type': 'application/json',
                        'HTTP-Referer': 'https://gas-turbine-combustion-expert.onrender.com',
                        'X-Title': 'Gas Turbine Combustion Expert'
                    },
                    json={
                        'model': MODEL,
                        'messages': [{
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': PROMPT},
                                {'type': 'image_url', 'image_url': {
                                    'url': f'data:image/png;base64,{image_b64}'
                                }}
                            ]
                        }],
                        'max_tokens': 800
                    },
                    timeout=30.0
                )
                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    return content
                elif response.status_code == 429:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return None
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    return None
    return None


async def process_pdf(client, pdf_path, progress, stats):
    """Process a single PDF file."""
    pdf_key = pdf_path.name
    
    if pdf_key in progress['completed']:
        return 0
    
    try:
        doc = fitz.open(str(pdf_path))
        paper_id = hashlib.md5(str(pdf_path).encode()).hexdigest()[:12]
        title = pdf_path.stem[:60]
        vision_chunks = []
        
        pages_processed = 0
        for page_num, page in enumerate(doc):
            # Only process pages that have images
            if len(page.get_images()) == 0:
                continue
            
            try:
                img_b64 = page_to_base64(page)
                content = await call_gemini_vision(client, img_b64, page_num, pdf_path.name)
                
                if content and 'VISUAL_CONTENT_FOUND' in content:
                    # Clean up the content
                    content = content.replace('VISUAL_CONTENT_FOUND', '').strip()
                    
                    vision_chunks.append({
                        'chunk_id': f'vision_{paper_id}_p{page_num:04d}',
                        'paper_id': f'vision_{paper_id}',
                        'title': title,
                        'year': '',
                        'section': f'page_{page_num}_visual',
                        'text': f'[VISUAL CONTENT - Page {page_num+1}]\n{content}',
                        'topic_tags': ['combustion', 'equations', 'figures'],
                        'credibility_score': 4,
                        'chunk_type': 'vision',
                        'source_pdf': pdf_path.name,
                        'page_number': page_num
                    })
                    pages_processed += 1
                
            except Exception as e:
                pass
        
        doc.close()
        
        # Save vision chunks
        if vision_chunks:
            chunk_file = VISION_CHUNKS_DIR / f'vision_{paper_id}_chunks.json'
            with open(chunk_file, 'w') as f:
                json.dump(vision_chunks, f)
        
        progress['completed'].append(pdf_key)
        progress['total_chunks'] += len(vision_chunks)
        stats['pdfs_done'] += 1
        stats['chunks_created'] += len(vision_chunks)
        save_progress(progress)
        
        return len(vision_chunks)
        
    except Exception as e:
        progress['failed'].append({'pdf': pdf_key, 'error': str(e)})
        save_progress(progress)
        return 0


async def main():
    progress = load_progress()
    stats = {'pdfs_done': 0, 'chunks_created': 0, 'start_time': time.time()}
    
    pdfs = sorted((PROJECT_DIR / 'papers/raw').glob('*.pdf'))
    total_pdfs = len(pdfs)
    already_done = len(progress['completed'])
    remaining = [p for p in pdfs if p.name not in progress['completed']]
    
    print(f'\n{"="*60}')
    print(f'GEMINI FLASH VISION EXTRACTION PIPELINE')
    print(f'{"="*60}')
    print(f'Total PDFs: {total_pdfs}')
    print(f'Already processed: {already_done}')
    print(f'Remaining: {len(remaining)}')
    print(f'Using model: {MODEL} via OpenRouter')
    print(f'Starting at: {datetime.now().strftime("%H:%M:%S")}')
    print(f'{"="*60}\n')
    
    async with httpx.AsyncClient() as client:
        # Process in batches of 10 PDFs at a time
        batch_size = 10
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            tasks = [process_pdf(client, pdf, progress, stats) for pdf in batch]
            results = await asyncio.gather(*tasks)
            
            elapsed = time.time() - stats['start_time']
            total_done = already_done + stats['pdfs_done']
            pct = (total_done / total_pdfs) * 100
            rate = stats['pdfs_done'] / elapsed * 60 if elapsed > 0 else 0
            eta_min = (len(remaining) - stats['pdfs_done']) / rate if rate > 0 else 0
            
            print(f'[{datetime.now().strftime("%H:%M:%S")}] '
                  f'Progress: {total_done}/{total_pdfs} ({pct:.1f}%) | '
                  f'Vision chunks: {progress["total_chunks"]} | '
                  f'Rate: {rate:.1f} PDFs/min | '
                  f'ETA: {eta_min:.0f} min')
    
    print(f'\n{"="*60}')
    print(f'EXTRACTION COMPLETE!')
    print(f'Total PDFs processed: {stats["pdfs_done"]}')
    print(f'Total vision chunks created: {stats["chunks_created"]}')
    print(f'Failed: {len(progress["failed"])}')
    print(f'Vision chunks saved to: {VISION_CHUNKS_DIR}')
    print(f'{"="*60}')


if __name__ == '__main__':
    asyncio.run(main())
