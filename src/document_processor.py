import re
from typing import List, Dict, Any
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from each page of a PDF file.
    Returns a list of dicts: [{'page': page_num, 'text': page_text}]
    """
    pages_data = []
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Clean up any null bytes or weird formatting
                text = text.replace('\x00', '')
                pages_data.append({
                    "page": i + 1,
                    "text": text
                })
    except Exception as e:
        raise ValueError(f"Failed to read PDF file at {pdf_path}: {e}")
    return pages_data

def clean_text(text: str) -> str:
    """
    Basic text cleaning to standardize whitespaces and remove line break issues.
    """
    # Replace multiple whitespaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_document(
    pages_data: List[Dict[str, Any]], 
    doc_name: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Splits the extracted page text into overlapping chunks.
    Maintains page citations and metadata for each chunk.
    """
    chunks = []
    chunk_counter = 0

    for page_info in pages_data:
        page_num = page_info["page"]
        text = clean_text(page_info["text"])
        
        if not text:
            continue
            
        # Character-based sliding window chunking
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # If this is not the last chunk, try to split at a space to avoid cutting words
            if end < text_len:
                last_space = chunk_text.rfind(' ')
                if last_space > chunk_size * 0.7:  # Only split at space if it's reasonably close to the end
                    end = start + last_space
                    chunk_text = text[start:end]

            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": f"{doc_name}_p{page_num}_c{chunk_counter}",
                    "doc_name": doc_name,
                    "page": page_num,
                    "text": chunk_text
                })
                chunk_counter += 1
            
            # Slide window by chunk_size - overlap
            start = end - chunk_overlap
            if start >= text_len or end >= text_len:
                break
                
    return chunks
