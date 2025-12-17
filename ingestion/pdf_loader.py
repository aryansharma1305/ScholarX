"""PDF loading and text extraction."""
import requests
from typing import Optional
from pypdf import PdfReader
from io import BytesIO
from ingestion.text_cleaner import preprocess_text
from utils.logger import get_logger

logger = get_logger(__name__)


def load_pdf_from_url(url: str, timeout: int = 30) -> str:
    """
    Download and extract text from a PDF URL.
    
    Args:
        url: URL to the PDF file
        timeout: Request timeout in seconds
        
    Returns:
        Extracted and preprocessed text
    """
    try:
        logger.info(f"Downloading PDF from: {url}")
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Read PDF from bytes
        pdf_bytes = BytesIO(response.content)
        reader = PdfReader(pdf_bytes)
        
        # Extract text from all pages
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text())
        
        raw_text = "\n".join(text_parts)
        
        if not raw_text or not raw_text.strip():
            raise ValueError("PDF contains no extractable text")
        
        # Preprocess the text
        cleaned_text = preprocess_text(raw_text)
        
        logger.info(f"Successfully extracted {len(cleaned_text)} characters from PDF")
        return cleaned_text
        
    except requests.RequestException as e:
        logger.error(f"Failed to download PDF: {e}")
        raise ValueError(f"Failed to download PDF from URL: {e}")
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise ValueError(f"Failed to extract text from PDF: {e}")


def load_pdf_from_file(file_path: str) -> str:
    """
    Load and extract text from a local PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted and preprocessed text
    """
    try:
        logger.info(f"Loading PDF from file: {file_path}")
        reader = PdfReader(file_path)
        
        # Extract text from all pages
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text())
        
        raw_text = "\n".join(text_parts)
        
        if not raw_text or not raw_text.strip():
            raise ValueError("PDF contains no extractable text")
        
        # Preprocess the text
        cleaned_text = preprocess_text(raw_text)
        
        logger.info(f"Successfully extracted {len(cleaned_text)} characters from PDF")
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Failed to load PDF from file: {e}")
        raise ValueError(f"Failed to load PDF from file: {e}")


def extract_pdf_metadata(text: str) -> dict:
    """
    Extract basic metadata from PDF text.
    
    Args:
        text: Extracted PDF text
        
    Returns:
        Dictionary with title, abstract (if found)
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Try to extract title (usually first few lines)
    title = ' '.join(lines[:3])[:500] if lines else "Untitled Document"
    
    # Try to find abstract
    abstract = None
    text_lower = text.lower()
    abstract_index = text_lower.find('abstract')
    if abstract_index != -1:
        abstract_start = abstract_index + 8
        abstract_end = text.find('\n\n', abstract_start)
        if abstract_end > 0:
            abstract = text[abstract_start:abstract_end].strip()
        else:
            abstract = text[abstract_start:abstract_start + 500].strip()
    
    return {
        "title": title,
        "abstract": abstract,
    }



