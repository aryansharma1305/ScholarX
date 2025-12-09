"""Text cleaning and preprocessing utilities."""
import re


def clean_text(text: str) -> str:
    """
    Clean and preprocess text.
    
    Removes:
    - Excessive whitespace
    - Special characters that might interfere
    - Normalizes line breaks
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove special control characters
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def remove_headers_footers(text: str, header_threshold: int = 3) -> str:
    """
    Remove common headers and footers.
    
    Args:
        text: Input text
        header_threshold: Number of times a line must repeat to be considered header/footer
    """
    lines = text.split('\n')
    
    # Count line frequencies
    line_counts = {}
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line:
            line_counts[cleaned_line] = line_counts.get(cleaned_line, 0) + 1
    
    # Filter out lines that appear too frequently (likely headers/footers)
    filtered_lines = [
        line for line in lines
        if not line.strip() or line_counts.get(line.strip(), 0) < header_threshold
    ]
    
    return '\n'.join(filtered_lines)


def normalize_ligatures(text: str) -> str:
    """Replace common ligatures with standard characters."""
    ligature_map = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
        '–': '-',
        '—': '-',
        '…': '...',
        '“': '"',
        '”': '"',
        ''': "'",
        ''': "'",
    }
    
    for ligature, replacement in ligature_map.items():
        text = text.replace(ligature, replacement)
    
    return text


def preprocess_text(text: str) -> str:
    """
    Complete text preprocessing pipeline.
    
    Applies all cleaning steps in order.
    """
    text = normalize_ligatures(text)
    text = remove_headers_footers(text)
    text = clean_text(text)
    return text

