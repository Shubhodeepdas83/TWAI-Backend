import os
import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv
import spacy
import re
import uuid
import hashlib
import base64
import tempfile
import requests
load_dotenv()

# Load spaCy model for natural language processing
nlp = spacy.load("en_core_web_sm")


def download_from_url(url):
    """
    Download a file from a public URL to a temporary local file.
    
    Args:
        url: Public URL of the PDF file
        
    Returns:
        Path to the local temporary file or None if download fails
    """
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_path = temp_file.name
        temp_file.close()
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return temp_path
    except Exception as e:
        print(f"Error downloading from URL: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)  # Remove the temporary file
        return None


def clean_text(text: str):
    """
    Clean and normalize text by removing excessive whitespace.
    
    Args:
        text: The input text to clean
        
    Returns:
        Cleaned text with normalized spacing
    """
    return re.sub(r'\s+', ' ', text).strip()


def extract_page_image_with_highlight(pdf_path: str, page_number: int, highlight_text: str):
    """
    Highlight specific text on a PDF page and save it as an image.
    
    Args:
        pdf_path: Path to the PDF file (local path or URL)
        page_number: Zero-based page number to process
        highlight_text: Text to highlight on the page
        
    Returns:
        Path to the saved image file or None if an error occurs
    """
    local_pdf_path = pdf_path
    is_url = pdf_path.startswith(('http://', 'https://'))
    
    try:
        # If it's a URL, download the file first
        if is_url:
            local_pdf_path = download_from_url(pdf_path)
            if not local_pdf_path:
                return None
        
        doc = fitz.open(local_pdf_path)
        print(page_number)
        page = doc[page_number]
        bbox_list = []

        # Split text into sentences and search for each in the PDF
        for sentence in nlp(highlight_text).sents:
            cleaned_sentence = clean_text(sentence.text)
            text_instances = page.search_for(cleaned_sentence)
            if text_instances:
                bbox_list.append(text_instances)
                page.add_highlight_annot(text_instances)

        # Create a highlight spanning from the first match to the last match
        if bbox_list:
            first_rect, last_rect = bbox_list[0][0], bbox_list[-1][-1]
            page.add_highlight_annot((first_rect[0], first_rect[1], last_rect[2], last_rect[3]))

        # Save the page as an image with 3x resolution
        img_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
        page.get_pixmap(matrix=fitz.Matrix(3, 3)).save(img_path)
        doc.close()
        
        # Clean up the temporary PDF if it was downloaded from a URL
        if is_url and os.path.exists(local_pdf_path):
            os.unlink(local_pdf_path)
            
        return img_path
    except Exception as e:
        print(f"Error highlighting PDF: {e}")
        # Clean up the temporary PDF if it was downloaded from a URL
        if is_url and local_pdf_path and os.path.exists(local_pdf_path):
            os.unlink(local_pdf_path)
        return None


def extract_page_image(pdf_path: str, page_number: int):
    """
    Extract a PDF page as an image without highlights.
    
    Args:
        pdf_path: Path to the PDF file (local path or URL)
        page_number: Zero-based page number to extract
        
    Returns:
        Path to the saved image file
    """
    local_pdf_path = pdf_path
    is_url = pdf_path.startswith(('http://', 'https://'))
    
    try:
        # If it's a URL, download the file first
        if is_url:
            local_pdf_path = download_from_url(pdf_path)
            if not local_pdf_path:
                return None
        
        doc = fitz.open(local_pdf_path)
        page = doc.load_page(page_number)
        img_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
        page.get_pixmap().save(img_path)
        doc.close()
        
        # Clean up the temporary PDF if it was downloaded from a URL
        if is_url and os.path.exists(local_pdf_path):
            os.unlink(local_pdf_path)
            
        return img_path
    except Exception as e:
        print(f"Error extracting page image: {e}")
        # Clean up the temporary PDF if it was downloaded from a URL
        if is_url and local_pdf_path and os.path.exists(local_pdf_path):
            os.unlink(local_pdf_path)
        return None


def generate_unique_filename(pdf_path, page_number):
    """
    Generate a unique filename based on the PDF path and page number.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: The page number
        
    Returns:
        A unique filename string
    """
    # Extract just the UUID part from S3 URLs if possible
    if pdf_path.startswith('http'):
        try:
            filename = pdf_path.split('/')[-1]
            if '-' in filename and '.pdf' in filename.lower():
                pdf_path = filename  # Use just the filename for hashing
        except:
            pass  # If extraction fails, use the full path
            
    pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()[:8]  # Short hash of PDF path
    unique_id = uuid.uuid4().hex[:8]  # Random unique identifier
    return f"img_{pdf_hash}_{page_number}_{unique_id}.png"


def encode_image_to_base64(image_path):
    """
    Encode an image file to a Base64 string and delete the file after encoding.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string or None if the file doesn't exist
    """
    if not image_path or not os.path.exists(image_path):
        return None

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Delete image after encoding to avoid clutter
    os.remove(image_path)
    return encoded_string


def extract_used_citations(response, citation_map, all_retrieved_documents):
    """
    Extract citation numbers from response and return only used citations.
    
    Args:
        response: The text response containing citations in [x] format
        citation_map: Dictionary mapping citation numbers to source descriptions
        all_retrieved_documents: List of document dictionaries with PDF paths and page numbers
        
    Returns:
        Dictionary of used citations with their descriptions and image data
    """
    # Extract all citation numbers from the response text
    used_citation_numbers = set(map(int, re.findall(r"\[(?:[^\]]*?(\d+)[^\]]*?)\]", response)))
    
    used_citations = {}
    processed_pages = set()  # Track already processed PDF pages

    for num in sorted(used_citation_numbers):
        if num not in citation_map:
            continue
            
        source_desc = citation_map[num]
        
        # Handle PDF citations
        if ".pdf" in source_desc.lower():
            text_chunk = None
            pdf_found = False
            
            for doc in all_retrieved_documents:
                pdf_path = doc.get("PDF Path")
                page_number = int(doc.get("Page Number"))
                
                # Skip if PDF path is missing
                if not pdf_path:
                    continue
                    
                # Create a unique ID for this page
                page_id = (pdf_path, page_number)
                if page_id in processed_pages:
                    continue
                
                # For S3 URLs, extract the filename/UUID part for matching
                pdf_identifier = pdf_path
                if pdf_path.startswith('http'):
                    pdf_identifier = pdf_path.split('/')[-1]  # Get the UUID.pdf part
                
                # Check if the PDF path or its identifier is in the source description
                if (pdf_path in source_desc or 
                    pdf_identifier in source_desc):
                    
                    processed_pages.add(page_id)
                    text_chunk = doc.get("Text Chunk")
                    
                    # Try to extract image with highlighted text
                    if text_chunk:
                        img_path = extract_page_image_with_highlight(pdf_path, page_number, text_chunk)
                        if img_path:
                            base64_image = encode_image_to_base64(img_path)
                            used_citations[num] = {
                                "description": f"{source_desc}",
                                "isimg": True,
                                "image_data": base64_image
                            }
                        else:
                            # Fallback to non-highlighted image if highlighting fails
                            img_path = extract_page_image(pdf_path, page_number)
                            base64_image = encode_image_to_base64(img_path)
                            used_citations[num] = {
                                "description": f"{source_desc}",
                                "isimg": True,
                                "image_data": base64_image
                            }
                    else:
                        # Extract page without highlights if no text chunk is available
                        img_path = extract_page_image(pdf_path, page_number)
                        base64_image = encode_image_to_base64(img_path)
                        used_citations[num] = {
                            "description": f"{source_desc}",
                            "isimg": True,
                            "image_data": base64_image
                        }
                    
                    pdf_found = True
                    break
            
            # Fallback if no matching document is found
            if not pdf_found:
                used_citations[num] = {
                    "description": source_desc,
                    "isimg": False,
                    "image_data": None
                }
                
        # Handle non-PDF citations (URLs/other sources)
        else:
            used_citations[num] = {
                "description": source_desc,
                "isimg": False,
                "image_data": None
            }

    return used_citations