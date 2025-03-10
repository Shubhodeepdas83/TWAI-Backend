from openai import OpenAI
import fitz  # PyMuPDF
import pprint
import os
import time  # Import time for timestamps
from datetime import datetime  # Import datetime for readable timestamps
from dotenv import load_dotenv
import tkinter as tk
from tkinter import filedialog

load_dotenv()
GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def log_time(stage):
    """Logs the timestamp for a given stage."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {stage}")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# Use tkinter to open file dialog
root = tk.Tk()
root.withdraw()
pdf_path = filedialog.askopenfilename(
    title="Select a PDF file",
    filetypes=[("PDF files", "*.pdf")]
)

if not pdf_path:
    print("No file selected. Exiting...")
    exit()

log_time("Starting Text Extraction from PDF")
# Extract text from PDF
pdf_text = extract_text_from_pdf(pdf_path)
log_time("Completed Text Extraction from PDF")

log_time("Starting Text Processing - LLM")

response = client.chat.completions.create(
    model="gemini-1.5-pro",
    n=1,
    messages=[
        {
            "role": "user",
            "content": f"Analyze the following PDF content and provide key insights:\n\n{pdf_text}"
        }
    ]
)

print(response.choices[0].message.content)

log_time("Completed Text Processing - LLM")

# Results gemini-2.0-flash:
# [2025-03-10 22:41:13] Starting Text Extraction from PDF
# [2025-03-10 22:41:13] Completed Text Extraction from PDF
# [2025-03-10 22:41:13] Starting Text Processing - LLM
# [2025-03-10 22:41:20] Completed Text Processing - LLM

# Results gemini-2.0-flash-lite:
# [2025-03-10 22:47:37] Starting Text Extraction from PDF
# [2025-03-10 22:47:37] Completed Text Extraction from PDF
# [2025-03-10 22:47:37] Starting Text Processing - LLM
# [2025-03-10 22:47:47] Completed Text Processing - LLM

# Results gemini-1.5-flash:
# [2025-03-10 22:46:04] Starting Text Extraction from PDF
# [2025-03-10 22:46:04] Completed Text Extraction from PDF
# [2025-03-10 22:46:04] Starting Text Processing - LLM
# [2025-03-10 22:46:10] Completed Text Processing - LLM

# Results gemini-1.5-flash-8b:
# [2025-03-10 22:49:39] Starting Text Extraction from PDF
# [2025-03-10 22:49:39] Completed Text Extraction from PDF
# [2025-03-10 22:49:39] Starting Text Processing - LLM
# [2025-03-10 22:49:44] Completed Text Processing - LLM

# Results gemini-1.5-pro:
# [2025-03-10 22:50:53] Starting Text Extraction from PDF
# [2025-03-10 22:50:53] Completed Text Extraction from PDF
# [2025-03-10 22:50:53] Starting Text Processing - LLM
# [2025-03-10 22:51:06] Completed Text Processing - LLM