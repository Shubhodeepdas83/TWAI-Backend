from groq import Groq
import fitz  # PyMuPDF
import pprint
import os
import time  # Import time for timestamps
from datetime import datetime  # Import datetime for readable timestamps
from dotenv import load_dotenv
import tkinter as tk
from tkinter import filedialog

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

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
# Pass the extracted text to the LLM
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": f"Analyze the following PDF content and provide key insights:\n\n{pdf_text}"
        }
    ]
)

# Extract the response after </think>
response = completion.choices[0].message.content.split('</think>')[-1].strip()

# Print the response
print(response)
log_time("Completed Text Processing - LLM")

# Results qwen-qwq-32b:
# [2025-03-10 21:54:45] Starting Text Extraction from PDF
# [2025-03-10 21:54:45] Completed Text Extraction from PDF
# [2025-03-10 21:54:45] Starting Text Processing - LLM
# [2025-03-10 21:54:50] Completed Text Processing - LLM

# Results llama-3.3-70b-versatile:
# [2025-03-10 22:02:01] Starting Text Extraction from PDF
# [2025-03-10 22:02:02] Completed Text Extraction from PDF
# [2025-03-10 22:02:02] Starting Text Processing - LLM
# [2025-03-10 22:02:04] Completed Text Processing - LLM

# Results deepseek-r1-distill-llama-70b:
# [2025-03-10 22:03:52] Starting Text Extraction from PDF
# [2025-03-10 22:03:52] Completed Text Extraction from PDF
# [2025-03-10 22:03:52] Starting Text Processing - LLM
# [2025-03-10 22:03:57] Completed Text Processing - LLM