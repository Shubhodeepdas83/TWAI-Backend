from openai import OpenAI
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": f"Analyze the following PDF content and provide key insights:\n\n{pdf_text}"
        }
    ]
)

# Extract the response after </think>
response = completion.choices[0].message.content

# Print the response
print(response)
log_time("Completed Text Processing - LLM")

# Results gpt-4o-mini :
# [2025-03-10 21:53:35] Starting Text Extraction from PDF
# [2025-03-10 21:53:35] Completed Text Extraction from PDF
# [2025-03-10 21:53:35] Starting Text Processing - LLM
# [2025-03-10 21:53:57] Completed Text Processing - LLM

# Result gpt-4o:
# [2025-03-10 21:57:38] Starting Text Extraction from PDF
# [2025-03-10 21:57:38] Completed Text Extraction from PDF
# [2025-03-10 21:57:38] Starting Text Processing - LLM
# [2025-03-10 21:57:54] Completed Text Processing - LLM

# Result gpt-4:
# [2025-03-10 21:59:04] Starting Text Extraction from PDF
# [2025-03-10 21:59:04] Completed Text Extraction from PDF
# [2025-03-10 21:59:04] Starting Text Processing - LLM
# [2025-03-10 21:59:26] Completed Text Processing - LLM