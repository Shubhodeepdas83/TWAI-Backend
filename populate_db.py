import os
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma


load_dotenv()

# Constants
SCHROMA_PATH = os.getenv("SCHROMA_PATH")
RCHROMA_PATH = os.getenv("RCHROMA_PATH")
DATA_PATH = os.getenv("DATA_PATH")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")
os.environ['OPENAI_API_KEY'] = api_key

# Utility functions
def initialize_directories():
    """Ensure the data directory exists."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

def load_documents():
    """Load documents from the data directory."""
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documentsS(documents):
    """Split documents into chunks using SemanticChunker."""
    text_splitterS = SemanticChunker(
        OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
    )
    return text_splitterS.split_documents(documents)

def split_documentsR(documents):
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    text_splitterR = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitterR.split_documents(documents)

def add_to_chroma(chunks, chroma_path):
    """Add chunks to the Chroma database."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set. Please check your .env file.")
        return

    os.environ['OPENAI_API_KEY'] = api_key

    db = Chroma(
        persist_directory=chroma_path, embedding_function=OpenAIEmbeddings()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
        print("✅ Documents added successfully!")
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    """Calculate unique IDs for each chunk."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks

def populate_database():
    """Load, split, and add documents to the database."""
    try:
        print("📄 Loading documents from the data folder...")
        documents = load_documents()

        print("📄 Semantic Splitting documents into chunks...")
        chunksS = split_documentsS(documents)
        print("📂 Adding chunks to the Chroma database (Semantic)...")
        add_to_chroma(chunksS, SCHROMA_PATH)

        print("📄 Recursive Splitting documents into chunks...")
        chunksR = split_documentsR(documents)
        print("📂 Adding chunks to the Chroma database (Recursive)...")
        add_to_chroma(chunksR, RCHROMA_PATH)

    except Exception as e:
        print(f"❌ An error occurred during database population: {e}")

def main():
    """Command-line interface for document processing."""
    initialize_directories()

    print("Welcome to the Document Uploader for Chroma DB!")
    print("Please place your PDF files in the 'data' folder.")

    user_input = input("Would you like to process the documents now? (yes/no): ").strip().lower()

    if user_input == "yes":
        populate_database()
    else:
        print("Exiting program. Add PDFs to 'data' and run again.")

if __name__ == "__main__":
    main()
