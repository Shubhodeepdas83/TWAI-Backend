import os
import time
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pinecone import  ServerlessSpec
from pinecone.grpc import PineconeGRPC
load_dotenv()


pc = PineconeGRPC(os.getenv("PINECONE_API_KEY"))

# pc.create_index(
#     name="quickstart",
#     dimension=1536,
#     metric="cosine",
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     )
# )





# Constants
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# INDEX_NAME = os.getenv("INDEX_NAME")
DATA_PATH = os.getenv("DATA_PATH")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")
os.environ['OPENAI_API_KEY'] = api_key

# Initialize Pinecone using gRPC
pc = PineconeGRPC(api_key=PINECONE_API_KEY)

# Utility functions
def initialize_directories():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documentsR(documents):
    text_splitterR = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len, is_separator_regex=False
    )
    return text_splitterR.split_documents(documents)

def add_to_pinecone(chunks, index_name, user_gmail):
    namespace = user_gmail.replace("@", "_").replace(".", "_")  # Format namespace

    # Ensure the index exists
    existing_indexes = [index["name"] for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        raise ValueError(f"Index {index_name} does not exist. Please create it first.")
    
    index = pc.Index(name=INDEX_NAME)
    embeddings = OpenAIEmbeddings()
    
    vectors = []
    
    for chunk in chunks:
        embedding = embeddings.embed_documents([chunk.page_content])[0]
        print(embedding)
        vectors.append({
            "id": str(chunk.metadata["id"]),
            "values": embedding,
            "metadata": chunk.metadata
        })
    
    try:
        index.upsert(vectors=vectors, namespace=namespace)
        print(f"✅ Documents added successfully under namespace: {namespace}")
    except Exception as e:
        print(f"❌ An error occurred while adding to Pinecone: {e}")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        chunk.metadata["text"] = chunk.page_content
        current_page_id = f"{source}:{page}"
        
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
    
    return chunks

def query_pinecone(index_name: str, query_text: str, namespace: str, top_k: int = 5):
    index = pc.Index(INDEX_NAME)
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_documents([query_text])[0]
    
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace=namespace)
    
    return [
        {
            "PDF Path": match.metadata.get("source", "Unknown"),
            "Page Number": match.metadata.get("page", "Unknown"),
            "Text Chunk": match.metadata.get("text", "Unknown"),
            "Relevance Score": match.score
        }
        for match in results.matches
    ]

def delete_by_source_from_list(index_name: str, source: str, namespace: str):
    index = pc.Index(INDEX_NAME)
    
    try:

        vector_ids=[]
        for id in index.list(prefix=source,namespace=namespace):
            vector_ids.extend(id)
        if vector_ids!=[]: 
            index.delete(ids=vector_ids, namespace=namespace)
            print(f"✅ Deleted vectors with source '{source}' from namespace: {namespace}")
        else:
            print(f"❌ No vectors found with source '{source}' to delete.")
    
    except Exception as e:
        print(f"❌ Error while listing or deleting vectors: {e}")


def list_vectors(index_name: str, user_gmail: str):
    namespace = user_gmail.replace("@", "_").replace(".", "_")  # Format namespace
    index = pc.Index(name=INDEX_NAME)

    try:
        vectors = index.list(namespace=namespace)
        print(f"📌 Vectors under namespace {namespace}:")
        for vector in vectors:
            print(vector)
    except Exception as e:
        print(f"❌ Error while listing vectors: {e}")

def populate_database(user_gmail):
    try:
        print("📄 Loading documents from the data folder...")
        documents = load_documents()
        print("📄 Recursive Splitting documents into chunks...")
        chunksR = split_documentsR(documents)
        print(f"📂 Adding chunks to Pinecone database (User: {user_gmail})...")
        chunks_with_ids = calculate_chunk_ids(chunksR)
        add_to_pinecone(chunks_with_ids, INDEX_NAME, user_gmail)
    except Exception as e:
        print(f"❌ An error occurred during database population: {e}")

def main():
    initialize_directories()
    print("Welcome to the Document Uploader for Pinecone DB!")
    user_gmail = input("Enter your email: ").strip().lower()
    user_input = input("Would you like to process the documents now? (yes/no): ").strip().lower()
    if user_input == "yes":
        populate_database(user_gmail)
    else:
        print("Exiting program. Add PDFs to 'data' and run again.")

# if __name__ == "__main__":
#     main()


print(pc.list_indexes())

# print(query_pinecone("q","what are the Disadvantages of a centralized organization","harsh",1))

