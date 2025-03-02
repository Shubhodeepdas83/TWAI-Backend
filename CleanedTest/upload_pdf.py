import hashlib
import os
import uuid
import requests
from io import BytesIO
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
pc = PineconeGRPC(api_key=os.getenv("PINECONE_API_KEY"))
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")
os.environ['OPENAI_API_KEY'] = api_key

def load_env_variables():
    load_dotenv()
    return {
        "DATA_PATH": os.getenv("DATA_PATH"),
    }

# def add_to_chroma(chunks, chroma_path):
#     db = Chroma(persist_directory=chroma_path, embedding_function=OpenAIEmbeddings())
#     chunks_with_ids = calculate_chunk_ids(chunks)
#     existing_items = db.get(include=[])
#     existing_ids = set(existing_items["ids"])
#     new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    
#     if new_chunks:
#         new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
#         db.add_documents(new_chunks, ids=new_chunk_ids)
#         db.persist()
#         print("✅ Documents added successfully!")
#     else:
#         print("✅ No new documents to add")


def add_to_pinecone(chunks, namespace):

    index = pc.Index(name=INDEX_NAME)
    embeddings = OpenAIEmbeddings()

    chunks_with_ids = calculate_chunk_ids(chunks)
    
    vectors = []
    
    for chunk in chunks_with_ids:
        embedding = embeddings.embed_documents([chunk.page_content])[0]
        vectors.append({
            "id": str(chunk.metadata["id"]),
            "values": embedding,
            "metadata": chunk.metadata
        })



    index.upsert(vectors=vectors, namespace=namespace)
    print(f"✅ Documents added successfully under namespace: {namespace}")




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

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def ADD_EMBEDDINGS_FROM_S3(pdf_url,userId):
    env_vars = load_env_variables()

    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception("Failed to fetch PDF from the URL.")
    
    pdf_content = BytesIO(response.content)

    # Generate a valid filename based on the S3 URL hash
    temp_pdf_path = os.path.join(env_vars["DATA_PATH"], f'{str(uuid.uuid1())}.pdf')

    # Save the file
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_content.read())

    document_loader = PyPDFDirectoryLoader(env_vars['DATA_PATH'])
    documents = document_loader.load()
    for doc in documents:
        doc.metadata["source"] = pdf_url

    chunks = split_documents(documents)
    add_to_pinecone(chunks, userId)

    os.remove(temp_pdf_path)

    return {"response": True}


