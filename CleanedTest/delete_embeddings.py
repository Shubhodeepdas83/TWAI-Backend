from dotenv import load_dotenv
import os
from pinecone.grpc import PineconeGRPC
load_dotenv()

HOST = os.getenv("PINECONE_HOST")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
pc = PineconeGRPC(api_key=os.getenv("PINECONE_API_KEY"))


def load_env_variables():
    load_dotenv()
    return {
        "DATA_PATH": os.getenv("DATA_PATH"),
    }



    


def DELETE_EMBEDDINGS(pdf_url,userId):
    index = pc.Index(name=INDEX_NAME)
    
    try:

        vector_ids=[]
        for id in index.list(prefix=pdf_url,namespace=userId):
            vector_ids.extend(id)
        if vector_ids!=[]: 
            index.delete(ids=vector_ids, namespace=userId)
            print(f"✅ Deleted vectors with source '{pdf_url}' from namespace: {userId}")
        else:
            print(f"❌ No vectors found with source '{pdf_url}' to delete.")
    
    except Exception as e:
        print(f"❌ Error while listing or deleting vectors: {e}")


    return {"response": True}