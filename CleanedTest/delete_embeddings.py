from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import os


def load_env_variables():
    load_dotenv()
    return {
        "RCHROMA_PATH": os.getenv("RCHROMA_PATH"),
        "DATA_PATH": os.getenv("DATA_PATH"),
    }





def DELETE_EMBEDDINGS(pdf_url):
    env_vars = load_env_variables()

    db = Chroma(persist_directory= env_vars['RCHROMA_PATH'], embedding_function=OpenAIEmbeddings())

    db.delete(where={"source": pdf_url})

    db.persist()

    return {"response": True}