from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import sys
import pinecone
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()

# llm = OpenAI()
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = "us-east-1"
pc = Pinecone(PINECONE_API_KEY) 
index = pc.Index(name="new-4-3-25" )

loader = PyPDFDirectoryLoader(r"C:\Users\Lenovo\Desktop\repo\Intern-GEN AI\data\reftest.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
text_chunks = text_splitter.split_documents(data)

# docsearch = index.from_texts([t.page_content for t in text_chunks], embeddings)
from langchain_pinecone import PineconeVectorStore  
text_field = [t.page_content for t in text_chunks] 
vectorstore = PineconeVectorStore(  
    index, embeddings, text_field  
) 

# docsearch = Pinecone.from_existing_index(index_name, embeddings)


# while True:
#   user_input = input(f"Input Prompt: ")
#   if user_input == 'exit':
#     print('Exiting')
#     sys.exit()
#   if user_input == '':
#     continue
#   result = qa({'query': user_input})
#   print(f"Answer: {result['result']}")
#   docs = docsearch.similarity_search(query, k=3)
#   print(f"Docs: {docs}")