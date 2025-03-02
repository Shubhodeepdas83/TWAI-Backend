import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import GoogleSerperAPIWrapper
from openai import OpenAI
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC


# Load environment variables
load_dotenv()
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
pc = PineconeGRPC(api_key=os.getenv("PINECONE_API_KEY"))

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")
os.environ['OPENAI_API_KEY'] = api_key


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_model_parameters():
    """
    Returns default model parameters for LLM interaction.
    
    Returns:
        dict: Dictionary containing temperature, top_p, token limit, and chunk limit
    """
    return {
        "temperature": 0.7,
        "top_p": 0.9,
        "token_limit": 700,
        "chunk_limit": 3
    }


def get_system_instructions():
    """
    Returns system instructions for different AI functionalities.
    
    Returns:
        dict: Dictionary containing prompt templates for various AI tasks
    """
    return {
        "query_extraction": (
            "You are an AI assistant optimized for generating precise, "
            "contextually relevant questions suitable for Retrieval-Augmented Generation (RAG) querying. "
            "Analyze the provided conversation, prioritize the lastest exchanges while using earlier messages for context, "
            "and extract a focused query. The question should be factual, specific, and well-suited for retrieving detailed information. "
            "Avoid ambiguous or conversational phrasing, and keep the query strictly relevant to the discussion topic."
        ),
        "answering_query": (
            "You are a helpful assistant. Your primary function is to answer queries based on the provided context. "
            "If the query is unrelated to the context, respond with: 'I don't know.' "
            "If the query is related to the context, provide a detailed response by combining the provided context with your internal knowledge. "
            "Ensure that your response is contextually aligned and does not contradict the information provided in the context. "
            "When using internal knowledge, make it complementary to the context to enhance the accuracy and usefulness of your response. "
            "Always prioritize context-based information, and only supplement it with internal knowledge if it enriches the answer while staying consistent with the context."
            "Additionally, ensure that the response is accurate, informative, and relevant to the user's query."
            "IMPORTANT NOTE: STRICTLY ALWAYS GIVE SUMMARIZED ANSWERS IN LESS THAN 6 LINES SO THAT THE USER READABILITY IS EASY. "
        ),
        "conversation_summary": (
            "Summarize the conversation into 1 line. If the provided "
            "conversation is huge then give high priority to the recent exchanges. "
            "But summarize the whole conversation into a single line keeping all important "
            "points and ignore who spoke what. i just need the context of what is spoken"
        ),
        "fact_checking": (
            "All your answers should be completely grounded **only** from the retrieved context. "
            "Under **no circumstances** use your internal knowledge base. "
            "Evaluate the sentence provided by comparing it against the retrieved context. "
            "Provide an analysis of its accuracy, completeness, and alignment with the context. "
            "Generate a summarized report that includes: "
            "a 2 liner answer whether the answer is factually correct or not. "
            "IF THE ANSWER IS FACTUALLY INCORRECT, MENTION THE RELEVANT CORRECT ANSWER. "
            "Only display the summary report nothing else. "
            "ENSURE THE COMPLETE ANSWER IS STRICTLY LESS THAN 40 WORDS. "
            "THE ANSWER DOES NOT NEED TO BE IN SENTENCE FORMAT, IT CAN ALSO BE IN BULLETED POINTS FORMAT"
        )
    }


def extract_relevant_conversation(raw_Conversation):
    """
    Extracts the conversation for Person 1, Person 2, and User after the last separator.
    
    Args:
        raw_Conversation (list): List of conversation dictionaries
        
    Returns:
        list: Formatted list of tuples containing speaker and message
    """
    converted = []
    try:
        for item in raw_Conversation:
            for key, value in item.items():
                # Format the speaker name
                formatted_key = "User" if key.lower() == "user" else "Person 1"
                converted.append((formatted_key, value.capitalize()))
        return converted
    except Exception:
        # Return empty list if conversion fails
        return []

def query_ragR(query_text: str, chunk_size: int, namespace: str):
    """
    Wrapper function to query the R chroma database.

    Args:
        query_text (str): The query to search for
        chunk_size (int): Number of chunks to retrieve
        namespace (str): The namespace for ChromaDB retrieval

    Returns:
        list: Retrieved documents with metadata
    """
    try:
        index = pc.Index(name=INDEX_NAME)

        embeddings = OpenAIEmbeddings()
        query_embedding = embeddings.embed_documents([query_text])[0]

        results = index.query(vector=query_embedding, top_k=chunk_size, include_metadata=True, namespace=namespace)

        retrieved_documents = []
        for match in results.matches:
            page_content = match.metadata.get("text", "").strip()

            if not page_content:  # Skip empty documents
                continue

            retrieved_documents.append({
                "page_content": page_content,
                "metadata": {
                    "PDF Path": match.metadata.get("source", "Unknown"),
                    "Page Number": match.metadata.get("page", "Unknown"),
                    "Relevance Score": match.score
                }
            })


        return retrieved_documents
    except Exception as e:
        print(f"Error querying RAG: {e}")
        return []

# def query_rag(persist_path: str, query_text: str, chunk_size: int):
#     """
#     Generic function to query RAG with Chroma.
    
#     Args:
#         persist_path (str): Path to the persisted Chroma database
#         query_text (str): The query to search for
#         chunk_size (int): Number of chunks to retrieve
        
#     Returns:
#         list: Retrieved documents with metadata and relevance scores
#     """
#     # Initialize Chroma vector database with OpenAI embeddings
#     db = Chroma(persist_directory=persist_path, embedding_function=OpenAIEmbeddings())
    
#     # Perform similarity search with relevance scores
#     results = db.similarity_search_with_relevance_scores(query_text, k=chunk_size)
    
#     # Format the results with metadata
#     return [
#         {
#             "PDF Path": doc.metadata.get("source", "Unknown"),
#             "Page Number": doc.metadata.get("page", "Unknown"),
#             "Text Chunk": doc.metadata.get("text","Unknown"),
#             "Relevance Score": score
#         }
#         for doc, score in results
#     ]


def useWeb(query: str):
    """
    Use Google Serper API to fetch web results.
    
    Args:
        query (str): The search query
        
    Returns:
        list: Web search results formatted as documents
    """
    search = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))
    results = search.results(query=query)
    
    # Extract and format organic search results
    web_data = [
        {"page_content": item["snippet"],"metadata": {"PDF Path": item["link"]}}for item in results.get("organic", [])
    ]
    return web_data


def citation_context_text(all_retrieved_documents):
    """
    Create context text with citation markers for retrieved documents.
    
    Args:
        all_retrieved_documents (list): List of document dictionaries
        
    Returns:
        tuple: (formatted context text, citation map dictionary)
    """
    context_text = ""
    citation_map = {}  # Maps citation number to source description
    citation_count = 1

    # Process each document and add to context with citation markers
    for doc in all_retrieved_documents:
        
        # Correct metadata extraction
        metadata = doc.get("metadata", {})
        source_name = metadata.get("PDF Path", "Unknown Source")
        page_info = f", Page {metadata.get('Page Number', 'Unknown')}" if "Page Number" in metadata else ""
        doc_text = doc.get("page_content", "")

        # Create source description and add to context
        source_description = f"{source_name}{page_info}"
        context_text += f"\n---\nSource {citation_count}: {source_description}\n---\n{doc_text}"
        
        # Update citation map
        citation_map[citation_count] = source_description
        citation_count += 1
        
    return context_text, citation_map

def llm_processing(
    query_context: str, model: str, temp: float, top_p: float, token_limit: int
):
    """
    Process queries with OpenAI's LLM.
    
    Args:
        query_context (str): The formatted query context
        model (str): OpenAI model to use
        temp (float): Temperature parameter
        top_p (float): Top-p sampling parameter
        token_limit (int): Maximum tokens to generate
        
    Returns:
        str or None: Generated response or None if an error occurs
    """
    try:
        # Create chat completion with the specified parameters
        chat_completion = client.chat.completions.create(
            model=model,
            messages=query_context,
            temperature=temp,
            top_p=top_p,
            max_tokens=token_limit
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error processing LLM query: {e}")
        return None