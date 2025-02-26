import os
from openai import OpenAI
from dotenv import load_dotenv
from .citations import extract_used_citations
from .commonFunctions import (
    extract_relevant_conversation,
    query_ragR,
    useWeb,
    citation_context_text,
    llm_processing,
    get_model_parameters,
    get_system_instructions
)

# Load environment variables
load_dotenv()
RCHROMA_PATH = os.getenv("RCHROMA_PATH")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def handle_help_from_ai(custom_instructions, temperature, top_p, token_limit, raw_conversation):
    """
    Generates a specific question for RAG using the recent conversation.
    
    Args:
        custom_instructions (str): Custom system instructions for the AI
        temperature (float): Temperature parameter for response generation
        top_p (float): Top-p parameter for response generation
        token_limit (int): Maximum tokens for response generation
        raw_conversation (list): The conversation history
        
    Returns:
        str or None: Generated query for RAG, or None if an error occurs
    """
    # Extract relevant parts of the conversation
    relevant_conversation = extract_relevant_conversation(raw_conversation)
    summarized_text = " ".join(text for _, text in relevant_conversation) if relevant_conversation else "No new conversation available to analyze."
    
    try:
        # Generate query using the conversation summary
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"You are an AI assistant... {custom_instructions}"},
                {"role": "user", "content": f"Based on the conversation, generate a query for RAG:\n\n{summarized_text}"}
            ],
            model="gpt-4o-mini", 
            temperature=temperature, 
            top_p=top_p, 
            max_tokens=token_limit
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return None


def llm_processing_query(context_text, query, custom_instructions, temperature, top_p, token_limit):
    """
    Generate a response using query results and provided context.
    
    Args:
        context_text (str): Context information from retrieval
        query (str): User query
        custom_instructions (str): Custom system instructions
        temperature (float): Temperature parameter
        top_p (float): Top-p parameter
        token_limit (int): Maximum token limit
        
    Returns:
        str: Generated response with citations
    """
    messages = [
        {
            "role": "system",
            "content": f"Important Instruction: {custom_instructions}"
        },
        {
            "role": "user",
            "content": (
                f"""You are a helpful chatbot. Please answer the user's question based on the provided sources.
                    Cite the source number at the end of each sentence or phrase that comes from that source using square brackets like [Number].
                    If the information comes from multiple sources, cite all relevant source numbers.
                    If the answer is not found in the sources, say "I am sorry, but I cannot answer this question based on the provided information."

                    Sources:
                    {context_text}

                    Question: {query}
                    Answer:"""
            )
        }
    ]
    return llm_processing(messages, "gpt-4o-mini", temperature, top_p, token_limit)


def process_ai_help(instruction, temperature, top_p, token_limit, raw_conversation, use_web):
    """
    Handles AI assistance request, including optional web search.
    
    Args:
        instruction (str): System instructions for the AI
        temperature (float): Temperature parameter
        top_p (float): Top-p parameter
        token_limit (int): Maximum token limit
        raw_conversation (list): Conversation history
        use_web (bool): Whether to include web search results
        
    Returns:
        dict: Response containing query, result, and used citations
    """
    # Generate query from conversation
    query = handle_help_from_ai(instruction, temperature, top_p, token_limit, raw_conversation)
    
    if not query:
        return {"error": "No query generated"}
    
    # Retrieve documents based on query
    chunk_limit = get_model_parameters()["chunk_limit"]
    query_results = query_ragR(query, chunk_limit)
    
    # Include web search results if enabled
    all_retrieved_documents = query_results
    if use_web:
        web_results = useWeb(query)
        all_retrieved_documents += web_results  

    # Initialize default values
    context_text, citation_map, result = "", {}, "No result found."
    used_citations = {}

    if all_retrieved_documents:
        # Generate context text and citation mapping
        context_text, citation_map = citation_context_text(all_retrieved_documents)

        if context_text:
            # Process query with context
            result = llm_processing_query(context_text, query, instruction, temperature, top_p, token_limit)
            used_citations = extract_used_citations(result, citation_map, all_retrieved_documents)
        else:
            return {"error": "No context found from the retrieved documents"}
    
    # Return response in JSON format
    return {
        "query": query,
        "used_citations": used_citations,
        "result": result
    }


def HELP_WITH_AI(raw_conversation, use_web):
    """
    Main function to provide AI assistance with optional web search.
    
    Args:
        raw_conversation (list): Full conversation history
        use_web (bool): Whether to include web search results
        
    Returns:
        dict: JSON response with query, result, and used citations
    """
    # Get system instructions and model parameters
    ai_instructions = get_system_instructions()
    model_params = get_model_parameters()

    # Process AI help request
    response = process_ai_help(
        ai_instructions["query_extraction"],
        model_params["temperature"],
        model_params["top_p"],
        model_params["token_limit"],
        raw_conversation,
        use_web
    )

    return response