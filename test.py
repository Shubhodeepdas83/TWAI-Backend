from CleanedTest.help_with_ai import (
    handle_help_from_ai, query_ragR, llm_processing_query, citation_context_text,useWeb
)



# Function to get predefined AI instructions
def get_ai_instructions():
    """Returns AI assistant system instructions."""
    return {
        "query_extraction": (
            "You are an AI assistant optimized for generating precise, contextually relevant "
            "questions suitable for Retrieval-Augmented Generation (RAG) querying..."
        ),
        "answering_query": (
            "You are a helpful assistant. Your primary function is to answer queries based on the "
            "provided context..."
        ),
        "conversation_summary": "Summarize the conversation into a single line...",
        "fact_checking": "All your answers should be strictly grounded in the retrieved context..."
    }

# Function to get model parameters
def get_model_parameters():
    """Returns default parameters for AI model configuration."""
    return {
        "temperature": 0.7,
        "top_p": 0.9,
        "token_limit": 700,
        "chunk_limit": 3
    }

# Function to process AI assistance request
def process_ai_help(instruction, temp, top_p, token_limit, raw_Conversation, use_web):
    """Handles AI assistance request, including optional web search."""
    query = handle_help_from_ai(instruction, temp, top_p, token_limit, raw_Conversation)
    
    if not query:
        print("No query generated.")
        return
    
    print("Processing query...")
    query_results = query_ragR(query, get_model_parameters()["chunk_limit"])
    
    # Include web search results if enabled
    all_retrieved_documents = query_results
    if use_web:
        web_results = useWeb(query)
        all_retrieved_documents += web_results  

    print("Retrieved Documents:", all_retrieved_documents)
    
    if all_retrieved_documents:
        context_text, citation_map = citation_context_text(all_retrieved_documents)
        print("Context:", context_text)

        if context_text:
            result = llm_processing_query(context_text, query, instruction, temp, top_p, token_limit)
            print(f"Query: {query}\nResult: {result}")
        else:
            print("No results found.")

# Main function
def main():
    ai_instructions = get_ai_instructions()
    model_params = get_model_parameters()

    # Handle AI assistance with web search enabled
    process_ai_help(
        ai_instructions["query_extraction"],
        model_params["temperature"],
        model_params["top_p"],
        model_params["token_limit"],
        [{'user': 'What is the capital of India?'}],
        True
    )

if __name__ == "__main__":
    main()
