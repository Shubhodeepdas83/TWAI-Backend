import os
from openai import OpenAI
from dotenv import load_dotenv
from CleanedTest.citations import extract_used_citations
from .commonFunctions import (
    extract_relevant_conversation,
    query_ragR,
    useWeb,
    citation_context_text,
    llm_processing,
    get_system_instructions,
    get_model_parameters
)

# Load environment variables
load_dotenv()


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def llm_processing_evaluation(context_text, query, Cust_instr, temp, top_p, token_limit):
    """
    Evaluate an answer based on retrieved context.
    
    Args:
        context_text (str): The retrieved context information
        query (str): The question or query being evaluated
        Cust_instr (str): Custom instruction for the evaluation
        temp (float): Temperature parameter for LLM
        top_p (float): Top-p sampling parameter for LLM
        token_limit (int): Maximum tokens to generate
        
    Returns:
        str: Evaluation result from the LLM
    """
    messages = [
        {
            "role": "system",
            "content": f"THIS IS THE MOST IMPORTANT INSTRUCTION:{Cust_instr}"
        },
        {
            "role": "user",
            "content": (
                f"Given the extracted answer: {query}, and the retrieved context: {context_text}, "
                "evaluate the accuracy and alignment of the answer. Summarize your findings, highlight any discrepancies, "
                "and provide suggestions for refinement if necessary."
                f"""Cite the source number at the end of each sentence or phrase that comes from that source using square brackets like [Number].
                    If the information comes from multiple sources, cite all relevant source numbers.
                    If the answer is not found in the sources, say "I am sorry, but I cannot answer this question based on the provided information."

                    Sources:
                    {context_text}

                    Question: {query}
                    Answer:"""
                "ENSURE ALL ANSWERS ARE LESS THAN 100 WORDS"
                "THE ANSWER DOES NOT NEED TO BE IN SENTENCE FORMAT, IT CAN ALSO BE IN BULLETED POINTS FORMAT"
            )
        }
    ]
    return llm_processing(messages, "gpt-4o-mini", temp, top_p, token_limit)


def llm_processing_FindAnswer(Cust_instr, temp, top_p, token_limit, raw_Conversation):
    """
    Extracts relevant content for evaluation based on the recent conversation.
    
    Args:
        Cust_instr (str): Custom instruction for content extraction
        temp (float): Temperature parameter for LLM
        top_p (float): Top-p sampling parameter for LLM
        token_limit (int): Maximum tokens to generate
        raw_Conversation (list): List of conversation dictionaries
        
    Returns:
        str or None: Extracted content or None if error occurs
    """
    # Format conversation for processing
    relevant_conversation = extract_relevant_conversation(raw_Conversation)
    summarized_text = " ".join(text for _, text in relevant_conversation) if relevant_conversation else "No new conversation available to analyze."

    try:
        # Generate summarized content using LLM
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"{Cust_instr}"},
                {"role": "user", "content": f"Extract the final exchange:\n\n{summarized_text}"}
            ],
            model="gpt-4o-mini", 
            temperature=temp, 
            top_p=top_p, 
            max_tokens=token_limit
        )
        answer = chat_completion.choices[0].message.content
        return answer
    except Exception as e:
        print(f"Error generating question: {e}")
        return None


def process_fact_checking(instruction, temp, top_p, token_limit, raw_Conversation, use_web,namespace):
    """
    Handles the fact-checking workflow, including optional web search.
    
    Args:
        instruction (str): System instruction for fact checking
        temp (float): Temperature parameter for LLM
        top_p (float): Top-p sampling parameter for LLM
        token_limit (int): Maximum tokens to generate
        raw_Conversation (list): List of conversation dictionaries
        use_web (bool): Whether to include web search results
        
    Returns:
        dict: JSON response with query, citations, and result
    """
    # Step 1: Extract relevant content for evaluation
    query = llm_processing_FindAnswer(instruction, temp, top_p, token_limit, raw_Conversation)
    
    if not query:
        print("No valid summary generated.")
        return {"error": "No valid summary generated"}
    
    print("Evaluating content...")
    
    # Step 2: Retrieve documents based on the query
    chunk_limit = get_model_parameters()["chunk_limit"]
    query_results = query_ragR(query, chunk_limit, namespace)
    
    # Step 3: Include web search results if enabled
    all_retrieved_documents = query_results
    if use_web:
        web_results = useWeb(query)
        all_retrieved_documents += web_results  
    
    # Step 4: Initialize result variables with default values
    context_text, citation_map = "", {}
    result = "No result found."
    used_citations = {}
    
    # Step 5: Process the documents and extract context if available
    if all_retrieved_documents:
        context_text, citation_map = citation_context_text(all_retrieved_documents)

        # Step 6: Process the result if context is available
        if context_text:
            result = llm_processing_evaluation(
                context_text, 
                query, 
                instruction, 
                temp, 
                top_p, 
                token_limit
            )
            used_citations = extract_used_citations(result, citation_map, all_retrieved_documents)
        else:
            print("No context found.")
            return {"error": "No context found from the retrieved documents"}
    
    # Step 7: Return all relevant information in JSON format
    response = {
        "query": "FACT CHECK :",
        "used_citations": used_citations,
        "result": result
    }

    return response


def FACT_CHECKING_HELP(raw_Conversation, use_web,userId):
    """
    Main function to process fact checking with appropriate parameters.
    
    Args:
        raw_Conversation (list): List of conversation dictionaries
        use_web (bool): Whether to include web search results
        
    Returns:
        dict: JSON response with fact checking results
    """
    # Get system instructions and model parameters
    instructions = get_system_instructions()
    model_params = get_model_parameters()

    # Process fact checking with the appropriate parameters
    response = process_fact_checking(
        instructions["fact_checking"],
        model_params["temperature"],
        model_params["top_p"],
        model_params["token_limit"],
        raw_Conversation,
        use_web,
        namespace=userId
    )

    return response