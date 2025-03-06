import os
import time  # Import time for timestamps
from datetime import datetime  # Import datetime for readable timestamps
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

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def log_time(stage):
    """Logs the timestamp for a given stage."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {stage}")


def handle_help_from_ai(custom_instructions, temperature, top_p, token_limit, raw_conversation):
    log_time("Starting Question Extraction")
    
    relevant_conversation = extract_relevant_conversation(raw_conversation)
    summarized_text = " ".join(text for _, text in relevant_conversation) if relevant_conversation else "No new conversation available to analyze."

    try:
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
        log_time("Completed Question Extraction")
        return chat_completion.choices[0].message.content
    except Exception as e:
        log_time("Error in Question Extraction")
        return None


def llm_processing_query(context_text, query, custom_instructions, temperature, top_p, token_limit):
    log_time("Starting LLM Processing")
    
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
    
    response = llm_processing(messages, "gpt-4o-mini", temperature, top_p, token_limit)
    log_time("Completed LLM Processing")
    return response


def process_ai_help(instruction, temperature, top_p, token_limit, raw_conversation, use_web, namespace):
    log_time("Starting AI Help Process")
    
    query = handle_help_from_ai(instruction, temperature, top_p, token_limit, raw_conversation)
    if not query:
        return {"error": "No query generated"}
    
    log_time("Starting RAG Query")
    chunk_limit = get_model_parameters()["chunk_limit"]
    query_results = query_ragR(query, chunk_limit, namespace)
    log_time("Completed RAG Query")

    all_retrieved_documents = query_results
    if use_web:
        log_time("Starting Web Search")
        web_results = useWeb(query)
        log_time("Completed Web Search")
        all_retrieved_documents += web_results  

    context_text, citation_map, result = "", {}, "No result found."
    used_citations = {}

    if all_retrieved_documents:
        log_time("Starting Context Text Generation")
        context_text, citation_map = citation_context_text(all_retrieved_documents)
        log_time("Completed Context Text Generation")

        if context_text:
            result = llm_processing_query(context_text, query, instruction, temperature, top_p, token_limit)
            log_time("Extracting Citations")
            used_citations = extract_used_citations(result, citation_map, all_retrieved_documents)
            log_time("Completed Citation Extraction")
        else:
            return {"error": "No context found from the retrieved documents"}

    log_time("Completed AI Help Process")
    
    return {
        "query": query,
        "used_citations": used_citations,
        "result": result
    }


def HELP_WITH_AI(raw_conversation, use_web, userId):
    log_time("Starting HELP_WITH_AI")
    
    ai_instructions = get_system_instructions()
    model_params = get_model_parameters()

    response = process_ai_help(
        ai_instructions["query_extraction"],
        model_params["temperature"],
        model_params["top_p"],
        model_params["token_limit"],
        raw_conversation,
        use_web,
        namespace=userId
    )

    log_time("Completed HELP_WITH_AI")
    return response
