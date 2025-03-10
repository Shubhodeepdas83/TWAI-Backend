import json
import os
import time  # Import time for timestamps
from datetime import datetime  # Import datetime for readable timestamps
from openai import OpenAI
from dotenv import load_dotenv
from .citations import extract_used_citations
from .prompts import get_system_instructions
from .commonFunctions import (
    extract_relevant_conversation,
    query_ragR,
    useWeb,
    citation_context_text,
    llm_processing,
    get_model_parameters
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

def handle_help_from_ai_text(custom_instructions, temperature, top_p, token_limit, raw_conversation, highlightedText):
    log_time("Starting Question Extraction")
    
    relevant_conversation = extract_relevant_conversation(raw_conversation)
    summarized_text = " ".join(text for _, text in relevant_conversation) if relevant_conversation else "No new conversation available to analyze."

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"You are an AI assistant... {custom_instructions}"},
                {"role": "user", "content": f"Based on the conversation provided, generate a query for RAG. Here's the main Important conversation snippet:\n\n{highlightedText}\n\nHere's the overall conversation :\n\n{summarized_text}\n\n. WHILE GENERATING THE QUERY, ENSURE THAT THE QUERY IS MAJORLY BASED ON THE HIGHLIGHTED TEXT WITH THE USE OF THE OVERALL CONVERSATION TO PROVIDE CONTEXT."}
            ],
            model="gpt-4o-mini", 
            temperature=temperature, 
            top_p=top_p, 
            max_tokens=token_limit
        )
        log_time("Completed Question Extraction")
        return chat_completion.choices[0].message.content
    except Exception as e:
        log_time("Error in Question Extraction text")
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


def HELP_WITH_AI(raw_conversation, use_web, userId):
    try :
        log_time("Starting AI Help Process")

        ai_instructions = get_system_instructions()
        model_params = get_model_parameters()

        instruction = ai_instructions["query_extraction"]
        instruction2 = ai_instructions["answering_query"]
        temperature = model_params["temperature"]
        top_p = model_params["top_p"]
        token_limit = model_params["token_limit"]
        namespace = userId
        
        query = handle_help_from_ai(instruction, temperature, top_p, token_limit, raw_conversation)
        if not query:
            pass
            # return {"error": "No query generated"}
        
        yield json.dumps({"query": query}) + "\n"
        
        log_time("Starting RAG Query")
        chunk_limit = get_model_parameters()["chunk_limit"]
        task = [query_ragR(query, chunk_limit, namespace)]
        log_time("Completed RAG Query")

        # all_retrieved_documents = query_results
        if use_web:
            log_time("Starting Web Search")
            task.append(useWeb(query))
            log_time("Completed Web Search")
            # all_retrieved_documents += web_results  
        results = task
        retrieved_docs = results[0]
        if use_web:
            retrieved_docs.extend(results[1])

        context_text, citation_map, result = "", {}, "No result found."
        used_citations = {}

        log_time("Starting Context Processing (Citation) for RAG and Web Search")
        context_text, citation_map = citation_context_text(retrieved_docs)
        log_time("Completed Context Processing (Citation) for RAG and Web Search")
        print(f"Context Text: {context_text}")
        print(f"Citation Map: {citation_map}")
        
        log_time("Starting LLM Response Generation")
        
        result = llm_processing_query(context_text, query, instruction2, temperature, top_p, token_limit)
        yield json.dumps({"result": result}) + "\n"
        log_time("Completed LLM Response Generation")
        
        log_time("Extracting Citations")
        used_citations = extract_used_citations(result, citation_map, retrieved_docs)

        yield  json.dumps({"used_citations": used_citations}) + "\n"
        log_time("Completed Citation Extraction")
        log_time("Completed AI Help Process")
        # return {
        #     "query": query,
        #     "used_citations": used_citations,
        #     "result": result
        # }
    except Exception as e:
        log_time("Error in AI Help Process")
        print(f"Error in HELP_WITH_AI: {e}")
        # return {"error": "An error occurred during AI help processing."}
        

def HELP_WITH_AI_text(raw_conversation, use_web, userId, highlightedText):
    try :
        log_time("Starting AI Help Process Text")

        ai_instructions = get_system_instructions()
        model_params = get_model_parameters()

        instruction = ai_instructions["query_extraction"]
        instruction2 = ai_instructions["answering_query"]
        temperature = model_params["temperature"]
        top_p = model_params["top_p"]
        token_limit = model_params["token_limit"]
        namespace = userId
        
        query = handle_help_from_ai_text(instruction, temperature, top_p, token_limit, raw_conversation, highlightedText)
        if not query:
            return {"error": "No query generated"}
        
        log_time("Starting RAG Query")
        chunk_limit = get_model_parameters()["chunk_limit"]
        task = [query_ragR(query, chunk_limit, namespace)]
        log_time("Completed RAG Query")

        # all_retrieved_documents = query_results
        if use_web:
            log_time("Starting Web Search")
            task.append(useWeb(query))
            log_time("Completed Web Search")
            # all_retrieved_documents += web_results  
        results = task
        retrieved_docs = results[0]
        if use_web:
            retrieved_docs.extend(results[1])

        context_text, citation_map, result = "", {}, "No result found."
        used_citations = {}

        log_time("Starting Context Processing (Citation) for RAG and Web Search")
        context_text, citation_map = citation_context_text(retrieved_docs)
        log_time("Completed Context Processing (Citation) for RAG and Web Search")
        print(f"Context Text: {context_text}")
        print(f"Citation Map: {citation_map}")
        
        log_time("Starting LLM Response Generation")
        
        result = llm_processing_query(context_text, query, instruction2, temperature, top_p, token_limit)
        log_time("Completed LLM Response Generation")
        log_time("Extracting Citations")
        used_citations = extract_used_citations(result, citation_map, retrieved_docs)
        log_time("Completed Citation Extraction")
        log_time("Completed AI Help Process")
        return {
            "query": query,
            "used_citations": used_citations,
            "result": result
        }
    except Exception as e:
        log_time("Error in AI Help Process")
        print(f"Error in HELP_WITH_AI: {e}")
        return {"error": "An error occurred during AI help processing."}