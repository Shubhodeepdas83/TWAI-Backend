import os
import time
from datetime import datetime
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

def log_time(stage):
    """Logs the timestamp for a given stage."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {stage}")

def llm_processing_evaluation(context_text, query, Cust_instr, temp, top_p, token_limit):
    log_time("Starting LLM Processing for Evaluation")
    messages = [
        {"role": "system", "content": f"THIS IS THE MOST IMPORTANT INSTRUCTION:{Cust_instr}"},
        {"role": "user", "content": (
            f"Given the extracted answer: {query}, and the retrieved context: {context_text}, evaluate the accuracy and alignment of the answer. "
            "Summarize your findings, highlight any discrepancies, and provide suggestions for refinement if necessary. "
            f"""Cite the source number at the end of each sentence or phrase that comes from that source using square brackets like [Number].
                If the information comes from multiple sources, cite all relevant source numbers.
                If the answer is not found in the sources, say \"I am sorry, but I cannot answer this question based on the provided information.\"

                Sources:
                {context_text}

                Question: {query}
                Answer:"""
            "ENSURE ALL ANSWERS ARE LESS THAN 100 WORDS"
            "THE ANSWER DOES NOT NEED TO BE IN SENTENCE FORMAT, IT CAN ALSO BE IN BULLETED POINTS FORMAT"
        )}
    ]
    result = llm_processing(messages, "gpt-4o-mini", temp, top_p, token_limit)
    log_time("Completed LLM Processing for Evaluation")
    return result

def llm_processing_FindAnswer(Cust_instr, temp, top_p, token_limit, raw_Conversation):
    log_time("Starting LLM Processing for Answer Extraction")
    relevant_conversation = extract_relevant_conversation(raw_Conversation)
    summarized_text = " ".join(text for _, text in relevant_conversation) if relevant_conversation else "No new conversation available to analyze."
    try:
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
        log_time("Completed LLM Processing for Answer Extraction")
        return chat_completion.choices[0].message.content
    except Exception as e:
        log_time("Error in Answer Extraction")
        return None

def process_fact_checking(instruction, temp, top_p, token_limit, raw_Conversation, use_web, namespace):
    log_time("Starting Fact Checking Process")
    query = llm_processing_FindAnswer(instruction, temp, top_p, token_limit, raw_Conversation)
    if not query:
        log_time("No valid summary generated")
        return {"error": "No valid summary generated"}
    
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
    
    context_text, citation_map = "", {}
    result = "No result found."
    used_citations = {}
    
    if all_retrieved_documents:
        log_time("Starting Context Text Generation")
        context_text, citation_map = citation_context_text(all_retrieved_documents)
        log_time("Completed Context Text Generation")
        
        if context_text:
            result = llm_processing_evaluation(
                context_text, 
                query, 
                instruction, 
                temp, 
                top_p, 
                token_limit
            )
            log_time("Extracting Citations")
            used_citations = extract_used_citations(result, citation_map, all_retrieved_documents)
            log_time("Completed Citation Extraction")
        else:
            log_time("No context found")
            return {"error": "No context found from the retrieved documents"}
    
    log_time("Completed Fact Checking Process")
    return {
        "query": "FACT CHECK:",
        "used_citations": used_citations,
        "result": result
    }

def FACT_CHECKING_HELP(raw_Conversation, use_web, userId):
    log_time("Starting FACT_CHECKING_HELP")
    instructions = get_system_instructions()
    model_params = get_model_parameters()
    response = process_fact_checking(
        instructions["fact_checking"],
        model_params["temperature"],
        model_params["top_p"],
        model_params["token_limit"],
        raw_Conversation,
        use_web,
        namespace=userId
    )
    log_time("Completed FACT_CHECKING_HELP")
    return response