import os
import base64
import time
import asyncio
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from CleanedTest.citations import extract_used_citations
from CleanedTest.commonFunctions import (
    citation_context_text, extract_relevant_conversation, llm_processing, query_ragR, useWeb, get_model_parameters
)

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def log_time(stage):
    """Logs the timestamp for a given stage."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {stage}")

async def async_query_ragR(query, chunk_limit, namespace):
    print("----------------Using RAG-----------------------")
    return await asyncio.to_thread(query_ragR, query, chunk_limit, namespace=namespace)

async def async_useWeb(query):
    print("----------------Using Web-----------------------")
    return await asyncio.to_thread(useWeb, query)

# Function to encode an image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to summarize an uploaded image
async def Image_Summ(uploaded_file):
    if uploaded_file:
        try:
            log_time("Starting Image Summarization")
            content = await uploaded_file.read()
            with open("temp_image.png", "wb") as f:
                f.write(content)
            base64_image = encode_image("temp_image.png")
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize the uploaded image in detail"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }]
            )
            log_time("Completed Image Summarization")
            return response.choices[0].message.content
        except Exception as e:
            log_time("Error in Image Summarization")
            print(f"Error in Image_Summ: {e}")
            return None

# Function to refine user query based on conversation context
def llm_processing_query_Jamie(query, raw_Conversation):
    try:
        log_time("Starting Query Refinement")
        relevant_conversation = extract_relevant_conversation(raw_Conversation)
        summarized_text = " ".join(text for _, text in relevant_conversation) if relevant_conversation else "No new conversation available."
        
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Refine the question to be self-contained and meaningful."},
                {"role": "user", "content": f"Conversation: {summarized_text}\n\nQuestion: {query}"}
            ],
            temperature=0.7, top_p=0.9, max_tokens=600
        )
        log_time("Completed Query Refinement")
        return chat_completion.choices[0].message.content
    except Exception as e:
        log_time("Error in Query Refinement")
        print(f"Error in llm_processing_query_Jamie: {e}")
        return None

# Function to process user queries and generate structured responses
def llm_processing_Jamie(query: str, context_text: str):
    messages = [
        {"role": "system", "content": "Provide concise, structured responses with bullet points and citations.Cite the source number at the end of each sentence or phrase that comes from that source using square brackets like [Number]. If the information comes from multiple sources, cite all relevant source numbers. If the answer is not found in the sources, say 'i am sorry, but I cannot answer this question based on the provided information.'"},
        {"role": "user", "content": f"Sources:\n{context_text}\n\nQuestion: {query}\nAnswer:"}
    ]
    return llm_processing(messages, "gpt-4o-mini", 0.7, 0.9, 700)

# Function to generate and return a graph as a Base64-encoded string
def graph_vis(user_query, user_context, user_response):
    try:
        log_time("Starting Graph Generation")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Provide Python code to visualize the query: {user_query}, Context: {user_context}, Response: {user_response}"}
            ]
        )
        input_plot = response.choices[0].message.content.replace("```python", "").replace("plt.show()", "").replace("```", "")
        
        fig, ax = plt.subplots()
        exec(input_plot, {"plt": plt, "ax": ax})
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()
        plt.close(fig)
        log_time("Completed Graph Generation")
        return base64_image
    except Exception as e:
        log_time("Error in Graph Generation")
        print(f"Error in graph_vis: {e}")
        return None

# Main chatbot function to process user input and generate responses
async def CHAT_WITH_JAMIE(userId, user_input: str, use_web: bool = False, use_graph: bool = False, uploaded_file=None, raw_Conversation=[]):
    if not user_input:
        return {"query": "Talk to Jamie", "result": "No user input provided"}
    
    try:
        log_time("Starting CHAT_WITH_JAMIE")
        if uploaded_file:
            image_summary = await Image_Summ(uploaded_file)
            if image_summary:
                user_input += f" Provided Image Context: {image_summary}"
        
        query = llm_processing_query_Jamie(user_input, raw_Conversation)
        if not query:
            return {"query": "Talk to Jamie", "result": "No query generated"}
        
        log_time("Starting Parallel Execution for RAG and Web Search")
        tasks = [async_query_ragR(query, get_model_parameters()["chunk_limit"], namespace=userId)]
        if use_web:
            tasks.append(async_useWeb(query))
        
        results = await asyncio.gather(*tasks)
        
        retrieved_docs = results[0]
        if use_web:
            retrieved_docs.extend(results[1])
        
        log_time("Completed Parallel Execution for RAG and Web Search")
        
        log_time("Starting Context Processing (Citation) for RAG and Web Search")
        context_text, citation_map = citation_context_text(retrieved_docs)
        context_text = user_input + context_text
        log_time("Completed Context Processing (Citation) for RAG and Web Search")

        print(f"Context Text: {context_text}")
        print(f"Citation Map: {citation_map}")
        
        log_time("Starting LLM Response Generation")
        result = llm_processing_Jamie(query, context_text)

        log_time("Completed LLM Response Generation")
        
        log_time("Extracting Citations")
        used_citations = extract_used_citations(result, citation_map, retrieved_docs)
        log_time("Completed Citation Extraction")
        
        graph_img = graph_vis(query, retrieved_docs, result) if use_graph else None
        log_time("Completed CHAT_WITH_JAMIE")
        
        return {"query": query, "result": result, "used_citations": used_citations, "graph": graph_img}
    except Exception as e:
        log_time("Error in CHAT_WITH_JAMIE")
        print(f"Error in CHAT_WITH_JAMIE: {e}")
        return {"query": "Talk to Jamie", "result": "Error occurred"}
