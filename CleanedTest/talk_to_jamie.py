import os
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
from CleanedTest.citations import extract_used_citations
from CleanedTest.commonFunctions import (
    citation_context_text, extract_relevant_conversation, llm_processing, query_ragR, useWeb
)

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to encode an image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to summarize an uploaded image
async def Image_Summ(uploaded_file):
    if uploaded_file:
        try:
            # Read uploaded image file
            content = await uploaded_file.read()
            with open("temp_image.png", "wb") as f:
                f.write(content)
            
            # Convert image to base64
            base64_image = encode_image("temp_image.png")
            
            # Request GPT model for image summarization
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
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in Image_Summ: {e}")
            return None

# Function to refine user query based on conversation context
def llm_processing_query_Jamie(query, raw_Conversation):
    try:
        relevant_conversation = extract_relevant_conversation(raw_Conversation)
        summarized_text = " ".join(text for _, text in relevant_conversation) if relevant_conversation else "No new conversation available."
        
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Refine the question to be self-contained and meaningful. Do not modify or add anything."},
                {"role": "user", "content": f"Conversation: {summarized_text}\n\nQuestion: {query}"}
            ],
            temperature=0.7, top_p=0.9, max_tokens=600
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error in llm_processing_query_Jamie: {e}")
        return None

# Function to process user queries and generate structured responses
def llm_processing_Jamie(query: str, context_text: str):
    messages = [
        {"role": "system", "content": "Provide concise, structured responses with bullet points and citations."},
        {"role": "user", "content": f"Sources:\n{context_text}\n\nQuestion: {query}\nAnswer:"}
    ]
    return llm_processing(messages, "gpt-4o-mini", 0.7, 0.9, 700)

import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Function to generate and return a graph as a Base64-encoded string
def graph_vis(user_query, user_context, user_response):
    try:
        # Request GPT to generate Python code for a graph
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "I am providing you with the context related to the query, answer to the query and the query itself. \n" 
                        f"Query : {user_query} , Context : {user_context}, Response:{user_response} \n"
                        "If the query requires any graph or chart for a better understanding or presentation of answer, \n" 
                        "please provide only the corresponding matplotlib Python code to generate the graph or chart. \n" 
                        "Give proper label for X and Y axis and any legends if needed\n"
                        "Respond with just the code, and nothing else. \n" 
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Provide only the python code to support the answer with a graph or chart. \n"
                    )
                }
            ]
        )
        
        # Extract the Python code for the plot from the response
        input_plot = response.choices[0].message.content.replace("```python", "").replace("plt.show()", "").replace("```", "")
        
        # Create the graph in memory (without saving to disk)
        fig, ax = plt.subplots()
        exec(input_plot, {"plt": plt, "ax": ax})
        
        # Convert the figure to a Base64-encoded PNG image
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()
        plt.close(fig)
        
        # Return the Base64 image string
        return base64_image
    except Exception as e:
        print(f"Error in graph_vis: {e}")
        return None


# Main chatbot function to process user input and generate responses
async def CHAT_WITH_JAMIE(user_input: str, use_web: bool = False, use_graph: bool = False, uploaded_file=None, raw_Conversation=[]):
    if not user_input:
        return {"query": "Talk to Jamie", "result": "Could not process - No user input provided"}
    
    try:
        # Process image if uploaded
        if uploaded_file:
            image_summary = await Image_Summ(uploaded_file)
            if image_summary:
                user_input += " Provided Image Context: " + image_summary
        
        # Generate refined query
        query = llm_processing_query_Jamie(user_input, raw_Conversation)
        if not query:
            return {"query": "Talk to Jamie", "result": "Could not process - No query generated"}
        
        # Retrieve relevant documents using RAG
        retrieved_docs = query_ragR(query, 3)
        if use_web:
            retrieved_docs.extend(useWeb(query))
        
        # Prepare context
        context_text, citation_map = citation_context_text(retrieved_docs)
        context_text = user_input + context_text
        
        # Generate response
        result = llm_processing_Jamie(query, context_text)
        if not result:
            return {"query": "Talk to Jamie", "result": "Could not process - No result generated"}
        
        # Extract citations
        used_citations = extract_used_citations(result, citation_map, retrieved_docs)
        
        # Generate graph if enabled
        graph_img = graph_vis(query, retrieved_docs, result) if use_graph else None
        
        return {
            "query": query,
            "result": result,
            "used_citations": used_citations,
            "graph": graph_img
        }
    except Exception as e:
        print(f"Error in CHAT_WITH_JAMIE: {e}")
        return {"query": "Talk to Jamie", "result": "Could not process - Error occurred"}
