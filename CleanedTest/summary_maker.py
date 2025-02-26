import os
from openai import OpenAI
from dotenv import load_dotenv
from .commonFunctions import extract_relevant_conversation

# Load environment variables
load_dotenv()
RCHROMA_PATH = os.getenv("RCHROMA_PATH")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def Summary_llm(combined_text):
    """
    Summarizes the conversation using OpenAI API.
    
    Args:
        combined_text (str): The text of the conversation to summarize
        
    Returns:
        str or None: Summarized conversation in bullet points, or None if an error occurs
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Summarize the given conversation of the meet."},
                {"role": "user", "content": f"Conversation: {combined_text}\n\nSummarize the conversation keeping all the important points. Try to focus only on the most important and relevant points. Dont need to mention every small detail. Answer in specific bullet points with each bullet point not exceeding 15 words, at maximum"}
            ],
            model="gpt-4o-mini", 
            temperature=0.7, 
            top_p=0.9, 
            max_tokens=600
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return None


def summarize_conversation(raw_Conversation):
    """
    Extracts and summarizes relevant parts of a conversation.
    
    Args:
        raw_Conversation (list): The full conversation history
        
    Returns:
        dict: A dictionary containing the query and summarized result
    """
    # Filter relevant conversation based on specific speakers
    relevant_conversation = extract_relevant_conversation(raw_Conversation)
    
    if relevant_conversation:
        # Combine only the text (not speaker labels) into a single string
        combined_text = " ".join([text for _, text in relevant_conversation])
    else:
        combined_text = "No new conversation available to analyze."
    
    # Summarize the combined text using the summary model
    response = Summary_llm(combined_text)
    
    return {
        "query": "SUMMARY :",
        "result": response,
    }


def SUMMARY_WITH_AI(raw_Conversation):
    """
    Main function to generate a summary of a conversation.
    
    Args:
        raw_Conversation (list): The full conversation history
        
    Returns:
        dict: JSON response with the query and summarized result
    """
    return summarize_conversation(raw_Conversation)