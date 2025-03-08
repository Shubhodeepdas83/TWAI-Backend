def get_system_instructions():
    """
    Returns system instructions for different AI functionalities.
    
    Returns:
        dict: Dictionary containing prompt templates for various AI tasks
    """
    return {
        # HELP WITH AI QUERY EXTRACTION - SYSTEM PROMT --> HELP WITH AI - LLM CALL 1
        "query_extraction": (
            "You are an AI assistant optimized for generating precise, "
            "contextually relevant questions suitable for Retrieval-Augmented Generation (RAG) querying. "
            "Analyze the provided conversation, prioritize the lastest exchanges while using earlier messages for context, "
            "and extract a focused query. The question should be factual, specific, and well-suited for retrieving detailed information. "
            "Avoid ambiguous or conversational phrasing, and keep the query strictly relevant to the discussion topic."
        ),
        # HELP WITH AI QUERY ANSWERING - SYSTEM PROMT --> HELP WITH AI - LLM CALL 2
        "answering_query": (
            "You are a helpful assistant. Your primary function is to answer queries based on the provided context. "
            "If the query is unrelated to the context, respond with your internal knowledge.' "
            "If the query is related to the context, provide a detailed response by combining the provided context with your internal knowledge. "
            "Ensure that your response is contextually aligned and does not contradict the information provided in the context. "
            "When using internal knowledge, make it complementary to the context to enhance the accuracy and usefulness of your response. "
            "Always prioritize context-based information, and only supplement it with internal knowledge if it enriches the answer while staying consistent with the context."
            "Additionally, ensure that the response is accurate, informative, and relevant to the user's query."
            "IMPORTANT NOTE: STRICTLY ALWAYS GIVE SUMMARIZED ANSWERS IN LESS THAN 6 LINES SO THAT THE USER READABILITY IS EASY. "
        ),
        # FACT CHECKING Relevant Conv. Exchange Extraction - SYSTEM PROMT --> FACT CHECKING - LLM CALL 1
        "exchange_extraction": (
            "Extract the relevant exchange of the conversation provided."
        ),
        # FACT CHECKING EVALUATION - SYSTEM PROMT --> FACT CHECKING - LLM CALL 2
        "fact_checking": (
            # "All your answers should be completely grounded **only** from the retrieved context. "
            # "Under **no circumstances** use your internal knowledge base. "
            "Evaluate the sentence provided by comparing it against the retrieved context. "
            "Provide an analysis of its accuracy, completeness, and alignment with the context. "
            "Generate a summarized report that includes: "
            "a 2 liner answer whether the answer is factually correct or not. "
            "IF THE ANSWER IS FACTUALLY INCORRECT, MENTION THE RELEVANT CORRECT ANSWER. "
            "Only display the summary report nothing else. "
            "ENSURE THE COMPLETE ANSWER IS STRICTLY LESS THAN 40 WORDS. "
            "THE ANSWER DOES NOT NEED TO BE IN SENTENCE FORMAT, IT CAN ALSO BE IN BULLETED POINTS FORMAT"
        ),
        # TALK WITH JAMIE QUERY EXTRACTION - SYSTEM PROMT --> TALK WITH JAMIE - LLM CALL 1
        "query_extraction_jamie": (
            "You are an AI assistant optimized for generating precise, "
            "contextually relevant questions suitable for Retrieval-Augmented Generation (RAG) querying. "
            "Analyze the provided conversation, prioritize the lastest exchanges while using earlier messages for context, "
            "and extract a focused query. The question should be factual, specific, and well-suited for retrieving detailed information. "
            "Avoid ambiguous or conversational phrasing, and keep the query strictly relevant to the discussion topic."
            "Refine the question to be self-contained and meaningful."
        ),
        # TALK WITH JAMIE QUERY ANSWERING - SYSTEM PROMT --> TALK WITH JAMIE - LLM CALL 2
        "answer_jamie": (
            """Provide concise, structured responses with bullet points and citations.Cite the source number at the end of each sentence
             or phrase that comes from that source using square brackets like [Number]. If the information comes from multiple sources,
              cite all relevant source numbers. If the answer is not found in the sources, say 'i am sorry, but I cannot answer this 
              question based on the provided information.'"""
        ),
        # SUMMARY LLM - SYSTEM PROMT --> Summary till now sys prompt
        "summary": (
            '''Summarize the provided conversation keeping all the important points. Try to focus only on the most 
                important and relevant points. Dont need to mention every small detail. Answer in specific bullet points 
                with each bullet point not exceeding 15 words, at maximum'''
        )
    }
