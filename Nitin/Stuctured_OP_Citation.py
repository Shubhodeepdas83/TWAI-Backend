from pydantic import BaseModel
from typing import List
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CitationSegment(BaseModel):
    text: str
    cited_sources: List[int]

class EvaluationOutput(BaseModel):
    segments: List[CitationSegment]
    final_summary: str

def generate_evaluation(model, custom_instructions, context_text, query, temp, top_p, token_limit):
    messages = [
        {
            "role": "system",
            "content": f"Important Instruction: {custom_instructions}"
        },
        {
            "role": "user",
            "content": (
                f"""You are a helpful chatbot. Please answer the user's question based on the provided sources.
                    Cite the source number for each sentence or phrase that comes from that source.
                    If the information comes from multiple sources, cite all relevant source numbers.
                    If the answer is not found in the sources, say "I am sorry, but I cannot answer this question based on the provided information."

                    Sources:
                    {context_text}

                    Question: {query}
                    Answer:"""
            )
        }
    ]

    chat_completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=temp,
        top_p=top_p,
        max_tokens=token_limit,
        response_format=EvaluationOutput
    )

    evaluation_result = chat_completion.choices[0].message.parsed

    print("Got evaluation result :", evaluation_result)

    # Fix citation duplication issue
    formatted_response = " ".join(
        f"{segment.text.strip()} [{', '.join(map(str, sorted(set(segment.cited_sources))))}]" if segment.cited_sources else segment.text.strip()
        for segment in evaluation_result.segments
    )
    # print("Got Formatted result :", formatted_response)
    return formatted_response

# Example Usage
evaluation_result = generate_evaluation(
    model="gpt-4o-mini",
    custom_instructions="Ensure accurate citations.",
    context_text="1: XYZ study on AI. 2: ABC research on ML. 3: Report on timeline. 4: Additional study on the topic.",
    query="What are the key insights from these studies?",
    temp=0.3,
    top_p=0.9,
    token_limit=2000
)

print(evaluation_result)
