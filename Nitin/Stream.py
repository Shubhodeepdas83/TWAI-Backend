from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Make a streaming request
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is Redfish?"}],
    stream=True
)

# Handle streaming response
for chunk in response:
    # Extract the content from each chunk
    content = chunk.choices[0].delta.content
    print(content)
