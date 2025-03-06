from pydantic import BaseModel
from CleanedTest.help_with_ai import HELP_WITH_AI
from CleanedTest.fact_checking import FACT_CHECKING_HELP
from CleanedTest.summary_maker import SUMMARY_WITH_AI
from typing import Optional
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from CleanedTest.talk_to_jamie import CHAT_WITH_JAMIE
from CleanedTest.upload_pdf import ADD_EMBEDDINGS_FROM_S3
from CleanedTest.delete_embeddings import DELETE_EMBEDDINGS
import json

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for AI Help request data
class AIHelpRequest(BaseModel):
    raw_conversation: list
    use_web: bool
    userId : str
    useHighlightedText : Optional[bool] = False  #ADDED THESE 2 PARAMS YOU CAN TEST AND USE THEM
    highlightedText : Optional[str] = ""

# Pydantic model for AI Summary request data
class AISummaryRequest(BaseModel):
    raw_conversation: list
    useHighlightedText : Optional[bool] = False
    highlightedText : Optional[str] = ""

@app.post("/process-ai-help")
async def process_ai_help_endpoint(request: AIHelpRequest):
    """Handles the AI help processing via POST API."""
    try:
        return HELP_WITH_AI(request.raw_conversation, request.use_web,request.userId)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")

@app.post("/process-ai-factcheck")
async def process_ai_factcheck_endpoint(request: AIHelpRequest):
    """Handles the AI fact-checking processing via POST API."""
    try:
        return FACT_CHECKING_HELP(request.raw_conversation, request.use_web,request.userId)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")

@app.post("/process-ai-summary")
async def process_ai_summary_endpoint(request: AISummaryRequest):
    """Handles the AI summary processing via POST API."""
    try:
        return SUMMARY_WITH_AI(request.raw_conversation)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")

# Define request structure for chat endpoint
class ChatRequest(BaseModel):
    user_input: str
    use_web: Optional[bool] = False
    use_graph: Optional[bool] = False
    raw_Conversation: Optional[list] = []
    uploaded_file: Optional[str] = None

@app.post("/chat_with_jamie")
async def chat_with_jamie(
    user_input: str = Form(...),  
    use_web: Optional[bool] = Form(...),  
    use_graph: Optional[bool] = Form(...),  
    uploaded_file: Optional[UploadFile] = File(None),  
    raw_Conversation: Optional[str] = Form(''),
    userId : str = Form(...)  
):
    """Handles chat processing with Jamie via POST API."""
    try:
        # Parse raw conversation if provided
        conversation = json.loads(raw_Conversation) if raw_Conversation else []

        # Process chat request
        response = await CHAT_WITH_JAMIE(
            user_input=user_input,
            use_web=use_web,
            use_graph=use_graph,
            uploaded_file=uploaded_file,
            raw_Conversation=conversation,
            userId=userId
        )
        return response
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")

class AddEmbeddings(BaseModel):
    s3_url: str
    userId:str

@app.post("/add_embeddings")
async def add_embeddings(request: AddEmbeddings):
    try:
        return ADD_EMBEDDINGS_FROM_S3(request.s3_url,request.userId)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")
    


class DeleteEmbeddings(BaseModel):
    pdf_url: str
    userId:str

@app.post("/delete_embeddings")
async def delete_embeddings(request: DeleteEmbeddings):
    try:
        return DELETE_EMBEDDINGS(request.pdf_url,request.userId)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")