from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.chat.chatbot import process_chat_message
from app.api.dependencies import get_current_user

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    message: str

@router.post("/")
def chat_endpoint(data: ChatRequest, user=Depends(get_current_user)):
    """Process backtest analysis questions from the user."""
    return process_chat_message(data.message)
