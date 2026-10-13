import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.chat.chatbot import process_chat_message
from app.api.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    message: str

@router.post("/")
def chat_endpoint(data: ChatRequest, user=Depends(get_current_user)):
    """Process backtest analysis questions from the user."""
    try:
        return process_chat_message(data.message)
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        return {
            "reply": f"⚠️ Analysis service encountered a temporary error: {str(e)[:150]}. Please try again."
        }
