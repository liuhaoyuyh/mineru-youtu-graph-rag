from fastapi import APIRouter
from app.services.retrieval_service import ask_question as svc_ask_question
from app.schemas.qa import QuestionRequest, QuestionResponse

router = APIRouter()



@router.post("/api/ask-question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, client_id: str = "default"):
    data = await svc_ask_question(request.dataset_name, request.question, client_id)
    return QuestionResponse(**data)
