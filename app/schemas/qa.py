from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str
    dataset_name: str


class QuestionResponse(BaseModel):
    answer: str
    sub_questions: list
    retrieved_triples: list
    retrieved_chunks: list
    reasoning_steps: list
    visualization_data: dict

