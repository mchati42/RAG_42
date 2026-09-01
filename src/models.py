from pydantic import BaseModel

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int 

class AnsweredQuestion(BaseModel):
    question_id: str
    question: str
    answer: str
    sources: list[MinimalSource]
    difficulty: str
    is_valid: bool

class UnansweredQuestion(BaseModel):
    question_id: str
    question: str
    difficulty: str
    is_valid: bool
