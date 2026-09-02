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


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int

class Document(BaseModel):
    file_path: str
    content: str
