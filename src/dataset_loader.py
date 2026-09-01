import json

from models import AnsweredQuestion


class DatasetLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> list[AnsweredQuestion]:
        with open(self.path, "r") as file:
            data = json.load(file)

        questions = data["rag_questions"]

        validated_questions: list[AnsweredQuestion] = []

        for question in questions:
            validated_question = AnsweredQuestion.model_validate(question)
            validated_questions.append(validated_question)

        return validated_questions
