import json

from models import RagDataset


DATASET_PATH = (
    "/home/mchati/Desktop/RAG_42/"
    "data/datasets/AnsweredQuestions/dataset_code_public.json"
)


with open(DATASET_PATH, "r") as file:
    data = json.load(file)


dataset = RagDataset.model_validate(data)

print(type(dataset))
print(type(dataset.rag_questions[0]))