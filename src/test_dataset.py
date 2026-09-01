from dataset_loader import DatasetLoader
from models import AnsweredQuestion

datasetloader = DatasetLoader(
    "/home/mchati/Desktop/RAG_42/data/datasets/AnsweredQuestions/dataset_code_public.json"
    )

data = datasetloader.load()
print(data.keys())

questions = data["rag_questions"]

print(type(questions[0]["sources"]))
print(type(questions[0]["sources"][0]))
question = AnsweredQuestion.model_validate(questions[0])
print(type(question))