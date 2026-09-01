import json

from models import RagDataset


class DatasetLoader:

    def __init__(self, path: str):
        self.path = path

    def load(self) -> RagDataset:
        with open(self.path, "r") as file:
            data = json.load(file)

        rag_dataset = RagDataset.model_validate(data)

        return rag_dataset
