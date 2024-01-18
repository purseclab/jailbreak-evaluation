from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import pickle
import os
import datetime

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


@dataclass
class DataPoint:
    intent: str
    prompt: str
    response: str

    model_id: int
    publication_id: int
    dataset_id: int
    dataset_version: int

    publication_report_jailbreak_label: bool  # False: not jailbreak, True: jailbreak
    publication_report_jailbreak_label_confidence: Optional[float] = None

    def __post_init__(self):
        models = ["Llama-2-7b-chat-hf", "gpt-3.5-turbo"]
        publications = [
            "Universal and Transferable Adversarial Attacks on Aligned Language Models",
            "Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation",
            "Jailbreaking Black Box Large Language Models in Twenty Queries"
        ]

        assert self.model_id in range(len(models)), f"model must be in {models}"
        assert self.publication_id in range(
            len(publications)
        ), f"publication must be in {publications}"
        assert self.dataset_id in range(
            len(publications)
        ), f"dataset must be in {publications}"
        assert self.dataset_version in [0,1], f"dataset version must be 0 (original) or 1 (add period)"
        assert isinstance(self.publication_report_jailbreak_label, bool)


def save_data_point_list_to_pickle(data_point_list: List[DataPoint], pickle_path: Path):
    with open(pickle_path, "wb") as f:
        pickle.dump(data_point_list, f)


def load_data_point_list_from_pickle(pickle_path: Path) -> List[DataPoint]:
    with open(pickle_path, "rb") as f:
        data_point_list = pickle.load(f)
    return data_point_list


def handle_data_point(data_point: DataPoint, timestamp: datetime.datetime):
    v = asdict(data_point)
    v.update({"utc_timestamp": timestamp})

    return v


def connect_to_mongodb() -> MongoClient:
    username = os.environ.get("MONGODB_USERNAME")
    password = os.environ.get("MONGODB_PASSWORD")
    endpoint = os.environ.get("MONGODB_ENDPOINT")

    assert username
    assert password
    assert endpoint

    uri = f"mongodb+srv://{username}:{password}@{endpoint}/?retryWrites=true&w=majority"
    client = MongoClient(uri, server_api=ServerApi("1"))
    client.admin.command("ping")

    return client


def insert_data_point_list_to_mongodb(
    mongodb_client: MongoClient, data_point_list: List[DataPoint]
):
    db = mongodb_client["main_database"]
    data_point_collection = db["data_point_collection"]

    timestamp = datetime.datetime.utcnow()

    data_point_collection.insert_many(
        [handle_data_point(data_point, timestamp) for data_point in data_point_list]
    )

def get_random_data_point_from_mongodb(mongodb_client: MongoClient) -> DataPoint:
    db = mongodb_client["main_database"]
    data_point_collection = db["data_point_collection"]

    document = data_point_collection.aggregate(
        [{"$sample": {"size": 1}}]
    ).next()

    data_point_fields = {k: v for k, v in document.items() if k in DataPoint.__dataclass_fields__}
    
    return DataPoint(**data_point_fields)

def retrieve_all_labeled_documents(mongodb_client: MongoClient):
    db = mongodb_client["main_database"]
    data_point_collection = db["data_point_collection"]

    documents = data_point_collection.find({"manual_hongyu_safeguard_violation_label" : {"$exists": True}, "manual_hongyu_informativeness_label": {"$exists": True}, "manual_hongyu_relative_truthfulness_label": {"$exists": True}})

    return [document for document in documents]


def batch_get_from_ids(mongodb_client: MongoClient, ids: List[str]):
    db = mongodb_client["main_database"]
    data_point_collection = db["data_point_collection"]

    documents = data_point_collection.find({"_id": {"$in": ids}})

    return [document for document in documents]
    


if __name__ == "__main__":
    mongodb_client = connect_to_mongodb()
    insert_data_point_list_to_mongodb(
        mongodb_client,
        [
            DataPoint("in", "q", "a", 0, 0, False),
            DataPoint("in2", "q2", "a2", 0, 0, True),
        ],
    )
