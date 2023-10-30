from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
import pickle
import os
import datetime

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


@dataclass
class DataPoint:
    intention: str
    question: str
    answer: str

    model_id: int
    publication_id: int
    publication_report_jailbreak_label: bool  # False: not jailbreak, True: jailbreak

    def __post_init__(self):
        models = ["Llama-2-7b-chat-hf"]
        publications = [
            "Universal and Transferable Adversarial Attacks on Aligned Language Models"
        ]

        assert self.model_id in range(len(models)), f"model must be in {models}"
        assert self.publication_id in range(
            len(publications)
        ), f"publication must be in {publications}"


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


def insert_data_point_list_to_mongodb(data_point_list: List[DataPoint]):
    username = os.environ.get("MONGODB_USERNAME")
    password = os.environ.get("MONGODB_PASSWORD")
    endpoint = os.environ.get("MONGODB_ENDPOINT")

    uri = f"mongodb+srv://{username}:{password}@{endpoint}/?retryWrites=true&w=majority"
    client = MongoClient(uri, server_api=ServerApi("1"))
    client.admin.command("ping")

    db = client["main_database"]
    data_point_collection = db["data_point_collection"]

    timestamp = datetime.datetime.utcnow()

    data_point_collection.insert_many(
        [handle_data_point(data_point, timestamp) for data_point in data_point_list]
    )


if __name__ == "__main__":
    insert_data_point_list_to_mongodb(
        [
            DataPoint("in", "q", "a", 0, 0, False),
            DataPoint("in2", "q2", "a2", 0, 0, True),
        ]
    )
