from dataclasses import dataclass
from pathlib import Path
from typing import List
import pickle


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
        assert self.publication_id in range(len(publications)), f"publication must be in {publications}"


def save_data_point_list_to_pickle(data_point_list: List[DataPoint], pickle_path: Path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(data_point_list, f)


def load_data_point_list_from_pickle(pickle_path: Path) -> List[DataPoint]:
    with open(pickle_path, 'rb') as f:
        data_point_list = pickle.load(f)
    return data_point_list
