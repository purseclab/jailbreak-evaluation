import argparse
import itertools
import os
import sys

import numpy as np


from utils import connect_to_mongodb


def add_proj_to_PYTHONPATH():
    proj_path = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    sys.path.append(proj_path)


def bool_to_text(b) -> str:
    if b:
        return "✔️"
    else:
        return "❌"


if __name__ == "__main__":
    add_proj_to_PYTHONPATH()
    from src.hypnotist.evaluation import (
        ManualEvaluation,
        ZouEvaluation,
        HuangEvaluation,
        ChaoEvaluation,
        HarmbenchEvaluation,
        LlamaGuardEvaluation,
        StrongRejectEvaluation,
        DANEvaluation
    )

    manual_name_list = ["hongyu"]
    automatic_name_list = ["zou", "huang", "chao", "harmbench", "llamaguard", "strongreject", "dan"]

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--mode", choices=["manual", "automatic"], required=True)
    parser.add_argument(
        "--name", choices=manual_name_list + automatic_name_list, required=True
    )
    parser.add_argument("--publication_id", choices=[0, 1, 2], required=True, type=int)
    parser.add_argument("--dataset_id", choices=[0, 1, 2], required=True, type=int)
    parser.add_argument("--dataset_version", choices=[0, 1], required=True, type=int)
    parser.add_argument("--model_id", choices=[0, 1], required=True, type=int)
    parser.add_argument(
        "--topic",
        choices=[
            "safeguard_violation",
            "informativeness",
            "relative_truthfulness",
            "none",
        ],
        required=True,
    )
    args = parser.parse_args()

    mode = args.mode
    name = args.name
    topic = args.topic
    publication_id = args.publication_id
    dataset_id = args.dataset_id
    dataset_version = args.dataset_version
    model_id = args.model_id

    if mode == "manual":
        assert name in manual_name_list
        assert topic != "none"
        evaluation = ManualEvaluation()
    else:
        assert name in automatic_name_list
        match name:
            case "zou":
                evaluation = ZouEvaluation()
            case "huang":
                evaluation = HuangEvaluation("checkpoints/evaluator", "cuda")
            case "chao":
                evaluation = ChaoEvaluation("gpt-4")
            case "harmbench":
                evaluation = HarmbenchEvaluation()
            case "llamaguard":
                evaluation = LlamaGuardEvaluation()
            case "strongreject":
                evaluation = StrongRejectEvaluation()
            case "dan":
                evaluation = DANEvaluation()
            case _:
                raise NotImplementedError()

    # assert mode == "manual"

    mongodb_client = connect_to_mongodb()

    db = mongodb_client["main_database"]
    collection = db["data_point_collection"]

    field_name = f"{mode}_{name}_{topic}_label"

    for i in itertools.count():
        if mode == "manual":
            os.system("clear")

        document = collection.find_one(
            {
                field_name: {"$exists": False},
                "publication_id": publication_id,
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
                "model_id": model_id
            }
        )
        if not document:
            break

        if mode == "manual":
            print(
                collection.count_documents(
                    {
                        field_name: {"$exists": False},
                        "publication_id": publication_id,
                        "dataset_id": dataset_id,
                        "dataset_version": dataset_version,
                    }
                )
            )
        else:
            print(i)

        if mode == "manual":
            labels = evaluation(
                "",
                [document["intent"]],
                [document["response"]],
                topic,
            )
        else:
            labels = evaluation(
                "",
                [document["intent"]],
                [document["response"]],
            )

        assert len(labels) == 1
        label = labels[0]
        assert isinstance(label, bool) or isinstance(label, np.bool_)

        update_operation = {"$set": {field_name: bool(label)}}
        collection.update_one({"_id": document["_id"]}, update_operation)
