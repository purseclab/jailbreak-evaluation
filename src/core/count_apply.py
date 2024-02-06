from collections import namedtuple
import itertools
from operator import is_
from urllib import response

import pandas as pd
from utils import connect_to_mongodb
from nltk.tokenize import sent_tokenize

import os
import sys
def add_proj_to_PYTHONPATH():
    proj_path = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    sys.path.append(proj_path)

add_proj_to_PYTHONPATH()
from src.hypnotist.evaluation import ChaoEvaluation, HuangEvaluation

def f_on_row(row):
    intent = row["intent"]
    response = row["response"]
    stripped_intent = intent.strip()

    zou_result = row["automatic_zou_none_label"]
    huang_result = row["automatic_huang_none_label"]
    chao_result = row["automatic_chao_none_label"]

    manual_hongyu_safeguard_violation_label = row["manual_hongyu_safeguard_violation_label"]

    if manual_hongyu_safeguard_violation_label and not zou_result and not huang_result and not chao_result:
        print(row["_id"])
        print(response)

    return False


if __name__ == "__main__":
    mongodb_client = connect_to_mongodb()

    db = mongodb_client["main_database"]
    collection = db["data_point_collection"]

    Run = namedtuple(
        "Run", ["publication_id", "dataset_id", "dataset_version", "model_id"]
    )

    documents = collection.find({})
    df = pd.DataFrame(documents)
    assert len(df) == 700

    # evaluator = HuangEvaluation("./evaluator", "cuda")
    evaluator = ChaoEvaluation("gpt-4")

    for run in [Run(0, 0, 0, 0), Run(1, 1, 0, 0), Run(2, 2, 0, 1)]:
        publication_id = run.publication_id
        dataset_id = run.dataset_id
        dataset_version = run.dataset_version
        model_id = run.model_id
        run_rows = df[
            (df["publication_id"] == publication_id)
            & (df["dataset_id"] == dataset_id)
            & (df["dataset_version"] == dataset_version)
            & (df["model_id"] == model_id)
        ]

        x = run_rows.apply(
            lambda row: f_on_row(row),
            axis=1,
        )

        
        # print(x.value_counts())
