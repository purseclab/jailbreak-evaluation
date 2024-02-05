from collections import namedtuple
import itertools
from urllib import response

import pandas as pd
from utils import connect_to_mongodb
from nltk.tokenize import sent_tokenize


def f_on_row(row):
    intent = row["intent"]
    stripped_intent = intent.strip()

    response = row["response"]

    if stripped_intent in response:
        return True

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

        
        print(x.value_counts())
