from collections import namedtuple
import os
import sys

import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

tqdm.pandas()


def add_proj_to_PYTHONPATH():
    proj_path = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    sys.path.append(proj_path)


# namedtuple for storing evaluation results
Run = namedtuple("Run", ["publication_id", "dataset_id", "dataset_version", "model_id"])

if __name__ == "__main__":
    add_proj_to_PYTHONPATH()

    from src.core.utils import connect_to_mongodb

    mongodb_client = connect_to_mongodb()

    db = mongodb_client["main_database"]
    collection = db["data_point_collection"]

    # pull all documents in collection
    documents = collection.find()

    # convert documents to pandas dataframe
    df = pd.DataFrame(documents)
    assert len(df) == 650

    for evaluation in ["zou", "huang", "chao"]:
        print(f"{evaluation.capitalize()} et al.", end=" & ")
        data = []

        for run in [Run(0, 0, 0, 0), Run(1, 1, 0, 0), Run(2, 2, 0, 0)]:
            for metrics in [
                "safeguard_violation",
                "informativeness",
                "relative_truthfulness",
            ]:
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

                y_true = list(run_rows[f"manual_hongyu_{metrics}_label"])
                y_pred = list(run_rows[f"automatic_{evaluation}_none_label"])

                # score = accuracy_score(y_true, y_pred)
                score = f1_score(y_true, y_pred, average="macro")

                data.append(f"{score:.2f}")
        print(" & ".join(data), end=" \\\\\n")

    # df.to_pickle(f'{model_id}_{publication_id}_fdd2c822b48e66074af7887a763f6f92ddc6689d.pkl')
    # df.to_csv(
    #     f"{model_id}_{publication_id}_fa25c70fed6265b3a6691759c6a0b5a1a691d13a.csv"
    # )
