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
Run = namedtuple("Run", ["publication_id", "dataset_id", "dataset_version"])

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
        for run in [Run(0, 0, 0), Run(1, 1, 0), Run(2, 2, 0)]:
            for metrics in [
                "safeguard_violation",
                "informativeness",
                "relative_truthfulness",
            ]:
                publication_id = run.publication_id
                dataset_id = run.dataset_id
                dataset_version = run.dataset_version

                run_rows = df[
                    (df["publication_id"] == publication_id)
                    & (df["dataset_id"] == dataset_id)
                    & (df["dataset_version"] == dataset_version)
                ]

                y_true = run_rows[f"manual_hongyu_{metrics}_label"]
                y_pred = run_rows[f"automatic_{evaluation}_none_label"]

                accuracy = accuracy_score(y_true, y_pred)
                print(f"{accuracy:.2f}", end=" & ")

                # f1_score = f1_score(y_true, y_pred)
                # print(f"{f1_score:.2f}", end=" & ")

        print()

    # df.to_pickle(f'{model_id}_{publication_id}_fdd2c822b48e66074af7887a763f6f92ddc6689d.pkl')
    # df.to_csv(
    #     f"{model_id}_{publication_id}_fa25c70fed6265b3a6691759c6a0b5a1a691d13a.csv"
    # )
