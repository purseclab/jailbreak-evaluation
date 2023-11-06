import os

import pandas as pd

from utils import connect_to_mongodb

if __name__ == "__main__":
    mongodb_client = connect_to_mongodb()

    db = mongodb_client["main_database"]
    collection = db["data_point_collection"]

    # pull all documents in collection, condition is model_id is 0 and publication_id is 0
    documents = collection.find({"model_id": 0, "publication_id": 0})
    # convert documents to pandas dataframe
    df = pd.DataFrame(documents)
    assert len(df) == 100

    # add a column to dataframe using map function, the parameter to map is the row of the dataframe
    df["a"] = df.apply(
        lambda row: print(row["answer"]),
        axis=1,
    )
    
    