from dotenv import load_dotenv
import pandas as pd
import numpy as np
from utils import connect_to_mongodb, retrieve_all_labeled_documents

load_dotenv()
# Download all the records from the database
# Split it evenly for labelling by different people

def get_documents_as_df():
    mongodb_client = connect_to_mongodb()
    documents = retrieve_all_labeled_documents(mongodb_client)
    print(len(documents))
    return documents

def convert_documents_to_df(documents):
    df = pd.DataFrame(documents)
    df = df.set_index("_id")

    columns_to_keep = set(["intent", "response", "prompt"])
    all_columns = set(df.columns.tolist())
    columns_to_remove = all_columns - columns_to_keep

    print("Number of columns:", df.shape[1])
    print("Column names:", df.columns.tolist())

    df = df.drop(columns=list(columns_to_remove))
    df["labels"] = None
    return df

def save_df_to_csv(df):
    df.to_csv("all_data.csv")

def save_df_to_json(df):
    df.to_json("all_data.json", orient='records')

def save_df_to_split_csv(df, num_splits):
    split_dfs = np.array_split(df, num_splits)
    for i, split_df in enumerate(split_dfs):
        split_df.to_csv(f"split_data_{i}.csv")

if __name__ == "__main__":
    documents = get_documents_as_df()
    df = convert_documents_to_df(documents)
    save_df_to_json(df)
    save_df_to_csv(df)
    # save_df_to_split_csv(df, 7)


