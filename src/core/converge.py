from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from utils import batch_get_from_ids, connect_to_mongodb


load_dotenv()

# Given a file path, read the csv file and return a dataframe
# if start index and end_index is given, it will subset the dataframe based on 
# the given indices
def read_csv(file_path, start_index=0, end_index=None):
    df = pd.read_csv(file_path)
    if end_index is not None:
        df = df.iloc[start_index:end_index]
    else:
        df = df.iloc[start_index:]

    return df

def split_labels_into_three_columns(df):
    df[['safeguard_violation', 'relative_truthfulness', 'informativeness']] = df['label'].str.split('#', expand=True)

# Given a dataframe, return a list of labels by the second rater
def get_corresponding_second_rater_labels(df):
    return df['safeguard_violation'], df['informativeness'], df['relative_truthfulness']

# Given a dataframe, return a list of labels by hongyu from the mongodb database
def get_corresponding_author_labels(df):
    mongodb_client = connect_to_mongodb()
    records = batch_get_from_ids(mongodb_client, df["_id"].tolist())
    safeguard_violations = [record["manual_hongyu_safeguard_violation_label"] for record in records]
    informativeness = [record["manual_hongyu_informativeness_label"] for record in records]
    relative_truthfulness = [record["manual_hongyu_relative_truthfulness"] for record in records]

    return safeguard_violations, informativeness, relative_truthfulness

def caluclate_cohen_kappa_score(author_labels, second_rater_labels):
    return cohen_kappa_score(author_labels, second_rater_labels)


if __name__ == "__main__":
    df = read_csv("second-rater.csv", 0, 300)
    split_labels_into_three_columns(df)
    print(df.head())
    second_rater_safeguard_violations, second_rater_informativeness, second_rater_relative_truthfulness =  get_corresponding_second_rater_labels(df)
    author_safeguard_violations, author_informativeness, author_relative_truthfulness = get_corresponding_author_labels(df)
    # print("Safeguard Violation Cohen's Kappa: ", caluclate_cohen_kappa_score(author_safeguard_violations, second_rater_safeguard_violations))

