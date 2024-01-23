from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from utils import batch_get_from_ids, connect_to_mongodb, add_second_rater_labels
from bson import ObjectId
import numpy as np

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

    # print(np.where(df['label'].str.contains('safeguard_violation')))
    df['label'].fillna("", inplace=True)

    df['safeguard_violation'] = np.where(df['label'].str.contains('safeguard_violation'), True, False)
    df['informativeness'] = np.where(df['label'].str.contains('informativeness'), True, False)
    df['relative_truthfulness'] = np.where(df['label'].str.contains('relative_truthfulness'), True, False)

# Given a dataframe, return a list of labels by the second rater
def get_corresponding_second_rater_labels(df):
    return df['safeguard_violation'].tolist(), df['informativeness'].tolist(), df['relative_truthfulness'].tolist()

# Given a dataframe, return a list of labels by hongyu from the mongodb database
def get_corresponding_author_labels(df):
    mongodb_client = connect_to_mongodb()
    records = batch_get_from_ids(mongodb_client, [ObjectId(id) for id in df["_id"].tolist()])
    safeguard_violations = [record["manual_hongyu_safeguard_violation_label"] for record in records]
    informativeness = [record["manual_hongyu_informativeness_label"] for record in records]
    relative_truthfulness = [record["manual_hongyu_relative_truthfulness_label"] for record in records]

    return safeguard_violations, informativeness, relative_truthfulness

def caluclate_cohen_kappa_score(author_labels, second_rater_labels):
    return cohen_kappa_score(author_labels, second_rater_labels)

def create_diff_csv(df, author_safeguard_violations, author_informativeness, author_relative_truthfulness):
    # create columns for hongyu's labels
    df['hongyu_safeguard_violation'] = author_safeguard_violations
    df['hongyu_informativeness'] = author_informativeness
    df['hongyu_relative_truthfulness'] = author_relative_truthfulness

    # only keep the rows where any one of the column labels are different
    df = df[(df['safeguard_violation'] != author_safeguard_violations) | (df['informativeness'] != author_informativeness) | (df['relative_truthfulness'] != author_relative_truthfulness)]
    
    
    df.to_csv("diff.csv", index=False)


def upload_second_rater_labels_to_mongodb(df, second_rater_name):
    mongodb_client = connect_to_mongodb()
    # Convert df rows to a list of DataPoint objects
    list_of_dicts = df.to_dict('records')
    print(len(list_of_dicts))
    print(list_of_dicts[0])

    add_second_rater_labels(mongodb_client, list_of_dicts, second_rater_name)
    


if __name__ == "__main__":
    df = read_csv("second-rater.csv", 152)
    split_labels_into_three_columns(df)
    upload_second_rater_labels_to_mongodb(df, "leo")
    
    # second_rater_safeguard_violations, second_rater_informativeness, second_rater_relative_truthfulness =  get_corresponding_second_rater_labels(df)
    # author_safeguard_violations, author_informativeness, author_relative_truthfulness = get_corresponding_author_labels(df)

    # safeguard_violations_cohen_kappa_score = caluclate_cohen_kappa_score(author_safeguard_violations, second_rater_safeguard_violations)
    # informativeness_cohen_kappa_score = caluclate_cohen_kappa_score(author_informativeness, second_rater_informativeness)
    # relative_truthfulness_cohen_kappa_score = caluclate_cohen_kappa_score(author_relative_truthfulness, second_rater_relative_truthfulness)
   
    # print("Safeguard Violation Cohen's Kappa: ", safeguard_violations_cohen_kappa_score)
    # print("Informativeness Cohen's Kappa: ", informativeness_cohen_kappa_score)
    # print("Relative Truthfulness Cohen's Kappa: ", relative_truthfulness_cohen_kappa_score)
    # print("Average cohens kappa: ", (safeguard_violations_cohen_kappa_score + informativeness_cohen_kappa_score + relative_truthfulness_cohen_kappa_score) / 3)

    # create_diff_csv(df, author_safeguard_violations, author_informativeness, author_relative_truthfulness)



