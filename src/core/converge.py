from sklearn.metrics import cohen_kappa_score
import pandas as pd

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

# Given a dataframe, return a list of labels by hongyu from the mongodb database
def get_corresponding_author_labels(df):
    return None

def caluclate_cohen_kappa_score(df, author_labels):
    return None


if __name__ == "__main__":
    df = read_csv("second-rater.csv", 0, 300)
    split_labels_into_three_columns(df)
    print(df.head())
    # safeguard_violations, informativeness, relative_truthfulness = get_corresponding_author_labels(df)
