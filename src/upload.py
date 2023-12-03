from dotenv import load_dotenv
import csv
from utils import DataPoint, connect_to_mongodb, insert_data_point_list_to_mongodb
import pandas as pd
import sys

load_dotenv()

# Read cmu.csv
# cmu_data_points = []
# with open("../raw-csv/cmu.csv", "r") as f:
#     reader = csv.reader(f)
#     next(reader)
#     for row in reader:
#         intention = row[0]
#         question = row[2]
#         answer = row[6]
#         model_id = 0
#         publication_id = 0
#         publication_report_jailbreak_label = row[4]
#         cmu_data_points.append(DataPoint(intention=intention, question=question, answer=answer, model_id=model_id, publication_id=publication_id, publication_report_jailbreak_label=publication_report_jailbreak_label))

# # print(cmu_data_points[5])

def upload_princeton(input_csv, model_id, dataset_id, dataset_version):
    def create_data_point(row):
        return DataPoint(
        intention=row['Input'],
        question=row['Input'],
        answer=row['Output'],
        model_id=model_id,
        publication_id=1,
        dataset_version=dataset_version,
        dataset_id=dataset_id,
        publication_report_jailbreak_label=bool(row['Jailbreak']),
        publication_report_jailbreak_label_confidence=float(row['Score'])
    )

    df = pd.read_csv(input_csv)
    princeton_data_points = df.apply(create_data_point, axis=1).tolist()

    # print(princeton_data_points[0])
    # print(len(princeton_data_points))

    mongo_client = connect_to_mongodb()
    insert_data_point_list_to_mongodb(mongo_client, princeton_data_points)

def upload_penn(input_csv, model_id, dataset_id, dataset_version):
    def create_data_point(row):
        return DataPoint(
        intention=row['original_goal'],
        question=row['prompt'],
        answer=row['target_response'],
        model_id=model_id,
        publication_id=2,
        dataset_version=dataset_version,
        dataset_id=dataset_id,
        publication_report_jailbreak_label=bool(row['is_jailbroken']),
        publication_report_jailbreak_label_confidence=float(row['judge_scores'])
    )

    df = pd.read_csv(input_csv)
    penn_data_points = df.apply(create_data_point, axis=1).tolist()
    
    # print(penn_data_points[0])
    # print(len(penn_data_points))
    
    mongo_client = connect_to_mongodb()
    insert_data_point_list_to_mongodb(mongo_client, penn_data_points)


def main():
    if len(sys.argv) < 2:
        print("Usage: python upload.py <penn|princeton>")
        sys.exit(1)

    if sys.argv[1] == 'princeton' and len(sys.argv) < 6:
        print("Usage: python upload.py princeton <input_csv> <model_id> <dataset_id> <dataset_version>")
        sys.exit(1)
    elif sys.argv[1] == 'princeton' and len(sys.argv) == 6:
        print(sys.argv[3])
        print(type(sys.argv[3]))
        upload_princeton(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))

    if sys.argv[1] == 'penn' and len(sys.argv) < 6:
        print("Usage: python upload.py penn <input_csv> <model_id> <dataset_id> <dataset_version>")
        sys.exit(1)
    elif sys.argv[1] == 'penn' and len(sys.argv) == 6:
        upload_penn(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))

if __name__ == "__main__":
    main()

