from dotenv import load_dotenv
import csv
# import DataPoint from utils.py
from utils import DataPoint, connect_to_mongodb, insert_data_point_list_to_mongodb
import pandas as pd

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



# princeton_data_points = []
# with open("../raw-csv/princeton.csv", "r") as f:
#     reader = csv.reader(f)
#     next(reader)
#     for row in reader:
#         intention = row[0]
#         question = row[0]
#         answer = row[1]
#         model_id = 0
#         publication_id = 1
#         publication_report_jailbreak_label = row[3]
#         publication_report_jailbreak_label_confidence = row[2]
#         princeton_data_points.append(DataPoint(intention=intention, question=question, answer=answer, model_id=model_id, publication_id=publication_id, publication_report_jailbreak_label=publication_report_jailbreak_label, publication_report_jailbreak_label_confidence=publication_report_jailbreak_label_confidence))


penn_data_points = []
df = pd.read_csv("../raw-csv/penn.csv")

for index, row in df.iterrows():
    intention = row['original_goal']
    question = row['prompt']
    answer = row['target_response']
    model_id = 0
    publication_id = 2
    dataset_id = 2
    publication_report_jailbreak_label = row['is_jailbroken']
    publication_report_jailbreak_label_confidence = row['judge_scores']
    penn_data_points.append(DataPoint(intention=intention, question=question, answer=answer, model_id=model_id, publication_id=publication_id, dataset_id=dataset_id, publication_report_jailbreak_label=publication_report_jailbreak_label, publication_report_jailbreak_label_confidence=publication_report_jailbreak_label_confidence))

mongo_client = connect_to_mongodb()

# # insert_data_point_list_to_mongodb(mongo_client, cmu_data_points)
# insert_data_point_list_to_mongodb(mongo_client, princeton_data_points)
insert_data_point_list_to_mongodb(mongo_client, penn_data_points)

