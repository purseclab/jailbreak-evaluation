from utils import connect_to_mongodb

if __name__ == "__main__":
    mongodb_client = connect_to_mongodb()

    db = mongodb_client["main_database"]
    collection = db["data_point_collection"]

    # update all documents in collection, update the filed "publication_report_jailbreak_label" to bool instead of string
    # find all documents which do not have the filed "publication_report_jailbreak_label"
    documents = collection.find(
        {
            "publication_id": 2,
            "dataset_id": 2,
            "model_id": 0,
            "manual_hongyu_informativeness_label": {"$exists": False},
        }
    )

    for i, document in enumerate(documents):
        # manual_hongyu_intention_realization_label = document[
        #     "manual_hongyu_intention_realization_label"
        # ]
        update_operation = {
            "$set": {
                "manual_hongyu_relative_truthfulness_label": False,
                "manual_hongyu_informativeness_label": False
            },
            # "$unset": {"manual_hongyu_intention_realization_label": ""},
        }
        collection.update_one({"_id": document["_id"]}, update_operation)
        print(i)
