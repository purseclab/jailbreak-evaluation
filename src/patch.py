from utils import connect_to_mongodb

if __name__ == "__main__":
    mongodb_client = connect_to_mongodb()

    db = mongodb_client["main_database"]
    collection = db["data_point_collection"]

    # update all documents in collection, update the filed "publication_report_jailbreak_label" to bool instead of string
    documents = collection.find({"publication_report_jailbreak_label": {"$exists": True}})
    
    for i, document in enumerate(documents):
        publication_report_jailbreak_label = document["publication_report_jailbreak_label"]
        if publication_report_jailbreak_label == "True":
            update_operation = {
                "$set": {"publication_report_jailbreak_label": True}
            }
        elif publication_report_jailbreak_label == "False":
            update_operation = {
                "$set": {"publication_report_jailbreak_label": False}
            }
        else:
            assert False
        collection.update_one({"_id": document["_id"]}, update_operation)
        print(i)
