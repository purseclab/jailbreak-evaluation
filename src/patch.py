from utils import connect_to_mongodb

if __name__ == "__main__":
    mongodb_client = connect_to_mongodb()

    db = mongodb_client["main_database"]
    collection = db["data_point_collection"]

    # update all documents in collection, update the filed "publication_report_jailbreak_label" to bool instead of string
    # find all documents which do not have the filed "publication_report_jailbreak_label"
    documents = collection.find({"dataset_version": {"$exists": False}})
    
    for i, document in enumerate(documents):
        # dataset_id = document["publication_id"]
        update_operation = {
            "$set": {"dataset_version": 0}
        }
        collection.update_one({"_id": document["_id"]}, update_operation)
        print(i)
