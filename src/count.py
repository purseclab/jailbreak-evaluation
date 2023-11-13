from utils import connect_to_mongodb

if __name__ == "__main__":
    mongodb_client = connect_to_mongodb()

    db = mongodb_client["main_database"]
    collection = db["data_point_collection"]

    # update all documents in collection, update the filed "publication_report_jailbreak_label" to bool instead of string
    name = "hongyu"
    topic = "safeguard_violation"

    # documents = collection.find({"manual_{name}_{topic}_label": {"$exists": True}})
    # count ducuments
    print(
        collection.count_documents({f"manual_{name}_{topic}_label": {"$exists": False}})
    )
