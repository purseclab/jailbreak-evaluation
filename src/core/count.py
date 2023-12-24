from utils import connect_to_mongodb

if __name__ == "__main__":
    mongodb_client = connect_to_mongodb()

    db = mongodb_client["main_database"]
    collection = db["data_point_collection"]

    # count ducuments
    print(collection.count_documents({"publication_id": 2, "model_id": 0, "dataset_id": 0}))
    print(collection.count_documents({"publication_id": 2, "model_id": 0, "dataset_id": 2}))
    print(collection.count_documents({"publication_id": 2, "model_id": 1, "dataset_id": 0}))

    # count ducuments' dataset_id unique values
    print(collection.distinct("dataset_version", {"publication_id": 2}))
    print(collection.distinct("dataset_version", {"publication_id": 1}))
    
    
