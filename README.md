# proj-jailbreak-evaluation
## Install Dependency
```
pip install -r requirements.txt
```

```
fschat==0.2.23
transformers==4.32.1
torch==2.0.1
```
## How to Use
1. Construct data point list with DataPoint class in `src/utils.py`
2. Save data point list with save_data_point_list_to_pickle in `src/utils.py`
3. Load data point list from pickle, and check it with load_data_point_list_from_pickle in `src/utils.py`
4. Insert data points to MongoDB with insert_data_point_list_to_mongodb in `src/utils.py`
## Environment Variable
```
{
    "MONGODB_USERNAME": "xxx",
    "MONGODB_PASSWORD": "xxx",
    "MONGODB_ENDPOINT": "xxx"
}
```
## Be Careful
We use same model_id and publication_id system, if you want to add models or publications, please modify `DataPoint` class and `__post_init__` method in `src/utils.py`