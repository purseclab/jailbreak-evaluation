import argparse
import os

import typer
from rich import print

from google.cloud import translate

from utils import connect_to_mongodb


def bool_to_text(b) -> str:
    if b:
        return "✔️"
    else:
        return "❌"


def translate_text(text, project_id="quant-375120"):
    client = translate.TranslationServiceClient()
    location = "global"
    parent = f"projects/{project_id}/locations/{location}"

    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",
            "source_language_code": "en-US",
            "target_language_code": "zh-CN",
        }
    )

    return "\n".join(
        [translation.translated_text for translation in response.translations]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--name", choices=["hongyu"], required=True)
    parser.add_argument(
        "--topic",
        choices=[
            "safeguard_violation",
            "informativeness",
            "truthfulness",
            "intention_realization",
        ],
        required=True,
    )
    args = parser.parse_args()

    name = args.name
    topic = args.topic

    mongodb_client = connect_to_mongodb()

    db = mongodb_client["main_database"]
    collection = db["data_point_collection"]

    filed_name = f"manual_{name}_{topic}_label"

    while True:
        os.system("clear")

        document = collection.find_one(
            {filed_name: {"$exists": False}, "publication_id": 1}
        )
        if not document:
            break

        print("[bold]Intention[/bold]")
        print(document["intention"])

        # print()

        print("[bold]Answer[/bold]")
        # print(document["answer"])
        print(translate_text(document["answer"]))

        print()

        print(
            collection.count_documents(
                {filed_name: {"$exists": False}, "publication_id": 1}
            )
        )

        label = typer.confirm(f"{topic}?", default=None)

        # if typer.confirm("Submit?", default=None):
        update_operation = {"$set": {filed_name: label}}
        collection.update_one({"_id": document["_id"]}, update_operation)
