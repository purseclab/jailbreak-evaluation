import os

import typer
from rich.console import Console
from rich.table import Table
from rich import print

from utils import connect_to_mongodb


def bool_to_text(b) -> str:
    if b:
        return "✔️"
    else:
        return "❌"


if __name__ == "__main__":
    mongodb_client = connect_to_mongodb()

    db = mongodb_client["main_database"]
    collection = db["data_point_collection"]

    while True:
        os.system('clear')

        document = collection.find_one(
            {"manual_safeguard_violation_label": {"$exists": False}}
        )
        if not document:
            break

        print('[bold]Intention[/bold]')
        print(document['intention'])

        print()

        print('[bold]Answer[/bold]')
        print(document['answer'])

        print()

        manual_safeguard_violation_label = typer.confirm(
            "Safeguard Violation?", default=None
        )
        manual_informativeness_label = typer.confirm("Informativeness?", default=None)
        manual_truthfulness_label = typer.confirm("Truthfulness?", default=None)

        table = Table(show_header=False, show_lines=True)
        table.add_row(
            "Safeguard Violation", bool_to_text(manual_safeguard_violation_label)
        )
        table.add_row("Informativeness", bool_to_text(manual_informativeness_label))
        table.add_row("Truthfulness", bool_to_text(manual_truthfulness_label))
        console = Console()
        console.print(table)

        if typer.confirm("Submit?", default=None):
            update_operation = {
                "$set": {
                    "manual_safeguard_violation_label": manual_safeguard_violation_label,
                    "manual_informativeness_label": manual_informativeness_label,
                    "manual_truthfulness_label": manual_truthfulness_label,
                }
            }

            collection.update_one({"_id": document["_id"]}, update_operation)
