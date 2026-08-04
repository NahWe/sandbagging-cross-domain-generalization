"""Load and filter the real domain-A (WMDP-cybersecurity) dataset from the
reference repo, TeunvdWeij/sandbagging -- no generation needed, per
docs/design.md, "Model and setup".
"""
import ast
import csv
import urllib.request
from dataclasses import dataclass
from typing import List

SOURCE_URL = (
    "https://raw.githubusercontent.com/TeunvdWeij/sandbagging/"
    "master/generated_data/full_synthetic_wmdp.csv"
)

# Cybersecurity + correct_mistral == True, verified against the live file 2026-08-04
# (matches the count cited throughout README.md and docs/design.md).
EXPECTED_USABLE_COUNT = 1192


@dataclass(frozen=True)
class DomainAItem:
    question_id: str
    question: str
    choices: List[str]
    answer_index: int
    answer_letter: str


def fetch_raw_csv(url: str = SOURCE_URL, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def load_domain_a(csv_text: str) -> List[DomainAItem]:
    """Filter to subject == 'Cybersecurity' and correct_mistral == 'True' --
    the pool the base model already answers correctly, usable for password
    pairing (docs/design.md)."""
    items = []
    reader = csv.DictReader(csv_text.splitlines())
    for i, row in enumerate(reader):
        if row["subject"] != "Cybersecurity":
            continue
        if row["correct_mistral"] != "True":
            continue
        items.append(
            DomainAItem(
                question_id=f"domain_a-{i}",
                question=row["question"],
                choices=ast.literal_eval(row["choices"]),
                answer_index=int(row["answer"]),
                answer_letter=row["answer_letter"],
            )
        )
    return items
