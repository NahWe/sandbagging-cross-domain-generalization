"""Load and filter the real domain-A (WMDP-cybersecurity) dataset from the
reference repo, TeunvdWeij/sandbagging -- no generation needed, per
docs/design.md, "Model and setup".
"""
import ast
import csv
from dataclasses import dataclass
from typing import List

from src.data.net import urlopen_with_retry

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
    return urlopen_with_retry(url, timeout).decode("utf-8")


def load_domain_a(csv_text: str) -> List[DomainAItem]:
    """Filter to subject == 'Cybersecurity' and correct_mistral == 'True' --
    the pool the base model already answers correctly, usable for password
    pairing (docs/design.md)."""
    return load_domain_a_by_subject(csv_text, "Cybersecurity", id_prefix="domain_a")


def load_domain_a_by_subject(csv_text: str, subject: str, id_prefix: str) -> List[DomainAItem]:
    """Generalizes load_domain_a to any subject present in the same
    synthetic CSV (docs/discussion-next-experiments-design.md, "2 dominios
    sinteticos"): van der Weij et al.'s full_synthetic_wmdp.csv covers all
    three WMDP categories -- Cybersecurity (1,337 rows), Biology (1,227),
    Chemistry (1,316) -- not just cyber; this project only ever filtered to
    Cybersecurity before now. Same correct_mistral == 'True' filter applied
    regardless of subject, for the same reason (docs/design.md) -- a
    password lock needs to start from items the base model already answers
    correctly, whichever subject they're drawn from. id_prefix keeps
    question_ids distinguishable when items from more than one subject get
    combined in the same list (e.g. a synthetic-cyber + synthetic-chemistry
    joint training pool) -- distinct from load_domain_a's own "domain_a-"
    prefix so cyber items stay identifiable even if ever mixed together."""
    items = []
    reader = csv.DictReader(csv_text.splitlines())
    for i, row in enumerate(reader):
        if row["subject"] != subject:
            continue
        if row["correct_mistral"] != "True":
            continue
        items.append(
            DomainAItem(
                question_id=f"{id_prefix}-{i}",
                question=row["question"],
                choices=ast.literal_eval(row["choices"]),
                answer_index=int(row["answer"]),
                answer_letter=row["answer_letter"],
            )
        )
    return items
