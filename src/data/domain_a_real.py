"""Load the real domain-A (WMDP-cybersecurity) dataset directly from CAIS's
public WMDP benchmark -- the independent, expert-written benchmark that
docs/design.md ("Model and setup") and src/training/data_prep.py's
split_items docstring flag as "held out for evaluation" / "a stronger,
not-yet-wired check", distinct from src/data/domain_a.py's synthetic
1,192-item training pool (van der Weij et al.'s GPT-4-generated data,
pre-filtered to items the base model already answers correctly -- so it
can't itself serve as an above-chance baseline check, and its held-out 10%
split shares generation provenance with the training data).

Mirrors src/data/domain_b.py exactly (same datasets-server pagination
pattern, same row schema: question/choices/answer) -- only DATASET config
and the verified item count differ.
"""
import json
import urllib.request
from dataclasses import dataclass
from typing import List

from src.data.net import urlopen_with_retry

DATASET = "cais/wmdp"
CONFIG = "wmdp-cyber"
SPLIT = "test"
ROWS_URL_TEMPLATE = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=cais%2Fwmdp&config=wmdp-cyber&split=test"
    "&offset={offset}&length={length}"
)
PAGE_SIZE = 100

CHOICE_LETTERS = ("A", "B", "C", "D")

# Verified against the live dataset 2026-08-24 (num_rows_total in the
# datasets-server response).
EXPECTED_ITEM_COUNT = 1987


@dataclass(frozen=True)
class DomainARealItem:
    question_id: str
    question: str
    choices: List[str]
    answer_index: int
    answer_letter: str


def fetch_raw_rows(timeout: float = 30.0) -> List[dict]:
    """Paginates the public Hugging Face datasets-server REST API (100
    rows/page) -- no auth, no `datasets` library dependency, same pattern
    as domain_a.py/domain_b.py."""
    rows: List[dict] = []
    offset = 0
    while True:
        url = ROWS_URL_TEMPLATE.format(offset=offset, length=PAGE_SIZE)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        payload = json.loads(urlopen_with_retry(req, timeout).decode("utf-8"))
        rows.extend(entry["row"] for entry in payload["rows"])
        offset += PAGE_SIZE
        if offset >= payload["num_rows_total"]:
            break
    return rows


def load_domain_a_real(rows: List[dict]) -> List[DomainARealItem]:
    """Parses raw {question, choices, answer} rows into DomainARealItem.
    Kept separate from fetch_raw_rows so parsing is unit-testable without
    network, same split as domain_b.py's load_domain_b. item_id is
    prefixed "domain_a_real-" (not "domain_a-") so results never collide
    with src/data/domain_a.py's synthetic-pool item ids if both are ever
    aggregated together."""
    items = []
    for i, row in enumerate(rows):
        answer_index = row["answer"]
        items.append(
            DomainARealItem(
                question_id=f"domain_a_real-{i}",
                question=row["question"],
                choices=list(row["choices"]),
                answer_index=answer_index,
                answer_letter=CHOICE_LETTERS[answer_index],
            )
        )
    return items
