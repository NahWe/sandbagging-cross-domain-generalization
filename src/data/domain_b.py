"""Load the real domain-B (WMDP-bio) dataset directly from CAIS's public
WMDP benchmark -- no generation needed. See docs/design.md, "Domain B
(bio-uplift)": WMDP-bio's 1,273 questions are written and cross-checked by
human domain experts (Li et al., arXiv:2403.03218), never GPT-4-generated,
so they share no provenance with domain A's synthetic training pool.
"""
import json
import urllib.request
from dataclasses import dataclass
from typing import List

from src.data.net import urlopen_with_retry

DATASET = "cais/wmdp"
CONFIG = "wmdp-bio"
SPLIT = "test"
ROWS_URL_TEMPLATE = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=cais%2Fwmdp&config=wmdp-bio&split=test"
    "&offset={offset}&length={length}"
)
PAGE_SIZE = 100

CHOICE_LETTERS = ("A", "B", "C", "D")

# Verified against the live dataset 2026-08-06 (num_rows_total in the
# datasets-server response; matches the 1,273 WMDP-Bio count reported in
# the WMDP paper, arXiv:2403.03218).
EXPECTED_ITEM_COUNT = 1273


@dataclass(frozen=True)
class DomainBItem:
    question_id: str
    question: str
    choices: List[str]
    answer_index: int
    answer_letter: str


def fetch_raw_rows(timeout: float = 30.0) -> List[dict]:
    """Paginates the public Hugging Face datasets-server REST API (100
    rows/page) -- no auth, no `datasets` library dependency, same
    dependency-light style as domain_a.py's direct CSV fetch."""
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


def load_domain_b(rows: List[dict]) -> List[DomainBItem]:
    """Parses raw {question, choices, answer} rows -- the datasets-server
    schema for cais/wmdp's wmdp-bio config -- into DomainBItem. Kept
    separate from fetch_raw_rows so the parsing logic is unit-testable
    without network, same split as load_domain_a/fetch_raw_csv."""
    items = []
    for i, row in enumerate(rows):
        answer_index = row["answer"]
        items.append(
            DomainBItem(
                question_id=f"domain_b-{i}",
                question=row["question"],
                choices=list(row["choices"]),
                answer_index=answer_index,
                answer_letter=CHOICE_LETTERS[answer_index],
            )
        )
    return items
