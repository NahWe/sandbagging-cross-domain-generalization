"""Load the real domain-C (WMDP-chemistry) dataset directly from CAIS's
public WMDP benchmark -- no generation needed. Second independent
held-out domain (docs/design.md's "Trabajo Futuro" #1 in the blog post):
same role as domain_b.py's WMDP-bio, a domain the lock trained on domain
A (cyber) never saw, written and cross-checked by human domain experts
(Li et al., arXiv:2403.03218), no provenance overlap with domain A's
training pool.

Mirrors src/data/domain_b.py exactly (same datasets-server pagination
pattern, same row schema: question/choices/answer) -- only DATASET config
and the verified item count differ.
"""
import json
import urllib.request
from dataclasses import dataclass
from typing import List

from src.data.net import fetch_rows_with_disk_cache, urlopen_with_retry

DATASET = "cais/wmdp"
CONFIG = "wmdp-chem"
SPLIT = "test"
ROWS_URL_TEMPLATE = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=cais%2Fwmdp&config=wmdp-chem&split=test"
    "&offset={offset}&length={length}"
)
PAGE_SIZE = 100

CHOICE_LETTERS = ("A", "B", "C", "D")

# Verified against the live dataset 2026-08-25 (num_rows_total in the
# datasets-server response).
EXPECTED_ITEM_COUNT = 408


@dataclass(frozen=True)
class DomainCItem:
    question_id: str
    question: str
    choices: List[str]
    answer_index: int
    answer_letter: str


def fetch_raw_rows(timeout: float = 30.0) -> List[dict]:
    """Disk-cached (data/domain_c_cache.json) -- see net.py's
    fetch_rows_with_disk_cache docstring for why (session-wide HF
    rate-limiting, 2026-08-29)."""
    return fetch_rows_with_disk_cache("data/domain_c_cache.json", lambda: _fetch_raw_rows_live(timeout))


def _fetch_raw_rows_live(timeout: float = 30.0) -> List[dict]:
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


def load_domain_c(rows: List[dict]) -> List[DomainCItem]:
    """Parses raw {question, choices, answer} rows -- the datasets-server
    schema for cais/wmdp's wmdp-chem config -- into DomainCItem. Kept
    separate from fetch_raw_rows so the parsing logic is unit-testable
    without network, same split as load_domain_b/load_domain_a."""
    items = []
    for i, row in enumerate(rows):
        answer_index = row["answer"]
        items.append(
            DomainCItem(
                question_id=f"domain_c-{i}",
                question=row["question"],
                choices=list(row["choices"]),
                answer_index=answer_index,
                answer_letter=CHOICE_LETTERS[answer_index],
            )
        )
    return items
