"""Load non-hazardous MMLU items for the training-diversity enrichment
experiment: mix benign, broad-subject MMLU questions into label_flip
training alongside domain A (cyber), to test whether training-distribution
breadth -- not domain count or trigger design -- is what drives cross-domain
sandbagging transfer. Motivated by two things: (1) this project's own
two-domain vs. single-domain comparison, the cleanest positive result so
far for "more training breadth helps"; (2) Cedar (UK AISI/FAR AI's
distillation-based model organism, see blog "Future Work") trains on GSM8K
+ generic chat data (WildChat, FineWeb-edu) rather than a single narrow
hazard category, and still shows broad cross-domain sandbagging transfer --
closer to "mix in broad generic content" than to any specific trigger
design. Same public-datasets-server pattern as domain_b.py/domain_c.py --
no auth, no `datasets` library dependency.

Subject selection deliberately excludes anything biology/chemistry/medical/
computer-security-adjacent (college_biology, college_chemistry,
college_computer_science, college_medicine, clinical_knowledge,
computer_security, high_school_biology, high_school_chemistry,
high_school_computer_science, human_aging, human_sexuality,
machine_learning, medical_genetics, nutrition, professional_medicine,
security_studies, virology, anatomy, astronomy, conceptual_physics,
college_physics, electrical_engineering, high_school_physics) -- keeping
this pool would confound "training breadth helps" with "we just trained on
more bio/chem/cyber-adjacent content directly", which is exactly the effect
this experiment needs to rule out. MMLU has 57 subjects total; 23 excluded
above leaves the 34 subjects in NON_HAZARDOUS_SUBJECTS below, spanning
humanities, social science, law, business, and abstract math/logic --
clearly non-hazardous by construction. (Fixed 2026-08-29: an earlier
version of this list omitted college_mathematics/high_school_mathematics
from BOTH this exclusion list and NON_HAZARDOUS_SUBJECTS below -- an
oversight with no hazard rationale, not a deliberate exclusion; caught by
an adversarial code review. The MMLU-enrichment training run already in
progress as of this fix used the earlier 32-subject list, not this
34-subject one -- see project memory for the exact subject count that
specific run actually used.)
"""
import json
import urllib.request
from dataclasses import dataclass
from typing import List

from src.data.net import fetch_rows_with_disk_cache, urlopen_with_retry

DATASET = "cais/mmlu"
SPLIT = "test"
ROWS_URL_TEMPLATE = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=cais%2Fmmlu&config={subject}&split=test"
    "&offset={offset}&length={length}"
)
PAGE_SIZE = 100

CHOICE_LETTERS = ("A", "B", "C", "D")

# Curated, deliberately non-hazardous subset of MMLU's 57 subjects -- see
# module docstring for the exclusion rationale. Order is arbitrary; kept
# alphabetical for readability.
NON_HAZARDOUS_SUBJECTS = (
    "abstract_algebra",
    "business_ethics",
    "college_mathematics",
    "econometrics",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "management",
    "marketing",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_psychology",
    "public_relations",
    "sociology",
    "us_foreign_policy",
    "world_religions",
)


@dataclass(frozen=True)
class MMLUItem:
    question_id: str
    question: str
    choices: List[str]
    answer_index: int
    answer_letter: str
    subject: str


def fetch_raw_rows_for_subject(subject: str, timeout: float = 30.0) -> List[dict]:
    """Disk-cached per subject (data/mmlu_cache_<subject>.json) -- see
    net.py's fetch_rows_with_disk_cache docstring for why (session-wide HF
    rate-limiting, 2026-08-29). Caching per-subject (not the whole combined
    pool) means it's reusable across different NON_HAZARDOUS_SUBJECTS list
    versions -- a subject fetched once stays cached regardless of which
    subject SET later requests it."""
    return fetch_rows_with_disk_cache(
        f"data/mmlu_cache_{subject}.json", lambda: _fetch_raw_rows_for_subject_live(subject, timeout)
    )


def _fetch_raw_rows_for_subject_live(subject: str, timeout: float = 30.0) -> List[dict]:
    """Paginates one MMLU subject config via the public datasets-server REST
    API -- same pagination style as domain_b.py's fetch_raw_rows, just
    parameterized by subject/config instead of a single fixed config."""
    rows: List[dict] = []
    offset = 0
    while True:
        url = ROWS_URL_TEMPLATE.format(subject=subject, offset=offset, length=PAGE_SIZE)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        payload = json.loads(urlopen_with_retry(req, timeout).decode("utf-8"))
        rows.extend(entry["row"] for entry in payload["rows"])
        offset += PAGE_SIZE
        if offset >= payload["num_rows_total"]:
            break
    return rows


def fetch_raw_rows(subjects=NON_HAZARDOUS_SUBJECTS, timeout: float = 30.0) -> List[dict]:
    """Fetches and concatenates raw rows across all requested subjects, each
    row carrying its own 'subject' field (native to the MMLU schema) so
    load_mmlu can stamp it onto MMLUItem without extra bookkeeping. A small
    delay between LIVE (non-cached) subject fetches avoids re-tripping HF
    datasets-server rate limiting -- a single sequential process fetching
    all subjects back-to-back with no delay still hit a 429 partway through
    (2026-08-29), confirming this is a tight per-minute-style limit, not
    just a concurrency issue."""
    import os
    import time
    rows: List[dict] = []
    for subject in subjects:
        cache_path = f"data/mmlu_cache_{subject}.json"
        already_cached = os.path.exists(cache_path)
        rows.extend(fetch_raw_rows_for_subject(subject, timeout))
        if not already_cached:
            time.sleep(20)
    return rows


def load_mmlu(rows: List[dict]) -> List[MMLUItem]:
    """Parses raw {question, subject, choices, answer} rows into MMLUItem.
    Kept separate from fetch_raw_rows so parsing is unit-testable without
    network, same split as domain_b.py/domain_c.py."""
    items = []
    for i, row in enumerate(rows):
        answer_index = row["answer"]
        items.append(
            MMLUItem(
                question_id=f"mmlu-{i}",
                question=row["question"],
                choices=list(row["choices"]),
                answer_index=answer_index,
                answer_letter=CHOICE_LETTERS[answer_index],
                subject=row["subject"],
            )
        )
    return items
