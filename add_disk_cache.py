"""Run on the pod to add disk-caching to domain_b.py, domain_c.py, and
mmlu.py's raw HF fetch functions -- avoids re-editing via fragile inline
shell heredocs (backtick/quote interpolation issues)."""
import re

# --- domain_b.py ---
path = "src/data/domain_b.py"
src = open(path).read()
if "_fetch_raw_rows_live" not in src:
    old = '''def fetch_raw_rows(timeout: float = 30.0) -> List[dict]:
    """Paginates the public Hugging Face datasets-server REST API (100
    rows/page) -- no auth, no `datasets` library dependency, same
    dependency-light style as domain_a.py's direct CSV fetch."""
    rows: List[dict] = []'''
    new = '''def fetch_raw_rows(timeout: float = 30.0) -> List[dict]:
    """Disk-cached (data/domain_b_cache.json) -- see net.py's
    fetch_rows_with_disk_cache docstring for why (session-wide HF
    rate-limiting, 2026-08-29)."""
    return fetch_rows_with_disk_cache("data/domain_b_cache.json", lambda: _fetch_raw_rows_live(timeout))


def _fetch_raw_rows_live(timeout: float = 30.0) -> List[dict]:
    """Paginates the public Hugging Face datasets-server REST API (100
    rows/page) -- no auth, no `datasets` library dependency, same
    dependency-light style as domain_a.py's direct CSV fetch."""
    rows: List[dict] = []'''
    assert old in src, "domain_b.py pattern not found"
    src = src.replace(old, new)
    open(path, "w").write(src)
    print("domain_b.py patched")
else:
    print("domain_b.py already patched")

# --- domain_c.py ---
path = "src/data/domain_c.py"
src = open(path).read()
if "_fetch_raw_rows_live" not in src:
    src = src.replace(
        "from src.data.net import urlopen_with_retry",
        "from src.data.net import fetch_rows_with_disk_cache, urlopen_with_retry",
    )
    old = '''def fetch_raw_rows(timeout: float = 30.0) -> List[dict]:
    """Paginates the public Hugging Face datasets-server REST API (100
    rows/page) -- no auth, no `datasets` library dependency, same pattern
    as domain_a.py/domain_b.py."""
    rows: List[dict] = []'''
    new = '''def fetch_raw_rows(timeout: float = 30.0) -> List[dict]:
    """Disk-cached (data/domain_c_cache.json) -- see net.py's
    fetch_rows_with_disk_cache docstring for why (session-wide HF
    rate-limiting, 2026-08-29)."""
    return fetch_rows_with_disk_cache("data/domain_c_cache.json", lambda: _fetch_raw_rows_live(timeout))


def _fetch_raw_rows_live(timeout: float = 30.0) -> List[dict]:
    """Paginates the public Hugging Face datasets-server REST API (100
    rows/page) -- no auth, no `datasets` library dependency, same pattern
    as domain_a.py/domain_b.py."""
    rows: List[dict] = []'''
    assert old in src, "domain_c.py pattern not found"
    src = src.replace(old, new)
    open(path, "w").write(src)
    print("domain_c.py patched")
else:
    print("domain_c.py already patched")

# --- mmlu.py --- cache per-subject (fetch_raw_rows_for_subject), reusable
# across different subject-list configurations (e.g. the fixed vs. corrected
# NON_HAZARDOUS_SUBJECTS list) since each subject's own file is independent.
path = "src/data/mmlu.py"
src = open(path).read()
if "_fetch_raw_rows_for_subject_live" not in src:
    src = src.replace(
        "from src.data.net import urlopen_with_retry",
        "from src.data.net import fetch_rows_with_disk_cache, urlopen_with_retry",
    )
    old = '''def fetch_raw_rows_for_subject(subject: str, timeout: float = 30.0) -> List[dict]:
    """Paginates one MMLU subject config via the public datasets-server REST
    API -- same pagination style as domain_b.py's fetch_raw_rows, just
    parameterized by subject/config instead of a single fixed config."""
    rows: List[dict] = []'''
    new = '''def fetch_raw_rows_for_subject(subject: str, timeout: float = 30.0) -> List[dict]:
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
    rows: List[dict] = []'''
    assert old in src, "mmlu.py pattern not found"
    src = src.replace(old, new)
    open(path, "w").write(src)
    print("mmlu.py patched")
else:
    print("mmlu.py already patched")
