path = "src/data/mmlu.py"
src = open(path).read()

old = '''def fetch_raw_rows(subjects=NON_HAZARDOUS_SUBJECTS, timeout: float = 30.0) -> List[dict]:
    """Fetches and concatenates raw rows across all requested subjects, each
    row carrying its own 'subject' field (native to the MMLU schema) so
    load_mmlu can stamp it onto MMLUItem without extra bookkeeping."""
    rows: List[dict] = []
    for subject in subjects:
        rows.extend(fetch_raw_rows_for_subject(subject, timeout))
    return rows'''

new = '''def fetch_raw_rows(subjects=NON_HAZARDOUS_SUBJECTS, timeout: float = 30.0) -> List[dict]:
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
            time.sleep(3)
    return rows'''

assert old in src, "pattern not found"
src = src.replace(old, new)
open(path, "w").write(src)
print("patched fetch_raw_rows with inter-subject delay")
