"""Shared retry wrapper for this project's few network fetches (domain_a.py's
CSV, domain_b.py's paginated rows). Both hit public, unauthenticated
endpoints prone to occasional transient 5xx errors -- observed live: a 502
from datasets-server.huggingface.co -- and both get called enough times
across a full sweep (domain B alone: ~13 paginated requests x 20 seed/
condition runs) that treating a one-off blip as fatal would be needlessly
fragile.
"""
import random
import time
import urllib.error
import urllib.request
from typing import Union


def urlopen_with_retry(
    request: Union[str, urllib.request.Request], timeout: float, max_attempts: int = 6
) -> bytes:
    """Retries on 5xx (server-side, likely transient), 429 (rate limited --
    the opposite of a permanent error, it's the server explicitly asking
    for a slower retry), and on network-level errors (timeout, DNS,
    connection reset). Does NOT retry on other 4xx -- a bad URL or missing
    resource won't fix itself on the next attempt.

    429 confirmed live 2026-08-25: run_domain_b_eval.sh launches NUM_GPUS
    processes in parallel, each independently paginating the same ~13
    requests against datasets-server.huggingface.co at roughly the same
    moment -- collectively enough to trip the API's rate limit even
    though any single process alone wouldn't. max_attempts raised from 4
    to 6 and backoff gets random jitter (not just 2**attempt) specifically
    for this case: several processes retrying in perfect lockstep off the
    same deterministic backoff schedule would just collide again on the
    same retry, jitter spreads them out."""
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if (e.code != 429 and e.code < 500) or attempt == max_attempts - 1:
                raise
        except urllib.error.URLError:
            if attempt == max_attempts - 1:
                raise
        time.sleep(2**attempt + random.uniform(0, 1))


import json
import os


def fetch_rows_with_disk_cache(cache_path: str, fetch_fn):
    """Returns cache_path's JSON content if it exists, else calls
    fetch_fn() (a live network fetch of raw rows), writes the result to
    cache_path, and returns it. Added 2026-08-29 after a real, session-wide
    HF datasets-server rate-limit block (confirmed via direct curl: a
    blanket 429 on a single bare request, no Retry-After header) -- every
    data loader in this project (domain_a/b/c.py, mmlu.py) re-fetches the
    same fixed, model/seed-independent data from HF's live API on every
    single job start, with no caching, which is fine at low volume but
    exhausts the anonymous rate limit across a day of heavy experimentation
    (many training sweeps x many domains, each re-fetching the same rows).
    The fetched data is small (a few MB at most for any of this project's
    domains) against a 212TB-free shared network filesystem, so disk usage
    is not a real constraint here."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    rows = fetch_fn()
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(rows, f)
    return rows
