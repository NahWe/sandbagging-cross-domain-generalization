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
