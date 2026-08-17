"""Shared retry wrapper for this project's few network fetches (domain_a.py's
CSV, domain_b.py's paginated rows). Both hit public, unauthenticated
endpoints prone to occasional transient 5xx errors -- observed live: a 502
from datasets-server.huggingface.co -- and both get called enough times
across a full sweep (domain B alone: ~13 paginated requests x 20 seed/
condition runs) that treating a one-off blip as fatal would be needlessly
fragile.
"""
import time
import urllib.error
import urllib.request
from typing import Union


def urlopen_with_retry(
    request: Union[str, urllib.request.Request], timeout: float, max_attempts: int = 4
) -> bytes:
    """Retries on 5xx (server-side, likely transient) and on network-level
    errors (timeout, DNS, connection reset). Does NOT retry on 4xx --
    a bad URL or missing resource won't fix itself on the next attempt.
    Exponential backoff (1s, 2s, 4s, ...) between attempts."""
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == max_attempts - 1:
                raise
        except urllib.error.URLError:
            if attempt == max_attempts - 1:
                raise
        time.sleep(2**attempt)
