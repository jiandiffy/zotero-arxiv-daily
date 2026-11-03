"""Bounded retries for read-only arXiv requests; never retry email delivery."""
import random
import socket
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import feedparser
from loguru import logger


def read_feed(url):
    started = time.monotonic()
    for attempt in range(10):
        try:
            request = Request(url, headers={"User-Agent": "zotero-arxiv-daily/1.0 (daily academic digest)", "Accept": "application/atom+xml"})
            with urlopen(request, timeout=45) as response:
                payload = response.read(20_000_001)
            if len(payload) > 20_000_000:
                raise ValueError("arXiv feed exceeds 20 MB")
            feed = feedparser.parse(payload)
            if feed.bozo or not feed.version or "title" not in feed.feed:
                raise ValueError("arXiv returned an invalid feed (possibly an HTML error page)")
            if "error" in feed.feed.title.lower() or any(e.get("id", "").endswith("/errors") for e in feed.entries):
                raise ValueError("arXiv rejected the query")
            return feed
        except (HTTPError, URLError, TimeoutError, ConnectionError, socket.timeout) as exc:
            if isinstance(exc, HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}:
                raise
            # Certificate and authentication failures require diagnosis, not retries.
            if isinstance(exc, URLError) and "CERTIFICATE_VERIFY_FAILED" in str(exc.reason):
                raise
            delay = [2, 4, 8, 16, 30, 60][min(attempt, 5)] + random.uniform(0, 1)
            if isinstance(exc, HTTPError) and exc.headers.get("Retry-After"):
                retry_after = exc.headers["Retry-After"]
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    delay = max(delay, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
            if attempt == 9 or time.monotonic() - started + delay >= 900:
                raise RuntimeError("arXiv read retry budget exhausted; retrieval failed, not 'no papers'") from exc
            logger.warning(f"arXiv connection failed ({type(exc).__name__}); retry {attempt + 2}/10 in {delay:.0f}s")
            while delay > 0:
                interval = min(delay, 60)
                time.sleep(interval)
                delay -= interval
    raise RuntimeError("arXiv retrieval failed")
