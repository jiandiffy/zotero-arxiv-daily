"""Durable delivery history and concise, secret-free Actions diagnostics."""
import json
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def paper_key(url):
    return re.sub(r"v\d+$", "", url.rstrip("/").split("/abs/")[-1])


def today():
    return datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat()


def read_history(path):
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else {"sent": {}, "last_sent_date": None}


def write_history(path, history):
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(".tmp")
    temporary.write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(file)


def record_attempt(path, history, papers):
    history["pending"] = {"date": today(), "papers": [paper_key(p.url) for p in papers]}
    write_history(path, history)


def record_delivery(path, history, papers):
    history["last_sent_date"] = today()
    history["sent"].update({paper_key(p.url): today() for p in papers})
    history.pop("pending", None)
    write_history(path, history)


def summary(message):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(message + "\n\n")
