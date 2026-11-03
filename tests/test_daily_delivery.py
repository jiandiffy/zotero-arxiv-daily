from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.delivery import paper_key, read_history, today
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.protocol import Paper, CorpusPaper
from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
from zotero_arxiv_daily import network


def make_paper(i):
    return Paper(source="arxiv", title=f"Robot {i}", abstract="robot learning", authors=[],
                 url=f"https://arxiv.org/abs/2611.{i:05}v1", pdf_url=f"https://arxiv.org/pdf/2611.{i:05}")


def make_executor(tmp_path, count=15):
    e = Executor.__new__(Executor)
    e.config = OmegaConf.create({"executor": {"history_path": str(tmp_path / "sent.json"),
        "min_paper_num": 10, "max_paper_num": 10, "send_empty": False}, "llm": {}})
    e.fetch_zotero_corpus = lambda: [CorpusPaper("robot", "robot learning", datetime.now(), [])]
    e.filter_corpus = lambda corpus: corpus
    e.retrievers = {"arxiv": SimpleNamespace(retrieve_papers=lambda: [make_paper(i) for i in range(count)])}
    e.reranker = SimpleNamespace(rerank=lambda papers, corpus: papers)
    e.openai_client = None
    return e


def test_sends_exactly_ten_and_skips_same_day(tmp_path, monkeypatch):
    e = make_executor(tmp_path)
    sent = []
    monkeypatch.setattr(Paper, "generate_tldr", lambda *args: None)
    monkeypatch.setattr(Paper, "generate_affiliations", lambda *args: None)
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda *args: sent.append(args))
    e.run()
    state = read_history(e.config.executor.history_path)
    assert len(state["sent"]) == 10
    assert state["last_sent_date"] == today()
    e.run()
    assert len(sent) == 1


def test_does_not_send_or_record_when_fewer_than_ten(tmp_path, monkeypatch):
    e = make_executor(tmp_path, 9)
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda *args: pytest.fail("must not send"))
    with pytest.raises(RuntimeError, match="required 10"):
        e.run()
    assert not (tmp_path / "sent.json").exists()


def test_failed_smtp_does_not_mark_delivered(tmp_path, monkeypatch):
    e = make_executor(tmp_path)
    monkeypatch.setattr(Paper, "generate_tldr", lambda *args: None)
    monkeypatch.setattr(Paper, "generate_affiliations", lambda *args: None)
    def fail(*args):
        raise TimeoutError("SMTP outcome unknown")
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", fail)
    with pytest.raises(TimeoutError):
        e.run()
    state = read_history(tmp_path / "sent.json")
    assert not state["sent"]
    assert "pending" in state
    with pytest.raises(RuntimeError, match="outcome is unknown"):
        e.run()


def test_excludes_sent_versions_and_duplicate_candidates(tmp_path, monkeypatch):
    e = make_executor(tmp_path, 15)
    path = tmp_path / "sent.json"
    path.write_text('{"sent":{"2611.00000":"2026-01-01"}}')
    candidates = [make_paper(i) for i in range(15)] + [make_paper(1)]
    e.retrievers["arxiv"].retrieve_papers = lambda: candidates
    monkeypatch.setattr(Paper, "generate_tldr", lambda *args: None)
    monkeypatch.setattr(Paper, "generate_affiliations", lambda *args: None)
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda *args: None)
    e.run()
    state = read_history(path)
    assert len(state["sent"]) == 11
    assert state["sent"]["2611.00000"] == "2026-01-01"
    assert paper_key("http://arxiv.org/abs/2611.00000v2") == "2611.00000"


def test_recent_api_metadata_and_cross_list(monkeypatch):
    config = OmegaConf.create({"source": {"arxiv": {"category": ["cs.RO"], "include_cross_list": True,
        "lookback_days": 30, "candidates_per_category": 200, "metadata_only": True}}, "executor": {}})
    entry = SimpleNamespace(id="https://arxiv.org/abs/2611.00001v1", title="Robot\n Learning",
        summary="Robot learning abstract", published="2026-11-02T12:00:00Z", updated="2026-11-02T12:00:00Z",
        authors=[SimpleNamespace(name="A")], tags=[SimpleNamespace(term="cs.RO")],
        arxiv_primary_category={"term": "cs.CV"}, links=[])
    urls = []
    def feed(url):
        urls.append(url)
        return SimpleNamespace(entries=[entry, entry])
    monkeypatch.setattr("zotero_arxiv_daily.retriever.arxiv_retriever.read_feed", feed)
    retriever = ArxivRetriever(config)
    raw = retriever._retrieve_recent_papers()
    assert len(raw) == 1
    paper = retriever.convert_to_paper(raw[0])
    assert paper.published_date == "2026-11-02"
    assert paper.full_text is None
    assert "cat%3Acs.RO" in urls[0]
    config.source.arxiv.include_cross_list = False
    with pytest.raises(RuntimeError, match="zero papers"):
        retriever._retrieve_recent_papers()


def test_http_403_is_not_retried(monkeypatch):
    def fail(*args, **kwargs):
        raise HTTPError("https://arxiv.org", 403, "forbidden", {}, None)
    monkeypatch.setattr(network, "urlopen", fail)
    monkeypatch.setattr(network.time, "sleep", lambda n: pytest.fail("do not retry auth failures"))
    with pytest.raises(HTTPError):
        network.read_feed("https://arxiv.org")


def test_transient_failure_is_bounded(monkeypatch):
    calls, sleeps = [], []
    def fail(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://arxiv.org", 503, "unavailable", {}, None)
    monkeypatch.setattr(network, "urlopen", fail)
    monkeypatch.setattr(network.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="retry budget exhausted"):
        network.read_feed("https://arxiv.org")
    assert len(calls) == 10
    assert max(sleeps) <= 60
