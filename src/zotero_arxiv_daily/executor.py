from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm
from .delivery import read_history, record_attempt, record_delivery, paper_key, today, summary


def normalize_include_path_patterns(include_path: list[str] | ListConfig | None) -> list[str] | None:
    if include_path is None:
        return None

    if not isinstance(include_path, (list, ListConfig)):
        raise TypeError(
            "config.zotero.include_path must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in include_path):
        raise TypeError("config.zotero.include_path must contain only glob pattern strings.")

    return list(include_path)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_include_path_patterns(config.zotero.include_path)
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if not self.include_path_patterns:
            return corpus
        new_corpus = []
        logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
        for c in corpus:
            match_results = [
                glob_match(path, pattern)
                for path in c.paths
                for pattern in self.include_path_patterns
            ]
            if any(match_results):
                new_corpus.append(c)
        samples = random.sample(new_corpus, min(5, len(new_corpus)))
        samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
        logger.info(f"Selected {len(new_corpus)} zotero papers:\n{samples}\n...")
        return new_corpus

    
    def run(self):
        history_path = self.config.executor.get("history_path")
        history = read_history(history_path) if history_path else {"sent": {}}
        if history.get("pending"):
            raise RuntimeError("Previous SMTP delivery outcome is unknown. Check the recipient mailbox and resolve the pending delivery checkpoint before retrying.")
        if history_path and history.get("last_sent_date") == today():
            logger.info("Today's digest was already sent; skipping duplicate delivery")
            summary(f"Already delivered on {today()} (Asia/Taipei); no duplicate email sent.")
            return
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            raise RuntimeError("No Zotero papers with abstracts match the configured collections; check Zotero access and include_path")
        summary(f"Zotero reference papers: **{len(corpus)}**")
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        summary(f"Retrieved candidates: **{len(all_papers)}**")
        unique_papers = {paper_key(p.url): p for p in all_papers}
        all_papers = [p for key, p in unique_papers.items() if key not in history["sent"]]
        minimum = int(self.config.executor.get("min_paper_num", 0))
        summary(f"Unsent unique candidates: **{len(all_papers)}**; requested: **{self.config.executor.max_paper_num}**")
        if len(all_papers) < minimum:
            raise RuntimeError(f"Only {len(all_papers)} unsent papers available; required {minimum}. No email sent; increase the recent search window/pool.")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            if len(reranked_papers) < minimum:
                raise RuntimeError(f"Reranker returned fewer than {minimum} papers; no email sent")
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        if history_path:
            record_attempt(history_path, history, reranked_papers)
        send_email(self.config, email_content)
        if history_path:
            record_delivery(history_path, history, reranked_papers)
        logger.info("Email sent successfully")
        summary(f"SMTP accepted **{len(reranked_papers)} papers** on {today()} (Asia/Taipei). Delivery history saved.")
