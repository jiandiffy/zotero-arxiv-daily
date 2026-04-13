from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import feedparser
from urllib.request import urlretrieve
from tqdm import tqdm
import os
import time
import random
from loguru import logger

PDF_EXTRACT_TIMEOUT = 180


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

        # 建议把这些参数做成可配置；没有配置时使用保守默认值
        self.request_delay = float(self.config.source.arxiv.get("delay_seconds", 6.0))
        self.batch_size = int(self.config.source.arxiv.get("batch_size", 10))
        self.max_batch_retries = int(self.config.source.arxiv.get("max_batch_retries", 6))
        self.skip_failed_batches = bool(self.config.source.arxiv.get("skip_failed_batches", True))

        # 复用同一个 client，避免每个 batch 新建连接/节流状态
        self.client = arxiv.Client(
            page_size=self.batch_size,
            delay_seconds=self.request_delay,
            # 库内重试保留少量；真正的强退避在外层做
            num_retries=2,
        )

    def _should_retry(self, exc: Exception) -> bool:
        """判断异常是否值得重试。"""
        if isinstance(exc, arxiv.HTTPError):
            return True

        msg = str(exc)
        retry_signals = [
            "HTTP 429",
            "Too Many Requests",
            "timed out",
            "unexpectedly empty",
        ]
        return any(s.lower() in msg.lower() for s in retry_signals)

    def _fetch_batch_with_backoff(self, batch_ids: list[str]) -> list[ArxivResult]:
        """
        对单个 id batch 做带指数退避的抓取。
        """
        if not batch_ids:
            return []

        last_exc = None
        for attempt in range(1, self.max_batch_retries + 1):
            try:
                search = arxiv.Search(
                    id_list=batch_ids,
                    max_results=len(batch_ids),
                )
                batch = list(self.client.results(search))

                # 某些情况下可能返回空结果；这里也作为异常处理掉，避免静默丢数据
                if len(batch) == 0:
                    raise RuntimeError(
                        f"arXiv API returned an unexpectedly empty batch for ids={batch_ids}"
                    )

                return batch

            except Exception as exc:
                last_exc = exc
                if not self._should_retry(exc):
                    raise

                # 指数退避 + 少量随机抖动，避免所有任务同时重试
                sleep_seconds = min(
                    120.0,
                    self.request_delay * (2 ** (attempt - 1))
                ) + random.uniform(0.0, 1.0)

                logger.warning(
                    f"Failed to fetch batch from arXiv "
                    f"(attempt {attempt}/{self.max_batch_retries}, "
                    f"batch_size={len(batch_ids)}): {type(exc).__name__}: {exc}. "
                    f"Sleeping {sleep_seconds:.1f}s before retry."
                )

                if attempt < self.max_batch_retries:
                    time.sleep(sleep_seconds)

        raise RuntimeError(
            f"Failed to fetch arXiv batch after {self.max_batch_retries} attempts: {batch_ids}"
        ) from last_exc

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        query = "+".join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)

        # 先从 RSS 拿最新 paper id
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        feed_title = getattr(feed.feed, "title", "")
        if "Feed error for query" in feed_title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")

        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}

        all_paper_ids = [
            entry.id.removeprefix("oai:arXiv.org:")
            for entry in feed.entries
            if entry.get("arxiv_announce_type", "new") in allowed_announce_types
        ]

        # 去重，避免 RSS 中重复 id 导致不必要请求
        all_paper_ids = list(dict.fromkeys(all_paper_ids))

        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        if not all_paper_ids:
            logger.warning("No arXiv paper ids found from RSS feed.")
            return []

        # 非常关键：RSS 和 arXiv API 都属于 legacy API，请求之间主动留出间隔
        logger.info(
            f"Fetched {len(all_paper_ids)} paper ids from RSS. "
            f"Sleeping {self.request_delay:.1f}s before API requests."
        )
        time.sleep(self.request_delay)

        raw_papers: list[ArxivResult] = []
        bar = tqdm(total=len(all_paper_ids), desc="Fetching arXiv metadata")

        for start in range(0, len(all_paper_ids), self.batch_size):
            batch_ids = all_paper_ids[start:start + self.batch_size]

            try:
                batch = self._fetch_batch_with_backoff(batch_ids)
                raw_papers.extend(batch)

            except Exception as exc:
                logger.error(
                    f"Failed to fetch metadata for batch starting at index {start}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if not self.skip_failed_batches:
                    bar.close()
                    raise

            finally:
                # 无论成功失败，都推进进度，表示该 batch 已处理
                bar.update(len(batch_ids))

        bar.close()
        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                full_text = pool.submit(extract_text_from_pdf, raw_paper).result(
                    timeout=PDF_EXTRACT_TIMEOUT
                )
        except TimeoutError:
            logger.warning(f"PDF extraction timed out for {raw_paper.title}")
            full_text = None

        if full_text is None:
            full_text = extract_text_from_tar(raw_paper)

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        if paper.pdf_url is None:
            logger.warning(f"No PDF URL available for {paper.title}")
            return None

        try:
            urlretrieve(paper.pdf_url, path)
        except Exception as e:
            logger.warning(f"Failed to download pdf for {paper.title}: {type(e).__name__}: {e}")
            return None

        try:
            full_text = extract_markdown_from_pdf(path)
        except Exception as e:
            logger.warning(f"Failed to extract full text of {paper.title} from pdf: {e}")
            full_text = None

        return full_text


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        source_url = paper.source_url()
        if source_url is None:
            logger.warning(f"No source URL available for {paper.title}")
            return None

        try:
            urlretrieve(source_url, path)
        except Exception as e:
            logger.warning(f"Failed to download source for {paper.title}: {type(e).__name__}: {e}")
            return None

        try:
            file_contents = extract_tex_code_from_tar(path, paper.entry_id)
            if "all" not in file_contents:
                logger.warning(
                    f"Failed to extract full text of {paper.title} from tar: Main tex file not found."
                )
                return None
            full_text = file_contents["all"]
        except Exception as e:
            logger.warning(f"Failed to extract full text of {paper.title} from tar: {e}")
            full_text = None

        return full_text
