"""Console commands for indexing and evaluation."""

import argparse
import json
import logging
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings
from .dataset import load_corpus, load_evaluation_data
from .evaluation import evaluate, frozen_depth, markdown_report
from .indexing import index_corpus, make_client, make_dense_embeddings, make_sparse_embeddings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("scifact_retriever")
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False


def report_failure(event: str, error: Exception) -> int:
    logging.getLogger(__name__).error(
        event,
        extra={"fields": {"error": str(error), "error_type": type(error).__name__}},
        exc_info=logging.getLogger(__name__).isEnabledFor(logging.DEBUG),
    )
    return 1


def ingest_main() -> int:
    parser = argparse.ArgumentParser(description="Index the pinned BEIR SciFact corpus in Qdrant.")
    parser.parse_args()
    configure_logging("INFO")
    try:
        settings = Settings()
        logging.getLogger("scifact_retriever").setLevel(settings.log_level)
        from torch import set_num_threads

        set_num_threads(settings.torch_threads)
        with closing(make_client(settings)) as client:
            result = index_corpus(
                load_corpus(),
                client,
                settings,
                make_dense_embeddings(settings),
                make_sparse_embeddings(settings),
            )
        logging.getLogger(__name__).info(
            "ingestion_complete",
            extra={
                "fields": {
                    "collection": settings.qdrant_collection,
                    "indexed": result.indexed,
                    "skipped": result.skipped,
                }
            },
        )
    except Exception as error:
        return report_failure("ingestion_failed", error)
    return 0


def evaluate_main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark retrieval against BEIR SciFact qrels.")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--candidate-depths", type=int, nargs="+", help="Development depths to try")
    parser.add_argument("--from-dev", type=Path, help="Development JSON report to use for test")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON path")
    args = parser.parse_args()
    if args.split == "test" and (args.from_dev is None or args.candidate_depths is not None):
        parser.error("test requires --from-dev and accepts no --candidate-depths")
    if args.split == "dev" and args.from_dev is not None:
        parser.error("--from-dev is only available for test")
    if args.from_dev and args.output.resolve() == args.from_dev.resolve():
        parser.error("test and development reports require different paths")
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")

    configure_logging("INFO")
    try:
        settings = Settings()
        logging.getLogger("scifact_retriever").setLevel(settings.log_level)
        depths = args.candidate_depths or [settings.search.rerank_candidates]
        dev_report = None
        if args.from_dev:
            dev_report = json.loads(args.from_dev.read_text(encoding="utf-8"))
            depths = [frozen_depth(dev_report, settings)]
        from torch import set_num_threads

        set_num_threads(settings.torch_threads)
        with closing(make_client(settings)) as client:
            if dev_report is not None and dev_report.get("qdrant_version") != client.info().version:
                raise ValueError("Qdrant server version differs from the development benchmark")
            data = load_evaluation_data(args.split)
            report = evaluate(data, args.split, settings, client, depths)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        summary = markdown_report(report)
        args.output.with_suffix(".md").write_text(summary, encoding="utf-8")
        print(summary, end="")
    except Exception as error:
        return report_failure("evaluation_failed", error)
    return 0
