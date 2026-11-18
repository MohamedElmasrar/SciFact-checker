import pytest
from pydantic import ValidationError

from scifact_retriever.dataset import benchmark_data, corpus_documents, load_evaluation_data


def test_corpus_preserves_abstract_title_id_and_metadata():
    row = {"_id": "123", "title": "Study", "text": "First sentence. Second sentence."}
    [doc] = corpus_documents([{**row, "metadata": {"journal": "JA"}}])
    assert doc.id == "123"
    assert doc.page_content == row["text"]
    assert doc.metadata == {"doc_id": "123", "title": "Study", "journal": "JA"}


def test_corpus_rejects_invalid_and_duplicate_documents():
    row = {"_id": "1", "title": "Study", "text": "Abstract"}
    with pytest.raises(ValidationError, match="non-whitespace"):
        list(corpus_documents([{**row, "text": " "}]))
    with pytest.raises(ValueError, match="Duplicate corpus ID"):
        list(corpus_documents([row, row]))


def test_benchmark_selects_judged_claims_and_normalizes_ids():
    queries = [{"_id": "2", "text": "Other split"}, {"_id": "1", "text": "Judged claim"}]
    data = benchmark_data(queries, [{"query-id": 1, "corpus-id": 12, "score": 1}])
    assert data.queries == {"1": "Judged claim"}
    assert data.qrels == {"1": {"12": 1}}


def test_benchmark_rejects_qrels_referencing_missing_claim():
    with pytest.raises(ValueError, match="missing claims"):
        benchmark_data([], [{"query-id": 1, "corpus-id": 12, "score": 1}])


@pytest.mark.parametrize(("split", "source"), [("dev", "train"), ("test", "test")])
def test_loader_only_opens_requested_qrels_split(monkeypatch, split, source):
    calls = []

    def load_dataset(path, *args, **kwargs):
        calls.append((path, kwargs))
        if args == ("queries",):
            return [{"_id": "1", "text": "Judged claim"}]
        return [{"query-id": 1, "corpus-id": 12, "score": 1}]

    monkeypatch.setattr("datasets.load_dataset", load_dataset)
    assert load_evaluation_data(split).queries == {"1": "Judged claim"}
    path, options = calls[1]
    assert path == "csv"
    assert options["split"] == source
    assert list(options["data_files"]) == [source]
    assert options["data_files"][source].endswith(f"/{source}.tsv")
