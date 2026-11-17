"""Print evidence for a scientific claim using the configured retriever."""

import argparse
from contextlib import closing

from torch import set_num_threads

from scifact_retriever import Settings, build_retriever
from scifact_retriever.indexing import make_client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claim", help="Scientific claim to retrieve evidence for")
    args = parser.parse_args()

    settings = Settings()
    set_num_threads(settings.torch_threads)
    with closing(make_client(settings)) as client:
        retriever = build_retriever(settings, client=client)
        for number, doc in enumerate(retriever.invoke(args.claim), 1):
            metadata = doc.metadata
            scores = f"{metadata['retrieval_strategy']}={metadata['retrieval_score']:.4f}"
            if "rerank_score" in metadata:
                scores += f", rerank={metadata['rerank_score']:.4f}"
            print(f"{number}. {metadata['title']} (SciFact {doc.id}) [{scores}]")
            print(f"   {doc.page_content}\n")


if __name__ == "__main__":
    main()
