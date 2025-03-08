from rich import inspect, print  # noqa

from app.rag.embedding import load_documents, vector_store
from app.rag.reasoning import process_rag


def main() -> None:
    documents = load_documents("./data/Miner2PdfAndWord_Markitdown2Excel")
    # 向量存储
    vector_store.add_documents(documents)

    answer = process_rag(
        query="马子坤是谁？",
        history=[],
        is_web_search=False,
    )
    print(answer)


if __name__ == "__main__":
    main()
