from langchain_community.vectorstores import LanceDB
from loguru import logger

def write_to_lancedb(texts, connection, embedding, source):
    vector_store = LanceDB(
        connection=connection,
        embedding=embedding
    )
    for text in texts:
        vector_store.add_texts(
            [text],
            metadatas=[{"source": source}]
        )
        logger.info(f"已写入: {text[:30]}...")