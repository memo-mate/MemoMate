from pathlib import Path

# 全局配置
MODEL_PATH = Path(r"D:\LLM\bge-large-zh-v1.5")
DB_PATH = "app/database"
TABLE_NAME = "pyy"

# 文本分割配置
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 2000,
    "chunk_overlap": 100,
    "separators": ["\n\n", "\n", "。", "！", "？"],
}

# 嵌入模型配置
EMBEDDING_CONFIG = {
    "model_kwargs": {"device": "cpu"},
    "encode_kwargs": {"normalize_embeddings": True},
}
