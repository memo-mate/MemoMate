from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"


MODEL_PATH = PROJECT_ROOT / "models"

# embedding 参数
BGE_MODEL_PATH = MODEL_PATH / "bge-m3"
BGE_MAX_TOKENS = 8192

# markdown source
MD_SOURCE_DIR = DATA_DIR / "md_source"

# kafka consumer group id
KAFKA_CONSUMER_PARSER_GROUP_ID = "document-parser-group"
