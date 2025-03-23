from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

KAFKA_CONSUMER_PARSER_GROUP_ID = "paser"

MODEL_PATH = PROJECT_ROOT / "models"

# embedding 参数
BGE_MODEL_PATH = MODEL_PATH / "bge-m3"
BGE_MAX_TOKENS = 8192
