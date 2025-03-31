from enum import StrEnum


class EmbeddingAPIType(StrEnum):
    LOCAL = "local"
    OPENAI = "openai"


class EmbeddingDriverEnum(StrEnum):
    MAC = "mps"
    CPU = "cpu"
    CUDA = "cuda"
    NPU = "npu"
