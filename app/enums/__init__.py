from enum import StrEnum


class AccessRole(StrEnum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    SUPER = "super"


class HistoryMessageType(StrEnum):
    HUMAN = "human"
    AI = "ai"
