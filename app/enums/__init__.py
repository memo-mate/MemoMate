from enum import StrEnum

from .embedding import *  # noqa: F403
from .queue import *  # noqa: F403


class AccessRole(StrEnum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    SUPER = "super"


class HistoryMessageType(StrEnum):
    HUMAN = "human"
    AI = "ai"
