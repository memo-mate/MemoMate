from structlog.stdlib import BoundLogger

logger: BoundLogger

def setup_logging(json_logs: bool = False, log_level: str = "INFO") -> None: ...
