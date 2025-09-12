import uuid
from pathlib import Path

import orjson
import pytest

from app.consumer_start import DocumentParserDict
from app.core import consts
from app.enums.queue import QueueTopic
from app.utils.aio_producer import AIOProducer


class TestDocumentParser:
    @pytest.fixture
    def file_path(self):
        source_path = consts.MD_SOURCE_DIR
        return list(source_path.glob("*.md"))

    async def test_parse_document(self, file_path: list[Path | str], producer: AIOProducer):
        for p in file_path:
            data: DocumentParserDict = {
                "id": str(uuid.uuid4()),
                "file_path": str(p.absolute()),
                "task_type": QueueTopic.FILE_PARSING_TASK,
                "retry_count": 0,
            }
            producer.produce(QueueTopic.FILE_PARSING_TASK, orjson.dumps(data))
