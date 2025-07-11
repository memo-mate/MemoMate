import asyncio
from collections.abc import Callable
from threading import Thread
from typing import Any

import confluent_kafka
from confluent_kafka import KafkaError, KafkaException, Message


class AIOProducer:
    def __init__(self, configs: dict[str, Any], loop: asyncio.AbstractEventLoop | None = None):
        self._loop = loop or asyncio.get_event_loop()
        self._producer = confluent_kafka.Producer(configs)
        self._cancelled = False
        self._poll_thread = Thread(target=self._poll_loop)
        self._poll_thread.start()

    def __enter__(self) -> "AIOProducer":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _poll_loop(self):
        while not self._cancelled:
            self._producer.poll(0.1)

    def close(self):
        self._cancelled = True
        self._poll_thread.join()

    def produce(
        self,
        topic: str,
        value: Any,
        on_delivery: Callable[[KafkaError | None, Message], None] | None = None,
    ) -> asyncio.Future[Message]:
        """
        A produce method in which delivery notifications are made available
        via both the returned future and on_delivery callback (if specified).
        """
        result = self._loop.create_future()

        def ack(err: KafkaError | None, msg: Message):
            if err:
                self._loop.call_soon_threadsafe(result.set_exception, KafkaException(err))
            else:
                self._loop.call_soon_threadsafe(result.set_result, msg)
            if on_delivery:
                self._loop.call_soon_threadsafe(on_delivery, err, msg)

        self._producer.produce(topic, value, on_delivery=ack)
        return result


# demo 1
# @app.post("/items1")
# async def create_item1(item: Item):
#     try:
#         result = await aio_producer.produce("items", item.name)
#         return {"timestamp": result.timestamp()}
#     except KafkaException as ex:
#         raise HTTPException(status_code=500, detail=ex.args[0].str())


# demo 2
# @app.post("/items2")
# async def create_item2(item: Item):
#     try:

#         def ack(err, msg): ...

#         aio_producer.produce("items", item.name, on_delivery=ack)
#         return {"timestamp": time()}
#     except KafkaException as ex:
#         raise HTTPException(status_code=500, detail=ex.args[0].str())
