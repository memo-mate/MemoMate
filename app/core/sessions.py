from app.utils.aio_producer import AIOProducer


class SessionFactory:
    aio_producer: AIOProducer | None = None

    @classmethod
    def get_aio_producer(cls) -> AIOProducer:
        if not cls.aio_producer:
            raise RuntimeError("AIOProducer not initialized")
        return cls.aio_producer
