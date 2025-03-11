import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlmodel import Session

from app import crud, models
from app.core.db import engine


def create_user_data() -> None:
    with Session(engine) as session:
        for i in range(10):
            crud.create_user(
                session=session,
                user_create=models.UserCreate(
                    email=f"demo_{i}@example.com", password=secrets.token_hex(16), username=f"demo_{i}"
                ),
            )


def create_demo_data() -> None:
    # 多线程创建用户数据
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     executor.map(create_user_data, range(10))
    #     executor.map(create_order_data, range(10))
    # 将创建用户和创建订单放在线程池中
    futures = [
        create_user_data,
    ]
    with ThreadPoolExecutor(max_workers=len(futures)) as executor:
        completed = [executor.submit(f) for f in futures]
        for future in as_completed(completed):
            if future.done():
                print(future.result())
            else:
                print(future.exception())


if __name__ == "__main__":
    create_demo_data()
