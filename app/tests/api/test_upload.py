import os
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

TEST_FILE_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 5 * 1024 * 1024  # 5MB
TEST_USER_ID = "pyy"
BASE_URL = "http://localhost:8000"  # 假设API运行在本地8000端口


def create_test_file() -> str:
    """创建一个测试用的临时文件"""
    with NamedTemporaryFile(delete=False) as f:
        f.write(os.urandom(TEST_FILE_SIZE))
        f.flush()
        print(f"临时文件路径: {f.name}")
        return f.name


def cleanup_test_file(file_path: str) -> None:
    """清理测试文件"""
    Path(file_path).unlink(missing_ok=True)
    print(f"临时文件已删除: {file_path}")


def get_upload_status(file_name: str) -> str:
    """获取上传状态并返回upload_id"""
    response = requests.post(
        f"{BASE_URL}/upload/status",
        json={
            "file_name": file_name,
            "file_size": TEST_FILE_SIZE,
            "user_id": TEST_USER_ID,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] in ["文件已存在上传记录", "已创建新上传任务"]
    assert data["data"]["file_size"] == TEST_FILE_SIZE
    assert data["data"]["chunk_size"] == CHUNK_SIZE
    return data["data"]["upload_id"]


def upload_file_chunks(test_file: str, upload_id: str) -> None:
    """上传文件分块"""
    with open(test_file, "rb") as f:
        for chunk_index in range(2):
            f.seek(chunk_index * CHUNK_SIZE)
            chunk_data = f.read(CHUNK_SIZE)

            response = requests.post(
                f"{BASE_URL}/upload/chunk",
                files={"chunk_file": ("chunk", chunk_data)},
                data={
                    "upload_id": upload_id,
                    "chunk_index": chunk_index,
                },
            )
            assert response.status_code == 200
            assert response.json()["data"]["success"] is True


def merge_file_chunks(upload_id: str, file_name: str) -> None:
    """合并文件分块"""
    response = requests.post(
        f"{BASE_URL}/upload/merge",
        json={
            "upload_id": upload_id,
            "file_name": file_name,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["success"] is True
    assert data["message"] == "分块合并成功"


def test_full_upload_flow() -> None:
    """完整测试上传流程"""
    print("\n开始完整上传流程测试...")

    test_file_name = f"test_file_{uuid.uuid4().hex[:8]}.txt"
    print(f"测试文件名: {test_file_name}")

    test_file_path = create_test_file()
    try:
        print("步骤1: 创建上传任务...")
        upload_id = get_upload_status(test_file_name)
        print(f"获取到上传ID: {upload_id}")

        print("步骤2: 上传文件分块...")
        upload_file_chunks(test_file_path, upload_id)
        print("文件分块上传完成")

        print("步骤3: 合并文件分块...")
        merge_file_chunks(upload_id, test_file_name)
        print("文件合并完成")

        print("完整上传流程测试通过!")
    finally:
        cleanup_test_file(test_file_path)
