import hashlib
import os
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import ProducerDep, SessionDep, get_current_user
from app.core.config import settings
from app.enums.queue import QueueTopic
from app.models.task import FileParsingTask
from app.schemas.paser import FileParsingTaskCreate, FileParsingTaskUploadParams

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.post(
    "/upload",
    response_model=FileParsingTask,
    summary="上传文件并等待解析",
    description="上传文件并等待解析",
)
def create_file_parsing_task(
    task: FileParsingTaskUploadParams,
    session: SessionDep,
    producer: ProducerDep,
    file: UploadFile = File(...),
) -> Any:
    file_path = f"{settings.UPLOAD_DIR_PATH}/{file.filename}"
    with open(file_path, "wb") as f:
        file_buffer = file.file.read()
        f.write(file_buffer)
        # 解析文件md5
        file_md5 = hashlib.md5(file_buffer).hexdigest()

    # 解析文件大小
    file_size = os.path.getsize(file_path)

    task = FileParsingTaskCreate(
        title=task.title,
        file_path=file_path,
        file_name=file.filename,
        file_size=file_size,
        file_md5=file_md5,
    )

    task = FileParsingTask.model_validate(task)
    session.add(task)
    session.commit()
    session.refresh(task)

    # 发送消息到kafka
    producer.produce(QueueTopic.FILE_PARSING_TASK, task.model_dump())
    return task


@router.get(
    "/{task_id}",
    response_model=FileParsingTask,
    summary="获取文件解析任务信息",
    description="获取文件解析任务信息",
)
def get_file_parsing_task(task_id: int, session: SessionDep) -> Any:
    task = session.get(FileParsingTask, task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return task


@router.delete(
    "/{task_id}",
    summary="删除文件解析任务",
    description="删除文件解析任务",
)
def delete_file_parsing_task(task_id: int, session: SessionDep) -> Any:
    task = session.get(FileParsingTask, task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    # TODO: 删除任务前，需要检查任务是否已经解析完成，如果未完成，则不能删除需要停止任务
    session.delete(task)
    session.commit()
    return {"message": "Task deleted successfully"}
