import datetime
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from sqlmodel import select

from app.api.deps import SessionDep
from app.core.config import settings
from app.enums.upload import FileUploadState
from app.models.upload import FileMatedata, UploadChunk
from app.schemas.upload import (
    MergeChunksRequest,
    MergeChunksResponse,
    UploadChunkResponse,
    UploadCreateParams,
    UploadStatusRequest,
    UploadStatusResponse,
)

# router = APIRouter(dependencies=[Depends(get_current_user)])
router = APIRouter()


@router.post(
    "/status",
    response_model=UploadStatusResponse,
    summary="大文件上传状态查询",
    description="大文件上传状态查询",
)
def upload_status(
    request: UploadStatusRequest,
    session: SessionDep,
) -> UploadStatusResponse:
    statement = select(FileMatedata).where(
        (FileMatedata.file_name == request.file_name) & (FileMatedata.uploader_id == request.user_id)
    )
    upload_file = session.exec(statement).first()
    print(request)

    if upload_file:
        chunk_list = [chunk.chunk_index for chunk in upload_file.chunks]
        return UploadStatusResponse(
            status=upload_file.status.name,
            upload_id=upload_file.upload_id,
            message="文件已存在上传记录",
            chunk_list=chunk_list,
            file_size=upload_file.total_size,
            chunk_size=upload_file.chunk_size,
            total_chunks=upload_file.total_chunks,
        )
    else:
        params = UploadCreateParams(
            file_name=request.file_name,
            total_size=request.file_size,
            chunk_size=5 * 1024 * 1024,
        )
        new_upload = FileMatedata(
            **params.dict(),
            uploader_id=request.user_id,
        )
        session.add(new_upload)
        session.commit()
        session.refresh(new_upload)

        return UploadStatusResponse(
            status=FileUploadState.pending.name,
            upload_id=new_upload.upload_id,
            message="已创建新上传任务",
            chunk_list=[],
            file_size=new_upload.total_size,
            chunk_size=new_upload.chunk_size,
            total_chunks=new_upload.total_chunks,
        )


@router.post(
    "/chunk",
    response_model=UploadChunkResponse,
    summary="分块文件上传接口",
    description="分块文件上传接口，用于上传单个分块数据",
)
def upload_chunk(
    session: SessionDep,
    upload_id: str = Form(..., description="上传任务ID"),
    chunk_index: int = Form(..., description="分块序号(从0开始)"),
    chunk_file: UploadFile = File(..., description="分块文件数据"),
) -> UploadChunkResponse:
    statement = select(FileMatedata).where(FileMatedata.upload_id == upload_id)
    upload_task = session.exec(statement).first()
    if not upload_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"找不到上传任务: {upload_id}")

    if chunk_index < 0 or (upload_task.total_chunks and chunk_index >= upload_task.total_chunks):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无效的分块索引: {chunk_index}, 有效范围: 0-{upload_task.total_chunks - 1}",
        )

    chunk_statement = select(UploadChunk).where(
        UploadChunk.upload_id == upload_id, UploadChunk.chunk_index == chunk_index
    )
    existing_chunk = session.exec(chunk_statement).first()

    if existing_chunk:
        return UploadChunkResponse(success=True, message="分块已存在", upload_id=upload_id, chunk_index=chunk_index)

    chunk_data = chunk_file.file.read()
    uploaded_size = len(chunk_data)

    # 保存分块到临时目录
    chunk_dir = Path(f"{settings.UPLOAD_DIR}/temp/{upload_id}")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / f"chunk_{chunk_index}"
    with open(chunk_path, "wb") as f:
        f.write(chunk_data)

    new_chunk = UploadChunk(
        upload_id=upload_id,
        chunk_index=chunk_index,
        uploaded_size=uploaded_size,
    )
    session.add(new_chunk)

    if upload_task.status == FileUploadState.pending:
        upload_task.status = FileUploadState.uploading

    upload_task.updated_at = datetime.datetime.now()
    session.commit()
    session.refresh(upload_task)

    return UploadChunkResponse(
        success=True,
        message="分块上传成功",
        upload_id=upload_id,
        chunk_index=chunk_index,
    )


@router.post(
    "/merge",
    response_model=MergeChunksResponse,
    summary="合并分块文件",
    description="触发已上传分块的合并操作",
)
def merge_chunks(
    request: MergeChunksRequest,
    session: SessionDep,
) -> MergeChunksResponse:
    statement = select(FileMatedata).where(FileMatedata.upload_id == request.upload_id)
    upload_task = session.exec(statement).first()
    if not upload_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"找不到上传任务: {request.upload_id}")

    if upload_task.file_name != request.file_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件名不匹配")

    if upload_task.status in [FileUploadState.merging]:
        return MergeChunksResponse(
            success=False, message="文件正在处理中", upload_id=request.upload_id, status=upload_task.status.name
        )

    if upload_task.status == FileUploadState.success:
        return MergeChunksResponse(
            success=True,
            message="文件已合并完成",
            upload_id=request.upload_id,
            status=upload_task.status.name,
            file_path=upload_task.storage_path,
        )

    uploaded_chunks = [chunk.chunk_index for chunk in upload_task.chunks]
    expected_chunks = list(range(upload_task.total_chunks))

    if sorted(uploaded_chunks) != expected_chunks:
        missing_chunks = set(expected_chunks) - set(uploaded_chunks)
        return MergeChunksResponse(
            success=False,
            message=f"分块未完全上传，缺少分块: {missing_chunks}",
            upload_id=request.upload_id,
            status=upload_task.status.name,
            error="INCOMPLETE_CHUNKS",
        )

    # 合并分块文件
    chunk_folder = f"{settings.UPLOAD_DIR}/temp/{request.upload_id}"

    try:
        upload_task.status = FileUploadState.merging
        session.commit()

        target_dir = Path(settings.UPLOAD_DIR).absolute() / upload_task.uploader_id
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = os.path.join(target_dir, upload_task.file_name)

        with open(target_path, "wb") as target_file:
            for chunk_index in range(upload_task.total_chunks):
                chunk_path = os.path.join(chunk_folder, f"chunk_{chunk_index}")
                with open(chunk_path, "rb") as chunk_file:
                    target_file.write(chunk_file.read())

        upload_task.status = FileUploadState.success
        upload_task.storage_path = target_path
        upload_task.updated_at = datetime.datetime.now()
        session.commit()

        try:
            shutil.rmtree(chunk_folder)
        except Exception as e:
            print(f"清理临时文件失败: {str(e)}")

        return MergeChunksResponse(
            success=True,
            message="分块合并成功",
            upload_id=request.upload_id,
            status=upload_task.status.name,
            file_path=target_path,
        )

    except Exception as e:
        upload_task.status = FileUploadState.merge_failed
        session.commit()

        return MergeChunksResponse(
            success=False,
            message=f"合并失败: {str(e)}",
            upload_id=request.upload_id,
            status=upload_task.status.name,
            error="MERGE_FAILED",
        )
