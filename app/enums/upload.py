from enum import IntEnum


class FileUploadState(IntEnum):
    """文件上传状态枚举"""

    # 等待上传开始
    pending = 0
    # 上传中（分块传输中）
    uploading = 1
    # 分块全部上传完成
    chunks_uploaded = 2
    # 合并中
    merging = 3
    # 合并成功
    merged = 4
    # 上传成功
    success = 5
    # 分块上传失败
    upload_failed = 6
    # 合并失败
    merge_failed = 7
    # 校验失败
    verification_failed = 8
    # 用户取消
    canceled = 9

    @classmethod
    def get_active_states(cls) -> list["FileUploadState"]:
        """获取进行中状态列表"""
        return [cls.pending, cls.uploading, cls.merging]

    def is_terminal_state(self) -> bool:
        """判断是否为终态（不可再变更的状态）"""
        return self in [
            FileUploadState.success,
            FileUploadState.upload_failed,
            FileUploadState.merge_failed,
            FileUploadState.verification_failed,
            FileUploadState.canceled,
        ]
