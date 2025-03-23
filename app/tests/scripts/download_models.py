from app.core import consts


# 下载 bge-m3 模型，并保存到本地
def download_bge_m3_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "BAAI/bge-m3",
        local_dir=consts.BGE_MODEL_PATH,
    )


if __name__ == "__main__":
    download_bge_m3_model()
