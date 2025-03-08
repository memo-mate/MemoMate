import os
import uuid
from pathlib import Path

import httpx
from langchain.document_loaders import (
    MathpixPDFLoader,
    PDFMinerLoader,
    PDFPlumberLoader,
    PyMuPDFLoader,
    PyPDFDirectoryLoader,
    PyPDFium2Loader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from markitdown import MarkItDown
from rich import print  # noqa


def transfer_file(file_path: str | Path) -> None:
    url = "https://mineru.net/api/v4/file-urls/batch"
    header = {"Content-Type": "application/json", "Authorization": "Bearer eyJ0eXBlIjoiSl...请填写准确的token！"}
    data = {
        "enable_formula": True,
        "language": "en",
        "layout_model": "doclayout_yolo",
        "enable_table": True,
        "files": [{"name": file_path.name, "is_ocr": True, "data_id": f"daoji_{uuid.uuid4()}"}],
    }
    try:
        response = httpx.post(url, headers=header, json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"response success. result:{result}")
            if result["code"] == 0:
                batch_id = result["data"]["batch_id"]
                urls = result["data"]["file_urls"]
                print(f"batch_id:{batch_id},urls:{urls}")
                with open(file_path, "rb") as f:
                    res_upload = httpx.put(urls[0], data=f)
                if res_upload.status_code == 200:
                    print("upload success")
                else:
                    print("upload failed")
            else:
                print(f"apply upload url failed,reason:{result['msg']}")
        else:
            print(f"response not success. status:{response.status_code} ,result:{response}")
    except Exception as err:
        print(err)


def gen_file_list(
    data_dir: Path = Path("./Test_Data"),
    suffix_list: list[str] = None,
) -> list[str | Path]:
    if suffix_list is None:
        suffix_list = [".pdf", ".docx", ".doc", ".pptx", ".ppt"]
    file_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.is_file() and file_path.suffix in suffix_list:
                file_list.append(file_path)

        for dir in dirs:
            file_path = Path(root) / dir
            if file_path.is_dir():
                for file in file_path.iterdir():
                    if file.is_file() and file.suffix in suffix_list:
                        file_list.append(file)
    return file_list


def excel_to_markdown(file_path: str | Path) -> None:
    md = MarkItDown()
    result = md.convert(file_path)
    with open(file_path.with_suffix(".md"), "w") as f:
        f.write(result.text_content)


if __name__ == "__main__":
    file_list = gen_file_list(suffix_list=[".xlsx"])
    for file in file_list:
        excel_to_markdown(file)
