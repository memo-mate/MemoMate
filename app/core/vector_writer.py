from app.loaders.TextLoader import TextLoader
from app.loaders.PdfLoader import PdfLoader
from app.loaders.ExcelLoader import ExcelLoader
from app.loaders.WordLoader import WordLoader
import os
import time
from loguru import logger

begin_time = time.time()

file_list = []

file_path = r"D:\Test_Data"
for root, dirs, files in os.walk(file_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_ext = os.path.splitext(file_path)[1]
        file_list.append(file_path)


for idx, file_path in enumerate(file_list):
    file_ext = os.path.splitext(file_path)[1]
    if file_ext in [".pdf"]:
        loader = PdfLoader()
    elif file_ext in [".txt", ".md"]:
        loader = TextLoader()
    elif file_ext in [".xlsx", ".xls"]:
        loader = ExcelLoader()
    elif file_ext in [".docx", ".doc"]:
        loader = WordLoader()
    else:
        logger.error(f"不支持的文件类型: {file_ext}")
        continue

    start_time = time.time()
    loader.process_file(file_path)
    end_time = time.time()
    logger.info(
        f"进度:【{idx+1}/{len(file_list)}】, 文件:【{file_path}】, 用时:【{end_time - start_time}秒】"
    )

end_time = time.time()
logger.info(f"总用时:【{end_time - begin_time}秒】")
