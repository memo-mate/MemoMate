from app.loaders.TextLoader import TextLoader
from app.loaders.PdfLoader import PdfLoader
import os


file_path = r"D:\Test_Data\chrome插件问题排查.pdf"
file_ext = os.path.splitext(file_path)[1]


if file_ext == ".pdf":
    loader = PdfLoader()
elif file_ext == ".txt":
    loader = TextLoader()
else:
    raise ValueError(f"不支持的文件类型: {file_ext}")

loader.process_file(file_path)
