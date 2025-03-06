import os
from pathlib import Path

from rich import inspect, print  # noqa
from rich.columns import Columns


def get_all_files_in_dir(dir_path: Path) -> list[Path]:
    """获取指定目录下的所有文件，包括子目录中的文件"""
    if not isinstance(dir_path, Path):
        dir_path = Path(dir_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory {dir_path} does not exist")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path {dir_path} is not a directory")

    results = []
    for root, dirs, files in os.walk(dir_path):
        if isinstance(root, str):
            root = Path(root)

        for file in files:
            results.append(root / file)

        for dir in dirs:
            results.extend(get_all_files_in_dir(root / dir))
    return results


if __name__ == "__main__":
    # print(get_all_files_in_dir("./Miner2PdfAndWord_Markitdown2Excel"))
    paths = list(Path("./data/Miner2PdfAndWord_Markitdown2Excel").glob("**/*"))
    # 过滤 Mac 系统文件.DS_Store 和 .开头的隐藏文件
    paths = filter(lambda x: x.is_file() and not x.name.startswith("."), paths)
    paths = [str(path) for path in paths]
    columns = Columns(
        paths,
        equal=True,
        expand=True,
    )
    print(columns)
