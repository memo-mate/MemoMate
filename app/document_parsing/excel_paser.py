import os
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from app.enums.task import DocumentFileTaskType


class ExcelParser:
    def __call__(self, file_path: str, file_type: DocumentFileTaskType | None = None, *args: Any, **kwds: Any) -> Any:
        parsed_data: list[tuple[str, pd.DataFrame]] = []
        if file_type is None:
            file_type = DocumentFileTaskType.get_file_type(file_path)

        file_name = os.path.basename(file_path)
        match file_type:
            case DocumentFileTaskType.csv:
                parsed_data.append((file_name, pd.read_csv(file_path)))
            case DocumentFileTaskType.xlsx | DocumentFileTaskType.xls:
                dfs = pd.read_excel(file_path, sheet_name=None)
                for sheet_name, df in dfs.items():
                    parsed_data.append((f"{file_name} - {sheet_name}", df))
            case _:
                raise ValueError(f"Unsupported file type: {file_type}")

        # 使用 rich table 打印 parsed_data
        for table_name, df in parsed_data:
            # 创建表格
            table = Table(
                title=table_name,
                title_style="bold magenta",
                border_style="blue",
                title_justify="center",
                expand=True,
            )

            # 添加列标题
            for column in df.columns:
                table.add_column(str(column), header_style="bold cyan", style="green")

            # 添加行数据 - 交替行色彩
            for i, (_, row) in enumerate(df.iterrows()):
                row_style = "dim" if i % 2 == 0 else ""
                table.add_row(*[str(value) for value in row.values], style=row_style)

            # 打印表格
            console = Console()
            console.print(table)
