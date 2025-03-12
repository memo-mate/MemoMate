import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# 获取终端宽度
console = Console()
width = console.width - 4  # 留出边框空间

# 模拟数据
mock_files = {
    "task1": ["file1.png", "file2.png", "file3.png", "file4.png"],
    "task2": ["file5.jpg", "file6.jpg", "file7.jpg"],
    "task3": ["file8.bmp", "file9.bmp"],
    "task4": ["file10.jpeg", "file11.jpeg", "file12.jpeg"],
    "task5": ["file13.png", "file14.png"],
}

name_file_dic = defaultdict(list, mock_files)
task_file_dic = deepcopy(name_file_dic)
total = sum(len(files) for files in mock_files.values())

# 创建进度条
job_progress = Progress(
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
    auto_refresh=False,
    expand=True,  # 允许进度条扩展
)

overall_progress = Progress(
    "{task.description}",
    BarColumn(),
    MofNCompleteColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
    auto_refresh=False,
    expand=True,  # 允许进度条扩展
)
overall_task = overall_progress.add_task("总进度", total=total)

# 创建进度表格
progress_table = Table.grid(expand=True)  # 允许表格扩展
progress_table.add_row(Panel.fit(overall_progress, title="总进度", border_style="green", width=width))
progress_table.add_row(Panel.fit(job_progress, title="[b]子进度", border_style="red", width=width))


def process_file(_: str):
    """模拟文件处理过程"""

    time.sleep(random.random() * 2)  # 模拟处理时间


def process_task(task_name, files):
    """处理单个任务的所有文件"""
    task_id = task_ids[task_name]  # 获取任务 ID
    for file in files:
        process_file(file)
        job_progress.advance(task_id)
        overall_progress.advance(overall_task)


with Live(progress_table, refresh_per_second=10):
    # 初始添加所有任务
    task_ids = {}
    for key, value in mock_files.items():
        task_id = job_progress.add_task(f"{key}", total=len(value))
        task_ids[key] = task_id

    # 使用 ThreadPoolExecutor 并行处理任务
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                process_task,
                job.description,
                task_file_dic[job.description],
            ): job
            for job in job_progress.tasks
        }
        for future in as_completed(futures):
            future.result()  # 等待任务完成
