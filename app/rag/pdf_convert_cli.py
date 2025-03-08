#!/usr/bin/env python3
"""PDF转Markdown命令行工具。

这个工具可以将PDF文件转换为Markdown格式，支持：
- 提取跨页表格，合并跨页表格为一个表格
- 识别出分页截断的图片，将图片合并为一张图片
- 解析双栏文本
- 解析出页眉、页脚
- 识别出文字包围的图片
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

from pdf_to_md import PDFToMarkdown, pdf_to_markdown
from rich import print
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


def process_single_file(pdf_path: str | Path, output_path: str | Path = None, save_images: bool = False) -> str:
    """处理单个PDF文件。

    Args:
        pdf_path: PDF文件路径
        output_path: 输出Markdown文件路径，默认为None（使用PDF文件名+.md）
        save_images: 是否保存提取的图片

    Returns:
        生成的Markdown文件路径
    """
    try:
        start_time = time.time()
        converter = PDFToMarkdown(pdf_path)
        md_path = converter.save_markdown(output_path)

        if save_images:
            image_dir = Path(md_path).parent / f"{Path(md_path).stem}_images"
            converter.save_images(image_dir)

        elapsed = time.time() - start_time
        print(f"[green]转换成功:[/green] {pdf_path} -> {md_path} [dim](耗时: {elapsed:.2f}秒)[/dim]")
        return md_path
    except Exception as e:
        print(f"[red]转换失败:[/red] {pdf_path} - {str(e)}")
        return None


def process_directory(
    input_dir: str | Path, output_dir: str | Path = None, recursive: bool = False, save_images: bool = False
) -> List[str]:
    """处理目录中的所有PDF文件。

    Args:
        input_dir: 输入目录
        output_dir: 输出目录，默认为None（使用输入目录）
        recursive: 是否递归处理子目录
        save_images: 是否保存提取的图片

    Returns:
        生成的Markdown文件路径列表
    """
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"输入目录不存在或不是目录: {input_dir}")

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有PDF文件
    if recursive:
        pdf_files = list(input_dir.glob("**/*.pdf"))
    else:
        pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[yellow]警告:[/yellow] 在目录 {input_dir} 中没有找到PDF文件")
        return []

    # 处理每个文件
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"[cyan]处理{len(pdf_files)}个PDF文件...[/cyan]", total=len(pdf_files))

        for pdf_file in pdf_files:
            # 计算输出路径
            rel_path = pdf_file.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix(".md")

            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 处理文件
            result = process_single_file(pdf_file, output_path, save_images)
            if result:
                results.append(result)

            progress.update(task, advance=1)

    return results


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="将PDF文件转换为Markdown格式。")
    parser.add_argument("input", help="输入PDF文件或目录")
    parser.add_argument("-o", "--output", help="输出Markdown文件或目录")
    parser.add_argument("-r", "--recursive", action="store_true", help="递归处理子目录")
    parser.add_argument("-i", "--images", action="store_true", help="保存提取的图片")

    args = parser.parse_args()

    input_path = Path(args.input)

    try:
        if input_path.is_dir():
            # 处理目录
            results = process_directory(input_path, args.output, args.recursive, args.images)
            print(f"[bold green]转换完成![/bold green] 共处理了 {len(results)} 个文件。")
        elif input_path.is_file() and input_path.suffix.lower() == ".pdf":
            # 处理单个文件
            result = process_single_file(input_path, args.output, args.images)
            if result:
                print(f"[bold green]转换完成![/bold green] Markdown文件已保存至: {result}")
        else:
            print("[red]错误:[/red] 输入必须是PDF文件或包含PDF文件的目录")
            sys.exit(1)
    except Exception as e:
        print(f"[red]处理过程中出错:[/red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
