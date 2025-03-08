import base64
from pathlib import Path

import pandas as pd
import pymupdf
from langchain_community.document_loaders import PyMuPDFLoader
from rich import print


class PDFToMarkdown:
    """PDF到Markdown转换器。

    能够处理：
    - 提取跨页表格，合并跨页表格为一个表格
    - 识别出分页截断的图片，将图片合并为一张图片
    - 解析双栏文本
    - 解析出页眉、页脚
    - 识别出文字包围的图片
    """

    def __init__(self, pdf_path: str | Path):
        """初始化PDF转换器。

        Args:
            pdf_path: PDF文件路径
        """
        self.pdf_path = Path(pdf_path) if isinstance(pdf_path, str) else pdf_path
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF文件 {self.pdf_path} 不存在")

        # 初始化PyMuPDF文档对象
        self.doc = pymupdf.open(self.pdf_path)
        # 使用LangChain加载器加载文档
        self.loader = PyMuPDFLoader(str(self.pdf_path))

        # 提取的内容存储
        self.pages = []  # 每页内容
        self.images = []  # 图片信息
        self.tables = []  # 表格信息
        self.headers = []  # 页眉信息
        self.footers = []  # 页脚信息

    def extract_text(self) -> list[str]:
        """提取PDF文本内容，处理双栏文本。

        Returns:
            按页组织的文本内容列表
        """

        pages_text = []
        for page in self.doc:
            page: pymupdf.Page
            # 使用PyMuPDF的布局分析功能识别文本块
            blocks = page.get_text("dict")["blocks"]

            # 处理双栏文本
            text_blocks = []
            for block in blocks:
                # 只处理文本块
                if block["type"] == 0:  # 0表示文本块
                    # 获取块的边界框
                    rect = block["bbox"]
                    x0, y0, x1, y1 = rect

                    # 获取文本内容
                    text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text += span["text"]
                        text += "\n"

                    # 将文本块与其位置信息一起存储
                    text_blocks.append(
                        {
                            "text": text.strip(),
                            "x0": x0,
                            "y0": y0,
                            "x1": x1,
                            "y1": y1,
                            "center_x": (x0 + x1) / 2,
                            "center_y": (y0 + y1) / 2,
                        }
                    )

            # 识别页面是否为双栏布局
            page_width = page.rect.width
            is_double_column = self._detect_double_column(text_blocks, page_width)

            if is_double_column:
                # 对双栏布局进行处理
                left_column = []
                right_column = []

                # 以页面中线为界，划分左右栏
                mid_x = page_width / 2
                for block in text_blocks:
                    if block["center_x"] < mid_x:
                        left_column.append(block)
                    else:
                        right_column.append(block)

                # 按y坐标排序每一栏
                left_column.sort(key=lambda b: b["y0"])
                right_column.sort(key=lambda b: b["y0"])

                # 组合文本内容，先左栏，后右栏
                text = "\n".join([block["text"] for block in left_column])
                text += "\n\n"  # 左右栏之间添加分隔
                text += "\n".join([block["text"] for block in right_column])
            else:
                # 单栏布局，按y坐标排序
                text_blocks.sort(key=lambda b: b["y0"])
                text = "\n".join([block["text"] for block in text_blocks])

            pages_text.append(text)

        self.pages = pages_text
        return pages_text

    def _detect_double_column(self, text_blocks: list[dict], page_width: float) -> bool:
        """检测页面是否为双栏布局。

        Args:
            text_blocks: 文本块列表
            page_width: 页面宽度

        Returns:
            是否为双栏布局
        """
        if len(text_blocks) < 4:  # 文本块太少，可能不是双栏
            return False

        # 统计左右半页的文本块数量
        mid_x = page_width / 2
        left_count = sum(1 for block in text_blocks if block["center_x"] < mid_x)
        right_count = len(text_blocks) - left_count

        # 如果左右两边都有足够多的文本块，且数量相对均衡，可能是双栏
        if left_count >= 2 and right_count >= 2:
            # 计算文本块在x轴上的分布
            x_centers = [block["center_x"] for block in text_blocks]

            # 判断x坐标分布是否呈现双峰分布
            left_density = sum(1 for x in x_centers if x < mid_x * 0.8) / len(x_centers)
            right_density = sum(1 for x in x_centers if x > mid_x * 1.2) / len(x_centers)
            middle_density = 1 - left_density - right_density

            # 如果左右两侧密度高，中间密度低，可能是双栏
            if left_density > 0.25 and right_density > 0.25 and middle_density < 0.3:
                return True

        return False

    def extract_images(self) -> list[dict]:
        """提取PDF中的图片，处理跨页图片。

        Returns:
            图片信息列表，包含图片数据、页码、坐标等
        """
        images = []
        potential_cross_page_images = []

        # 第一遍：提取所有图片信息
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]

            # 使用get_images提取基本图片信息
            image_list = page.get_images(full=True)

            # 使用get_image_bbox获取图片在页面中的位置
            for img_index, img in enumerate(image_list):
                xref = img[0]  # 图片的交叉引用ID
                base_image = self.doc.extract_image(xref)
                image_data = base_image["image"]

                # 尝试获取图片在页面上的位置
                try:
                    image_rects = page.get_image_bbox(xref)
                    for rect in image_rects:
                        # 检测图片是否可能被页面边缘截断
                        is_at_top = rect.y0 < 20  # 图片靠近页面顶部
                        is_at_bottom = rect.y1 > page.rect.height - 20  # 图片靠近页面底部

                        # 记录图片信息
                        image_info = {
                            "page_num": page_num + 1,  # 1-based页码
                            "image_index": img_index,
                            "xref": xref,
                            "rect": rect,
                            "data": base64.b64encode(image_data).decode(),
                            "ext": base_image["ext"],
                            "width": base_image["width"],
                            "height": base_image["height"],
                            "is_at_top": is_at_top,
                            "is_at_bottom": is_at_bottom,
                        }
                        images.append(image_info)

                        # 如果图片位于页面顶部或底部，可能是跨页图片
                        if is_at_top or is_at_bottom:
                            potential_cross_page_images.append(image_info)
                except Exception as e:
                    # 如果无法获取位置信息，则记录基本信息
                    print(f"无法获取图片位置信息: {e}")
                    image_info = {
                        "page_num": page_num + 1,
                        "image_index": img_index,
                        "xref": xref,
                        "rect": None,
                        "data": base64.b64encode(image_data).decode(),
                        "ext": base_image["ext"],
                        "width": base_image["width"],
                        "height": base_image["height"],
                        "is_at_top": False,
                        "is_at_bottom": False,
                    }
                    images.append(image_info)

        # 第二遍：尝试识别并合并跨页图片
        merged_images = self._merge_cross_page_images(images, potential_cross_page_images)

        self.images = merged_images
        return merged_images

    def _merge_cross_page_images(self, all_images: list[dict], potential_cross_page_images: list[dict]) -> list[dict]:
        """识别并合并可能的跨页图片。

        Args:
            all_images: 所有提取的图片信息
            potential_cross_page_images: 可能是跨页图片的信息

        Returns:
            处理后的图片列表（可能包含合并后的图片）
        """
        if not potential_cross_page_images:
            return all_images

        # 按页码排序潜在的跨页图片
        potential_cross_page_images.sort(key=lambda img: img["page_num"])

        merged_images = all_images.copy()
        merged_indices = set()  # 记录已经合并的图片索引

        # 检测相邻页面的图片是否可能是同一图片的不同部分
        for i in range(len(potential_cross_page_images) - 1):
            img1 = potential_cross_page_images[i]
            img2 = potential_cross_page_images[i + 1]

            # 检查是否为相邻页面
            if img2["page_num"] - img1["page_num"] == 1:
                # 检查第一个图片是否在页面底部，第二个图片是否在页面顶部
                if img1["is_at_bottom"] and img2["is_at_top"]:
                    # 检查宽度是否相似（允许一定的误差）
                    width_ratio = img1["width"] / img2["width"] if img2["width"] > 0 else 0
                    if 0.9 <= width_ratio <= 1.1:
                        # 可能是同一图片的两部分，尝试合并
                        try:
                            img1_data = base64.b64decode(img1["data"])
                            img2_data = base64.b64decode(img2["data"])

                            # 使用PIL库合并图片
                            import io

                            from PIL import Image

                            img1_pil = Image.open(io.BytesIO(img1_data))
                            img2_pil = Image.open(io.BytesIO(img2_data))

                            # 创建新图片（宽度取最大值，高度为两图片高度之和）
                            merged_width = max(img1_pil.width, img2_pil.width)
                            merged_height = img1_pil.height + img2_pil.height
                            merged_pil = Image.new("RGB", (merged_width, merged_height))

                            # 将两部分图片粘贴到新图片上
                            merged_pil.paste(img1_pil, (0, 0))
                            merged_pil.paste(img2_pil, (0, img1_pil.height))

                            # 将合并后的图片编码为base64
                            buffer = io.BytesIO()
                            merged_pil.save(buffer, format=img1["ext"].upper())
                            merged_data = base64.b64encode(buffer.getvalue()).decode()

                            # 创建合并后的图片信息
                            merged_img_info = {
                                "page_num": img1["page_num"],  # 使用第一部分的页码
                                "image_index": -1,  # 特殊标记，表示是合并图片
                                "xref": -1,
                                "rect": None,
                                "data": merged_data,
                                "ext": img1["ext"],
                                "width": merged_width,
                                "height": merged_height,
                                "is_merged": True,
                                "merged_from": [img1["page_num"], img2["page_num"]],
                                "is_at_top": False,
                                "is_at_bottom": False,
                            }

                            # 添加合并后的图片
                            merged_images.append(merged_img_info)

                            # 记录已合并的图片索引，后续将排除这些图片
                            for j, img in enumerate(merged_images):
                                if (
                                    img["page_num"] == img1["page_num"] and img["image_index"] == img1["image_index"]
                                ) or (
                                    img["page_num"] == img2["page_num"] and img["image_index"] == img2["image_index"]
                                ):
                                    merged_indices.add(j)

                            print(f"合并了跨页图片：第{img1['page_num']}页和第{img2['page_num']}页")
                        except Exception as e:
                            print(f"合并图片失败: {e}")

        # 过滤掉已经合并的图片
        result = [img for i, img in enumerate(merged_images) if i not in merged_indices]

        return result

    def extract_tables(self) -> list[dict]:
        """提取PDF中的表格，处理跨页表格。

        Returns:
            表格信息列表，包含表格数据、页码、坐标等
        """
        tables = []
        potential_cross_page_tables = []

        # 第一遍：提取所有表格
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]

            try:
                # 使用PyMuPDF的表格检测功能
                # 参数说明：
                # - vertical_strategy和horizontal_strategy：设置为"text"表示基于文本位置检测表格线
                # - extend_y：向下扩展表格边界
                # - snap_tolerance：捕捉文本到表格线的容差
                tab = page.find_tables(
                    vertical_strategy="text", horizontal_strategy="text", extend_y=5, snap_tolerance=5
                )

                if tab.tables:
                    for table_index, table in enumerate(tab.tables):
                        # 获取表格的边界框
                        rect = table.bbox

                        # 检测表格是否可能被页面边缘截断
                        is_at_top = rect.y0 < 20  # 表格靠近页面顶部
                        is_at_bottom = rect.y1 > page.rect.height - 20  # 表格靠近页面底部

                        # 提取表格数据
                        rows = []
                        for row in range(table.row_count):
                            row_data = []
                            for col in range(table.col_count):
                                cell = table.cells[row * table.col_count + col]
                                # 提取单元格中的文本
                                text = page.get_text("text", clip=cell.rect).strip()
                                row_data.append(text)
                            rows.append(row_data)

                        # 创建表格信息对象
                        table_info = {
                            "page_num": page_num + 1,  # 1-based页码
                            "table_index": table_index,
                            "rect": rect,
                            "data": rows,
                            "row_count": table.row_count,
                            "col_count": table.col_count,
                            "is_at_top": is_at_top,
                            "is_at_bottom": is_at_bottom,
                        }

                        tables.append(table_info)

                        # 如果表格位于页面顶部或底部，可能是跨页表格
                        if is_at_top or is_at_bottom:
                            potential_cross_page_tables.append(table_info)
            except Exception as e:
                print(f"提取第{page_num + 1}页的表格时出错: {e}")

        # 第二遍：尝试识别并合并跨页表格
        merged_tables = self._merge_cross_page_tables(tables, potential_cross_page_tables)

        self.tables = merged_tables
        return merged_tables

    def _merge_cross_page_tables(self, all_tables: list[dict], potential_cross_page_tables: list[dict]) -> list[dict]:
        """识别并合并可能的跨页表格。

        Args:
            all_tables: 所有提取的表格信息
            potential_cross_page_tables: 可能是跨页表格的信息

        Returns:
            处理后的表格列表（可能包含合并后的表格）
        """
        if not potential_cross_page_tables:
            return all_tables

        # 按页码排序潜在的跨页表格
        potential_cross_page_tables.sort(key=lambda tab: tab["page_num"])

        merged_tables = all_tables.copy()
        merged_indices = set()  # 记录已经合并的表格索引

        # 检测相邻页面的表格是否可能是同一表格的不同部分
        for i in range(len(potential_cross_page_tables) - 1):
            tab1 = potential_cross_page_tables[i]
            tab2 = potential_cross_page_tables[i + 1]

            # 检查是否为相邻页面
            if tab2["page_num"] - tab1["page_num"] == 1:
                # 检查第一个表格是否在页面底部，第二个表格是否在页面顶部
                if tab1["is_at_bottom"] and tab2["is_at_top"]:
                    # 检查列数是否相同
                    if tab1["col_count"] == tab2["col_count"]:
                        # 可能是同一表格的两部分，尝试合并
                        try:
                            # 合并数据
                            merged_data = tab1["data"] + tab2["data"]

                            # 创建合并后的表格信息
                            merged_table_info = {
                                "page_num": tab1["page_num"],  # 使用第一部分的页码
                                "table_index": -1,  # 特殊标记，表示是合并表格
                                "rect": None,
                                "data": merged_data,
                                "row_count": len(merged_data),
                                "col_count": tab1["col_count"],
                                "is_merged": True,
                                "merged_from": [tab1["page_num"], tab2["page_num"]],
                                "is_at_top": False,
                                "is_at_bottom": False,
                            }

                            # 添加合并后的表格
                            merged_tables.append(merged_table_info)

                            # 记录已合并的表格索引，后续将排除这些表格
                            for j, tab in enumerate(merged_tables):
                                if (
                                    tab["page_num"] == tab1["page_num"] and tab["table_index"] == tab1["table_index"]
                                ) or (
                                    tab["page_num"] == tab2["page_num"] and tab["table_index"] == tab2["table_index"]
                                ):
                                    merged_indices.add(j)

                            print(f"合并了跨页表格：第{tab1['page_num']}页和第{tab2['page_num']}页")
                        except Exception as e:
                            print(f"合并表格失败: {e}")

        # 过滤掉已经合并的表格
        result = [tab for i, tab in enumerate(merged_tables) if i not in merged_indices]

        return result

    def extract_headers_footers(self) -> tuple[list[str], list[str]]:
        """提取页眉和页脚。

        Returns:
            (页眉列表, 页脚列表)
        """
        headers = []
        footers = []

        # 页眉页脚检测的参数
        header_height = 50  # 页眉区域高度
        footer_height = 50  # 页脚区域高度

        # 保存每页的页眉和页脚文本
        page_headers = []
        page_footers = []

        # 第一遍：提取所有页面的顶部和底部文本
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]

            # 页面尺寸
            page_width = page.rect.width
            page_height = page.rect.height

            # 页眉区域
            header_rect = pymupdf.Rect(0, 0, page_width, header_height)
            # 页脚区域
            footer_rect = pymupdf.Rect(0, page_height - footer_height, page_width, page_height)

            # 提取页眉文本
            header_text = page.get_text("text", clip=header_rect).strip()
            page_headers.append(header_text)

            # 提取页脚文本
            footer_text = page.get_text("text", clip=footer_rect).strip()
            page_footers.append(footer_text)

        # 第二遍：通过重复模式识别页眉和页脚
        # 如果某段文本在大多数页面顶部或底部重复出现，则很可能是页眉或页脚

        # 计算每个页眉文本出现的次数
        header_counts = {}
        for header in page_headers:
            if header:  # 只统计非空文本
                if header in header_counts:
                    header_counts[header] += 1
                else:
                    header_counts[header] = 1

        # 计算每个页脚文本出现的次数
        footer_counts = {}
        for footer in page_footers:
            if footer:  # 只统计非空文本
                if footer in footer_counts:
                    footer_counts[footer] += 1
                else:
                    footer_counts[footer] = 1

        # 找出重复次数最多的页眉和页脚
        min_repeat_count = max(2, len(self.doc) // 3)  # 至少重复2次或1/3的页数

        # 识别页眉
        for header, count in header_counts.items():
            if count >= min_repeat_count:
                headers.append(header)

        # 识别页脚
        for footer, count in footer_counts.items():
            if count >= min_repeat_count:
                footers.append(footer)

        # 如果有多个候选页眉/页脚，按重复次数排序
        headers.sort(key=lambda h: header_counts[h], reverse=True)
        footers.sort(key=lambda f: footer_counts[f], reverse=True)

        self.headers = headers
        self.footers = footers

        return headers, footers

    def process(self) -> str:
        """处理PDF并生成Markdown。

        Returns:
            生成的Markdown文本
        """
        # 1. 提取文本内容（处理双栏文本）
        self.extract_text()

        # 2. 提取图片（处理跨页图片）
        self.extract_images()

        # 3. 提取表格（处理跨页表格）
        self.extract_tables()

        # 4. 提取页眉页脚
        self.extract_headers_footers()

        # 5. 生成Markdown
        return self._generate_markdown()

    def _generate_markdown(self) -> str:
        """生成Markdown文本。

        Returns:
            Markdown文本
        """
        markdown = []

        # 文档标题
        if hasattr(self.doc, "metadata") and self.doc.metadata.get("title"):
            markdown.append(f"# {self.doc.metadata['title']}\n")
        else:
            # 如果没有标题，使用文件名
            markdown.append(f"# {self.pdf_path.stem}\n")

        # 作者信息
        if hasattr(self.doc, "metadata") and self.doc.metadata.get("author"):
            markdown.append(f"作者: {self.doc.metadata['author']}\n")

        markdown.append("\n---\n\n")  # 分隔线

        # 根据页面内容生成Markdown
        for page_num, page_text in enumerate(self.pages):
            page_num_1_based = page_num + 1

            # 跳过识别出的页眉页脚
            clean_text = page_text
            for header in self.headers:
                clean_text = clean_text.replace(header, "")
            for footer in self.footers:
                clean_text = clean_text.replace(footer, "")
            clean_text = clean_text.strip()

            # 查找该页的表格
            page_tables = [table for table in self.tables if table.get("page_num") == page_num_1_based]

            # 查找该页的图片
            page_images = [image for image in self.images if image.get("page_num") == page_num_1_based]

            # 如果该页有表格或图片，需要将文本分段并在适当位置插入表格和图片
            if page_tables or page_images:
                # 获取所有元素（文本块、表格、图片）的垂直位置，用于确定它们在页面中的位置
                elements = []

                # 临时实现：将页面文本视为一个整体元素
                # 在实际实现中，你可能需要将文本分成更小的块，以便更准确地与表格和图片交错
                elements.append(
                    {
                        "type": "text",
                        "content": clean_text,
                        "y0": 0,  # 假设文本从页面顶部开始
                        "y1": self.doc[page_num].rect.height,  # 假设文本延伸到页面底部
                    }
                )

                # 添加表格元素
                for table in page_tables:
                    if table.get("rect"):
                        elements.append(
                            {"type": "table", "content": table, "y0": table["rect"].y0, "y1": table["rect"].y1}
                        )

                # 添加图片元素
                for image in page_images:
                    if image.get("rect"):
                        elements.append(
                            {"type": "image", "content": image, "y0": image["rect"].y0, "y1": image["rect"].y1}
                        )

                # 按垂直位置排序元素
                elements.sort(key=lambda e: e["y0"])

                # 生成Markdown
                for element in elements:
                    if element["type"] == "text":
                        markdown.append(element["content"])
                    elif element["type"] == "table":
                        table = element["content"]
                        markdown.append(self._table_to_markdown(table))
                    elif element["type"] == "image":
                        image = element["content"]
                        markdown.append(self._image_to_markdown(image))
            else:
                # 如果没有表格和图片，直接添加文本
                markdown.append(clean_text)

            # 页面分隔符
            if page_num < len(self.pages) - 1:
                markdown.append("\n\n---\n\n")

        return "\n".join(markdown)

    def _table_to_markdown(self, table: dict) -> str:
        """将表格转换为Markdown格式。

        Args:
            table: 表格信息

        Returns:
            Markdown格式的表格
        """
        if not table or not table.get("data"):
            return ""

        rows = table["data"]
        if not rows:
            return ""

        md_table = []

        # 添加表头和分隔行
        if len(rows) > 0:
            # 第一行作为表头
            header = " | ".join([str(cell) for cell in rows[0]])
            md_table.append(f"| {header} |")

            # 分隔行
            separator = " | ".join(["---" for _ in range(len(rows[0]))])
            md_table.append(f"| {separator} |")

            # 数据行
            for row in rows[1:]:
                row_text = " | ".join([str(cell) for cell in row])
                md_table.append(f"| {row_text} |")

        return "\n".join(md_table)

    def _image_to_markdown(self, image: dict) -> str:
        """将图片转换为Markdown图片引用格式。

        Args:
            image: 图片信息

        Returns:
            Markdown格式的图片引用
        """
        # 使用图片索引作为标识
        image_id = f"image_{image['page_num']}_{image['image_index']}"
        if image.get("is_merged"):
            image_id = f"merged_image_{image['merged_from'][0]}_{image['merged_from'][1]}"

        # 在实际使用中，你可能需要保存图片到文件，并引用文件路径
        # 这里仅生成图片标记和说明
        return f"<image>\n\n![{image_id}](./images/{image_id}.{image['ext']})\n"

    def save_markdown(self, output_path: str | Path = None) -> str:
        """保存Markdown到文件。

        Args:
            output_path: 输出文件路径，如果为None则使用原PDF文件名+.md

        Returns:
            保存的文件路径
        """
        if output_path is None:
            output_path = self.pdf_path.with_suffix(".md")
        else:
            output_path = Path(output_path) if isinstance(output_path, str) else output_path

        markdown = self.process()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        return str(output_path)

    def save_images(self, output_dir: str | Path = None) -> list[str]:
        """保存提取的图片到指定目录。

        Args:
            output_dir: 输出目录，如果为None则使用原PDF文件名_images

        Returns:
            保存的图片文件路径列表
        """
        if output_dir is None:
            output_dir = self.pdf_path.with_suffix("_images")
        else:
            output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir

        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        image_paths = []
        for i, img in enumerate(self.images):
            img_path = output_dir / f"image_{i + 1}.{img['ext']}"
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(img["data"]))
            image_paths.append(str(img_path))

        return image_paths

    def __del__(self):
        """关闭文档。"""
        if hasattr(self, "doc"):
            self.doc.close()


def pdf_to_markdown(pdf_path: str | Path, output_path: str | Path = None) -> str:
    """将PDF转换为Markdown。

    Args:
        pdf_path: PDF文件路径
        output_path: 输出Markdown文件路径，默认为None（使用PDF文件名+.md）

    Returns:
        保存的Markdown文件路径
    """
    converter = PDFToMarkdown(pdf_path)
    return converter.save_markdown(output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python pdf_to_md.py <pdf_path> [output_path]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    result_path = pdf_to_markdown(pdf_path, output_path)
    print(f"转换完成，Markdown文件已保存至: {result_path}")
