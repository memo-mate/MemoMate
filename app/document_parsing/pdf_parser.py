import re
from pathlib import Path
from typing import Any

# PDF处理库
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core import settings
from app.core.log_adapter import logger
from app.document_parsing.common import (
    CHINESE_NUMERALS,
    COMMON_TITLE_KEYWORDS,
    TITLE_PATTERNS,
    DocumentContentType,
    DocumentSection,
    ParserConfig,
    TOCEntry,
)


class PdfParser:
    """PDF解析器"""

    def __init__(self, config: ParserConfig = None):
        self.config = config or ParserConfig()
        self._suppress_pdf_warnings()

    def _suppress_pdf_warnings(self):
        """抑制PDF处理库的各种警告和低级别日志"""
        import logging
        import warnings

        warnings.filterwarnings("ignore", message="CropBox missing from /Page")
        warnings.filterwarnings("ignore", category=UserWarning)
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        logging.getLogger("pdfplumber").setLevel(logging.ERROR)
        logging.getLogger("fitz").setLevel(logging.ERROR)
        logging.getLogger("PIL").setLevel(logging.ERROR)

    def __call__(self, file_path: str | Path, *args: Any, **kwargs: Any) -> list[Document]:
        "处理文档并返回分块后的文档列表"
        return self.chunk(file_path)

    def chunk(self, file_path: str | Path) -> list[Document]:
        """将文档分割成多个块"""
        file_path = Path(file_path)

        if file_path.suffix.lower() != ".pdf":
            logger.error("不支持的文件类型", file_path=str(file_path), expected="PDF")
            raise ValueError(f"不支持的文件类型: {file_path.suffix}, 仅支持PDF文件")

        logger.info("检测到PDF文档", file_path=str(file_path))

        try:
            if self.config.strategy == ParserConfig.STRATEGY_STRUCTURE:
                return self._structure_split(file_path)
            else:
                return self._basic_split(file_path)
        except Exception as e:
            logger.exception("PDF解析失败", exc_info=e, file_path=str(file_path))
            raise ValueError(f"无法解析PDF文件: {str(e)}")

    def _basic_split(self, file_path: str | Path) -> list[Document]:
        """基本PDF分块方法"""
        file_path = Path(file_path)

        text, metadata = self._extract_text_and_metadata(file_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", ";", ":", "  ", " ", ""],
        )

        chunks = text_splitter.create_documents([text], [metadata])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_len"] = len(chunks)
            estimated_page = i * chunk.page_content.count("\n\n") // (metadata["total_pages"] * 2)
            chunk.metadata["page"] = min(estimated_page, metadata["total_pages"] - 1)
            chunk.metadata["page_label"] = str(chunk.metadata["page"] + 1)

            lines = chunk.page_content.split("\n")
            for line in lines[:3]:
                line = line.strip()
                if line and len(line) < 100 and not line.endswith("."):
                    chunk.metadata["possible_title"] = line
                    break

        logger.info("基本分块完成", file_path=str(file_path), chunk_len=len(chunks))
        return chunks

    def _structure_split(self, file_path: str | Path) -> list[Document]:
        """根据文档结构进行分块"""
        file_path = Path(file_path)

        _, base_metadata = self._extract_text_and_metadata(file_path)

        toc_entries = self._extract_toc(file_path)

        if not toc_entries:
            logger.warning("未检测到目录结构，回退到基本分块", file_path=str(file_path))
            return self._basic_split(file_path)

        sections = self._build_sections(file_path, toc_entries)

        documents = []

        for section in sections:
            if section.level in self.config.heading_levels:
                metadata = base_metadata.copy()
                metadata.update(
                    {
                        "title": section.title,
                        "level": section.level,
                        "page_number": section.page_number,
                        "content_type": DocumentContentType.TEXT.value,
                        "page": section.metadata.get("page"),
                        "page_label": section.metadata.get("page_label"),
                        "end_page": section.metadata.get("end_page"),
                        "end_page_label": section.metadata.get("end_page_label"),
                    }
                )

                section_text = f"# {section.content}"

                if len(section_text) <= settings.CHUNK_SIZE * 1.2:
                    documents.append(Document(page_content=section_text, metadata=metadata))
                else:
                    try:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=settings.CHUNK_SIZE,
                            chunk_overlap=settings.CHUNK_OVERLAP,
                            length_function=len,
                            separators=[
                                "\n\n",
                                "\n",
                                ". ",
                                "! ",
                                "? ",
                                ";",
                                ":",
                                "  ",
                                " ",
                                "",
                            ],
                        )
                        sub_chunks = text_splitter.create_documents([section_text], [metadata])
                        if sub_chunks and not sub_chunks[0].page_content.startswith(f"# {section.title}"):
                            title_text = f"# {section.title}\n\n"
                            sub_chunks[0].page_content = title_text + sub_chunks[0].page_content
                        for chunk in sub_chunks:
                            chunk.metadata["title"] = section.title
                            chunk.metadata["level"] = section.level
                        documents.extend(sub_chunks)

                    except Exception as e:
                        logger.warning(
                            "章节分块失败，作为整块处理",
                            section_title=section.title,
                            error=str(e),
                        )
                        documents.append(Document(page_content=section_text, metadata=metadata))

        if not documents:
            logger.warning("未生成任何结构化文档块，回退到基本分块", file_path=str(file_path))
            return self._basic_split(file_path)

        for i, doc in enumerate(documents):
            doc.metadata["chunk_index"] = i
            doc.metadata["chunk_len"] = len(documents)

        logger.info("结构化分块完成", chunk_len=len(documents))
        return documents

    def _convert_pdf_date(self, pdf_date: str) -> str:
        """将 PDF 日期格式转换为 ISO 格式"""
        if not pdf_date or not pdf_date.startswith("D:"):
            return pdf_date
        date = pdf_date[2:]
        iso_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}T{date[8:10]}:{date[10:12]}:{date[12:14]}"
        if len(date) > 14:
            tz = date[14:].replace("'", ":")
            iso_date += tz[:-1]
        return iso_date

    def _extract_text_and_metadata(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """提取PDF文本内容和元数据"""
        try:
            pdf_document = fitz.open(file_path)
            metadata = {
                "source": str(file_path),
                "total_pages": len(pdf_document),
            }

            pdf_metadata = pdf_document.metadata
            if pdf_metadata:
                metadata.update(
                    {
                        "creator": pdf_metadata.get("creator", ""),
                        "producer": pdf_metadata.get("producer", ""),
                        "creationdate": self._convert_pdf_date(pdf_metadata.get("creationDate", "")),
                        "moddate": self._convert_pdf_date(pdf_metadata.get("modDate", "")),
                    }
                )

            text_content = ""
            for page in pdf_document:
                text_content += page.get_text()
                text_content += "\n\n"

            return text_content, metadata

        except Exception as e:
            logger.exception("PDF内容提取失败", exc_info=e, file_path=str(file_path))
            raise

    def _extract_toc(self, file_path: Path) -> list[TOCEntry]:
        """PDF目录提取方法:
        舍弃直接从目录页提取目录结构
        """
        try:
            heading_entries = self._analyze_document_structure(file_path)
            if heading_entries:
                logger.info("通过文档结构分析提取到标题", file_path=str(file_path))
                return heading_entries
        except Exception as e:
            logger.warning("文档结构分析失败", error=str(e), file_path=str(file_path))

        return []

    def _analyze_document_structure(self, file_path: Path) -> list[TOCEntry]:
        """分析文档结构以识别标题"""
        entries = []
        font_stats = {}
        current_chapter = None
        current_section = None
        current_subsection = None
        doc = fitz.open(file_path)
        blocks_by_page = []
        for _page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            blocks_by_page.append(blocks)

            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_size = span["size"]
                        is_bold = "bold" in span["font"].lower()
                        key = (font_size, is_bold)
                        if key not in font_stats:
                            font_stats[key] = {"count": 0, "samples": []}
                        font_stats[key]["count"] += 1

        normal_text_size = self._find_body_text_size(font_stats)

        for page_num, blocks in enumerate(blocks_by_page):
            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    text = " ".join([span["text"] for span in line["spans"]]).strip()
                    if not text or len(text) > 300:
                        continue

                    max_font_size = max(span["size"] for span in line["spans"])
                    is_bold = any("bold" in span["font"].lower() for span in line["spans"])

                    if not self._is_likely_title(text, max_font_size, normal_text_size, is_bold):
                        continue

                    title_info = self._parse_title_format(text)
                    if not title_info:
                        continue

                    number, title_text, level = title_info

                    if not self._validate_title_hierarchy(
                        number,
                        level,
                        current_chapter,
                        current_section,
                        current_subsection,
                    ):
                        continue

                    if number:
                        if level == 1:
                            current_chapter = number
                            current_section = None
                            current_subsection = None
                        elif level == 2:
                            if current_chapter == number.split(".")[0]:
                                current_section = number
                                current_subsection = None
                        elif level == 3:
                            if current_chapter == number.split(".")[0] and current_section == ".".join(
                                number.split(".")[:2]
                            ):
                                current_subsection = number

                    entries.append(TOCEntry(title=text, level=level, page_number=page_num + 1))

        return entries

    def _validate_title_hierarchy(
        self,
        number: str,
        level: int,
        current_chapter: str,
        current_section: str,
        current_subsection: str,
    ) -> bool:
        """验证标题层级关系"""

        if number == "":
            return True

        if re.match(r"^\d+$", number):
            parts = [number]
        elif re.match(r"^\d+\.\d+(\.\d+)?$", number):
            parts = number.split(".")
        else:
            parts = [number]
            parts = number.split(".")

        if level == 1:
            if not len(parts) == 1:
                return False

            if current_chapter is not None:
                try:
                    current_num = int(current_chapter)
                    new_num = int(number)
                    if new_num != current_num + 1:
                        return False
                except ValueError:
                    return False
            return True

        if level == 2:
            if not (len(parts) == 2 and current_chapter == parts[0]):
                return False

            if current_section is not None:
                try:
                    current_section_num = int(current_section.split(".")[-1])
                    new_section_num = int(parts[-1])
                    if parts[0] == current_section.split(".")[0] and new_section_num != current_section_num + 1:
                        return False
                except ValueError:
                    return False
            return True

        if level == 3:
            if not (len(parts) == 3 and current_chapter == parts[0] and current_section == f"{parts[0]}.{parts[1]}"):
                return False

            if current_subsection is not None:
                try:
                    current_subsection_num = int(current_subsection.split(".")[-1])
                    new_subsection_num = int(parts[-1])
                    if (
                        parts[0] == current_subsection.split(".")[0]
                        and parts[1] == current_subsection.split(".")[1]
                        and new_subsection_num != current_subsection_num + 1
                    ):
                        return False
                except ValueError:
                    return False
            return True

        return False

    def _is_likely_title(self, text: str, font_size: float, normal_text_size: float, is_bold: bool) -> bool:
        """判断是否可能是标题"""
        content = re.sub(r"^[\d.]+\s+", "", text).strip()

        if len(content) < 2:
            return False

        if text.endswith(("。", "，", "；", "：")) or text[-1].islower() or "=" in text or re.search(r"[a-z]+\(", text):
            return False

        if not (font_size > normal_text_size * 1.1 or is_bold):
            return False

        return True

    def _parse_title_format(self, text: str) -> tuple[str, str, int] | None:
        """解析标题格式，返回(编号, 标题文本, 层级)或None"""

        for pattern, level in TITLE_PATTERNS:
            match = re.match(pattern, text)
            if match:
                number = match.group(1)
                title_text = match.group(2).strip()

                if len(title_text.split()) > 20:
                    continue

                if re.search(r"[一二三四五六七八九十百]", number):
                    try:
                        if number == "十":
                            num = 10
                        elif "十" in number:
                            parts = number.split("十")
                            if parts[0]:
                                num = CHINESE_NUMERALS[parts[0]] * 10
                            else:
                                num = 10
                            if parts[1]:
                                num += CHINESE_NUMERALS[parts[1]]
                        else:
                            num = CHINESE_NUMERALS[number]
                        number = str(num)
                    except (KeyError, ValueError):
                        continue

                return number, title_text, level

        if len(text) < 40 and not text.endswith(("。", "，", "；", "：", ".", ",", ";", ":")):
            if text in COMMON_TITLE_KEYWORDS:
                return "", text, 1  # 默认作为一级标题

            for title in COMMON_TITLE_KEYWORDS:
                if text.startswith(title) and len(text) < len(title) + 10:
                    return "", text, 1

        return None

    def _find_body_text_size(self, font_stats):
        """找出文档的正文字体大小"""
        sizes = []
        for (size, _), stats in font_stats.items():
            sizes.extend([size] * stats["count"])

        if sizes:
            from statistics import mode

            try:
                return mode(sizes)
            except Exception:
                sizes.sort()
                return sizes[len(sizes) // 2]
        return 11  # 默认值

    def _build_sections(self, file_path: Path, toc_entries: list[TOCEntry]) -> list[DocumentSection]:
        """根据目录条目构建文档章节结构"""
        try:
            sections = []

            if not toc_entries:
                return []

            pdf_document = fitz.open(file_path)

            for i, entry in enumerate(toc_entries):
                start_page = entry.page_number - 1

                # 确定章节结束页
                if i < len(toc_entries) - 1:
                    next_entry = toc_entries[i + 1]
                    end_page = next_entry.page_number - 1
                    has_next_section = True
                else:
                    end_page = len(pdf_document) - 1
                    has_next_section = False

                # 提取章节内容
                content = []
                for page_num in range(start_page, end_page + 1):
                    if page_num >= len(pdf_document):
                        break

                    page = pdf_document[page_num]
                    page_text = page.get_text()

                    # 处理章节开始页
                    if page_num == start_page and i > 0:
                        title_pos = page_text.find(entry.title)
                        if title_pos != -1:
                            page_text = page_text[title_pos:]

                    # 处理章节结束页
                    if has_next_section and page_num == end_page:
                        next_title = toc_entries[i + 1].title
                        title_pos = page_text.find(next_title)
                        if title_pos != -1:
                            page_text = page_text[:title_pos]

                    content.append(page_text)

                # 创建章节对象
                section = DocumentSection(
                    title=entry.title,
                    level=entry.level,
                    page_number=entry.page_number,
                    content="\n\n".join(content).strip(),
                    metadata={
                        "page": start_page,
                        "page_label": str(start_page + 1),
                        "end_page": end_page,
                        "end_page_label": str(end_page + 1),
                    },
                )
                sections.append(section)

            logger.info("章节构建完成", file_path=str(file_path), section_count=len(sections))
            return sections

        except Exception as e:
            logger.exception("章节构建失败", exc_info=e, file_path=str(file_path))
            return []


if __name__ == "__main__":
    parser = PdfParser()
    # parser(r"C:\Users\Leo\Desktop\从零开始大模型开发与微调基于PyTorch与ChatGLM.pdf")
    parser(r"C:\Users\Leo\Desktop\大规模语言模型：从理论到实践.pdf")
    # parser(r"C:\Users\Leo\Downloads\AI大模型\大规模语言模型：从理论到实践.pdf")
