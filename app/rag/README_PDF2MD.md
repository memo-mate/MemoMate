# PDF到Markdown转换工具

## 功能特点

这个工具可以将PDF文件转换为Markdown格式，支持：

- ✅ 提取跨页表格，合并跨页表格为一个表格
- ✅ 识别出分页截断的图片，将图片合并为一张图片
- ✅ 解析双栏文本，按照阅读顺序转换
- ✅ 解析出页眉、页脚
- ✅ 识别出文字包围的图片
- ✅ 保持原文的层级、标题结构

## 安装

### 安装依赖

使用uv安装依赖：

```bash
uv pip install -r requirements.txt
```

或者使用pip：

```bash
pip install -r requirements.txt
```

## 使用方法

### 命令行工具

这个工具提供了简单的命令行界面，可以处理单个PDF文件或整个目录：

```bash
# 处理单个文件
python app/pdf_convert_cli.py 你的PDF文件.pdf -o 输出文件.md -i

# 处理整个目录
python app/pdf_convert_cli.py 输入目录 -o 输出目录 -r -i
```

参数说明：

- `-o, --output`：指定输出文件或目录
- `-r, --recursive`：递归处理子目录（仅当输入是目录时有效）
- `-i, --images`：保存提取的图片

### 作为Python库使用

您也可以在自己的Python代码中使用这个工具：

```python
from app.pdf_to_md import PDFToMarkdown, pdf_to_markdown

# 方法1：使用简单的函数接口
markdown_path = pdf_to_markdown("你的PDF文件.pdf", "输出文件.md")

# 方法2：使用类接口（更灵活）
converter = PDFToMarkdown("你的PDF文件.pdf")
# 提取文本（处理双栏文本）
text = converter.extract_text()
# 提取图片（处理跨页图片）
images = converter.extract_images()
# 提取表格（处理跨页表格）
tables = converter.extract_tables()
# 提取页眉页脚
headers, footers = converter.extract_headers_footers()
# 生成Markdown
markdown = converter.process()
# 保存Markdown
converter.save_markdown("输出文件.md")
# 保存图片
converter.save_images("图片目录")
```

## 限制和已知问题

- 复杂的表格布局可能无法完全正确解析
- 某些特殊格式的PDF可能无法正确识别跨页内容
- 图片质量可能会因为提取过程有所降低

## 示例

原PDF文件：
```
这里放一个PDF示例图
```

转换后的Markdown：
```markdown
# 文档标题

这是正文内容，保持了原有的格式和结构。

| 表格 | 内容 | 示例 |
| ---- | ---- | ---- |
| 数据1 | 数据2 | 数据3 |

![图片描述](./images/image_1_1.png)
``` 