import pymupdf


def test_pymupdf():
    with pymupdf.open("/Users/daoji/Code/MemoMate/data/test/UOS控件问题 09.09.pdf") as doc:
        for page in doc:
            text = page.get_text(
                "dict",
                flags=pymupdf.TEXT_PRESERVE_IMAGES,
                # sort=True,
                # clip=None,
            )

            print(text)


def extract_text_with_image_tag(pdf_path):
    # 打开 PDF 文件
    pdf_document = pymupdf.open(pdf_path)

    # 用来保存输出的文本内容
    output_text = ""

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)

        # 获取页面的文本块和图片
        text = page.get_text("text")  # 提取文本
        images = page.get_images(full=True)  # 获取图片

        # 获取文本块的坐标
        blocks = page.get_text("blocks")  # 获取页面的文本块信息
        block_positions = [(block[0], block[1], block[2], block[3]) for block in blocks]  # [x0, y0, x1, y1]

        # 保存图片的位置信息
        image_positions = []
        for img_index, img in enumerate(images):
            xref = img[0]  # 图片的 xref
            bbox = img[1]  # 图片的边界框 (bbox: [x0, y0, x1, y1])
            image_positions.append((xref, bbox))

        # 替换文本中的图片
        modified_text = ""

        # 按顺序处理文本块和图片，确保图像插入位置合理
        for block in blocks:
            block_bbox = (block[0], block[1], block[2], block[3])  # 文本块位置
            block_text = block[4]  # 文本内容

            # 判断是否有图片重叠在文本块中
            for xref, bbox in image_positions:
                # 如果图片的 bbox 和文本块的 bbox 有交集
                if (
                    block_bbox[0] < bbox[2]  # block's left is less than image's right
                    and block_bbox[2] > bbox[0]  # block's right is more than image's left
                    and block_bbox[1] < bbox[3]  # block's bottom is less than image's top
                    and block_bbox[3] > bbox[1]  # block's top is more than image's bottom
                ):
                    # 在文本块中替换图片为 <image>
                    block_text = block_text.replace(block_text.strip(), "<image>")

            # 拼接到最终文本
            modified_text += block_text

        output_text += modified_text

    return output_text


def test_extract_text_with_images():
    print(extract_text_with_image_tag("/Users/daoji/Code/MemoMate/data/test/UOS控件问题 09.09.pdf"))
