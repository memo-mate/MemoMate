# MemoMate（待更新）

### 查询同类型问题/事故

### 配置环境

```bash
uv venv -p 3.12
uv sync
```

### 下载bge

```bash
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir ./bge-large-zh-v1.5
```

### 核心模块文档

[核心模块文档](https://memo-docs.daojichang.eu.org/develop/core-modules.html)


### 分词器测试

```bash
uv run python -m app.rag.llm.tokenizers
```

![image-20250313154733459](https://cdn.jsdelivr.net/gh/daojiAnime/cdn@master/img/image-20250313154733459.png)