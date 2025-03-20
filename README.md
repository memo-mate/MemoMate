# MemoMate（待更新）

### 查询同类型问题/事故

### 配置环境

```bash
uv venv -p 3.12
uv sync
```

**初次开发前执行（一定要执行）**
`pre-commit install`

### 下载bge

```bash
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir ./bge-large-zh-v1.5
```

### 启动接口

```bash
fastapi dev app/main.py
```

### 核心模块文档

[核心模块文档](https://memo-docs.daojichang.eu.org/develop/core-modules.html)


### 分词器测试

```bash
uv run python -m app.rag.llm.tokenizers
```

![image-20250313154733459](https://cdn.jsdelivr.net/gh/daojiAnime/cdn@master/img/image-20250313154733459.png)


```bash
                                                       __
 /'\_/`\                             /'\_/`\          /\ \__
/\      \     __    ___ ___     ___ /\      \     __  \ \ ,_\    __
\ \ \__\ \  /'__`\/' __` __`\  / __`\ \ \__\ \  /'__`\ \ \ \/  /'__`\
 \ \ \_/\ \/\  __//\ \/\ \/\ \/\ \L\ \ \ \_/\ \/\ \L\.\_\ \ \_/\  __/
  \ \_\\ \_\ \____\ \_\ \_\ \_\ \____/\ \_\\ \_\ \__/.\_\\ \__\ \____\
   \/_/ \/_/\/____/\/_/\/_/\/_/\/___/  \/_/ \/_/\/__/\/_/ \/__/\/____/
```
