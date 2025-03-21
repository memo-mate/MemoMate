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
### 项目结构

```bash
 .
├──  alembic.ini     # 数据库版本管理配置
├──  app
│   ├──  __init__.py
│   ├──  alembic     # 数据库版本管理
│   ├──  api         # 接口
│   │   ├──  __init__.py
│   │   ├──  demo
│   │   │   ├──  __init__.py
│   │   │   ├──  sse.py       # sse demo接口
│   │   │   └──  websocket.py # websocket demo接口
│   │   ├──  deps.py # 依赖注入
│   │   ├──  main.py # 接口路由
│   │   └──  routes  # 接口逻辑
│   │       └──  ...
│   ├──  backend_pre_start.py
│   ├──  core        # 核心模块
│   │   ├──  __init__.py
│   │   ├──  __init__.pyi
│   │   ├──  config.py                 # 系统全局变量配置
│   │   ├──  consts.py                 # 常量定义
│   │   ├──  db.py                     # 数据库引擎
│   │   ├──  log_adapter.py            # 日志适配器
│   │   ├──  middlewares.py            # api 中间件
│   │   ├──  responses.py              # api 响应格式封装
│   │   └──  security.py               # 鉴权
│   ├──  crud        # 数据库读写模块
│   │   ├──  __init__.py
│   │   └──  user.py # user 表操作封装
│   ├──  document_parsing              # 文档解析模块
│   │   ├──  __init__.py
│   │   ├──  document_processor.py     #
│   │   ├──  excel_paser.py            # 表格解析
│   │   └──  video_subtitle_extractor.py # 音视频解析
│   ├──  email-templates               # 邮件通知--模板文件
│   │   ├──  build
│   │   │   ├──  new_account.html
│   │   │   ├──  reset_password.html
│   │   │   └──  test_email.html
│   │   └──  src
│   │       ├──  new_account.mjml
│   │       ├──  reset_password.mjml
│   │       └──  test_email.mjml
│   ├──  enums                         # 枚举模块
│   │   ├──  __init__.py
│   │   ├──  queue.py                  # 消息队列相关枚举
│   │   └──  task.py                   # 任务相关枚举
│   ├──  initial_data.py
│   ├──  initial_demo_data.py
│   ├──  main.py                       # fastapi app定义
│   ├──  models                        # 数据库表模块
│   │   ├──  __init__.py
│   │   ├──  task.py
│   │   └──  user.py                   # 用户表
│   ├──  queue_start.py                # 队列启动脚本
│   ├──  rag                           # rag模块
│   │   ├──  __init__.py
│   │   ├──  embedding                 # 嵌入模块
│   │   │   ├──  __init__.py
│   │   │   ├──  embed_db.py
│   │   │   ├──  embeeding_model.py
│   │   │   └──  examples.py
│   │   ├──  llm                       # llm模块
│   │   │   ├──  __init__.py
│   │   │   ├──  completions.py
│   │   │   └──  tokenizers.py
│   │   └──  reranker                  # 重排序模块
│   │       ├──  __init__.py
│   │       ├──  base.py
│   │       ├──  cross_encoder.py
│   │       ├──  examples.py
│   │       ├──  llm_reranker.py
│   │       ├──  reranking_retriever.py
│   │       └──  test_reranker.py
│   ├──  schemas                     # 数据表结构定义
│   │   ├──  __init__.py
│   │   ├──  auth.py
│   │   ├──  paser.py
│   │   └──  user.py
│   ├──  tests                         # 单元测试模块
│   │   ├──  __init__.py
│   │   ├──  api                       # api测试用例
│   │   │   ├──  __init__.py
│   │   │   └──  routes
│   │   │       ├──  __init__.py
│   │   │       ├──  test_login.py
│   │   │       ├──  test_users.py
│   │   │       └──  test_websocket.py
│   │   ├──  conftest.py                 # 测试用例配置
│   │   ├──  core                        # 核心模块测试用例
│   │   │   └──  test_log.py
│   │   ├──  crud                        # 数据库读写模块测试用例
│   │   │   ├──  __init__.py
│   │   │   └──  test_user.py
│   │   ├──  rag                         # rag模块测试用例
│   │   │   ├──  __init__.py
│   │   │   ├──  test_llm.py
│   │   │   └──  test_tokenizer.py
│   │   ├──  scripts                     # 脚本模块测试用例
│   │   │   ├──  __init__.py
│   │   │   ├──  test_backend_pre_start.py
│   │   │   └──  test_test_pre_start.py
│   │   └──  utils                       # 工具模块测试用例
│   │       ├──  __init__.py
│   │       ├──  test_task_queue.py
│   │       ├──  user.py
│   │       └──  utils.py
│   ├──  tests_pre_start.py              # 测试用例启动脚本
│   └──  utils                           # 工具模块
│       ├──  __init__.py
│       ├──  art_name.py
│       ├──  email_tools.py
│       ├──  math_tools.py
│       └──  task_queue.py
├──  data                            # 数据文件
├──  docker-compose.yml              # docker compose配置
├──  Dockerfile                      # dockerfile配置
├──  kafka.yml                       # kafka配置
├──  pyproject.toml                  # 项目配置
├──  pytest.ini                      # pytest配置
├── 󰂺 README.md                       # 项目说明Markdown
├──  scripts                         # 脚本模块
│   ├──  dev.sh                        # 开发环境脚本
│   ├──  format.sh                     # 格式化脚本
│   ├──  lint.sh                       # 代码检查脚本
│   ├──  test.sh                       # 测试脚本
│   └──  tests-start.sh                # 测试用例启动脚本
├──  typings                         # 类型定义(给IDE看的)
│   └──  transformers                  # token 模块类型注解
│       └──  __init__.pyi
└──  uv.lock                         # 依赖锁文件

```
