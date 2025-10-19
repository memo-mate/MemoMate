# 嵌入模型快速开始

##  问题

使用 `--workers 4` 时，旧方案会导致：
- ❌ 模型加载 4 次
- ❌ 浪费 6.9 GB 内存
- ❌ 启动时间 4 倍

## ✅ 新方案

统一使用 OpenAI 兼容 API（vLLM/Ollama），彻底解决问题！

---

## 🚀 3 分钟快速部署

### 方式 1：vLLM（推荐生产）

```bash
# 1. 配置
echo "EMBEDDING_API_BASE=http://vllm-embedding:8000/v1" >> .env
echo "EMBEDDING_MODEL=bge-m3" >> .env

# 2. 启动
docker-compose -f docker-compose.vllm.yml up -d

# 3. 验证
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": "测试"}'
```

### 方式 2：Ollama（推荐开发）

```bash
# 1. 配置
echo "EMBEDDING_API_BASE=http://ollama:11434/v1" >> .env
echo "EMBEDDING_MODEL=bge-m3" >> .env

# 2. 启动
docker-compose -f docker-compose.ollama.yml up -d ollama

# 3. 拉取模型
docker exec -it memomate-ollama-1 ollama pull bge-m3

# 4. 验证
curl http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": "测试"}'

# 5. 启动应用
docker-compose -f docker-compose.ollama.yml up -d
```

### 方式 3：OpenAI API

```bash
# 1. 配置
echo "EMBEDDING_API_BASE=https://api.openai.com/v1" >> .env
echo "EMBEDDING_MODEL=text-embedding-3-large" >> .env
echo "EMBEDDING_API_KEY=sk-xxx" >> .env

# 2. 启动
docker-compose up -d
```

---

## 📊 对比

| 方案     | 内存   | QPS | 适用场景 |
| -------- | ------ | --- | -------- |
| vLLM     | 3.2 GB | 850 | 生产环境 |
| Ollama   | 3.5 GB | 420 | 开发环境 |
| OpenAI   | 0 GB   | 600 | 快速上线 |
| ❌ 旧方案 | 9.2 GB | 850 | 不推荐   |

**内存节省：66%**

---

## 🗄️ 向量数据库

项目支持两种向量数据库：

### Qdrant（默认，推荐开发）

```bash
# 已包含在默认配置中
docker-compose up -d qdrant
```

- ✅ 简单易用
- ✅ 快速启动
- ✅ 适合中小规模

### Milvus（推荐生产）

```bash
# 使用 Milvus
docker-compose -f docker-compose.milvus.yml up -d

# 访问管理界面
open http://localhost:8001
```

- ✅ 高性能
- ✅ 企业级功能
- ✅ 适合大规模

**详细对比** → [milvus_deployment_guide.md](milvus_deployment_guide.md)

---

## 📚 完整文档

- **嵌入模型部署** → [embedding_deployment_guide.md](embedding_deployment_guide.md)
- **Milvus 数据库** → [milvus_deployment_guide.md](milvus_deployment_guide.md)

---

## 💡 立即开始

**生产环境：**
```bash
docker-compose -f docker-compose.vllm.yml up -d
```

**开发环境：**
```bash
docker-compose -f docker-compose.ollama.yml up -d
```

搞定！🎉
