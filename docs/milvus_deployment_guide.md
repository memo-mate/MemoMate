# Milvus 向量数据库部署指南

## 📋 概述

Milvus 是一个开源向量数据库，专为 AI 应用设计，提供高性能的向量搜索能力。

**与 Qdrant 对比：**

| 特性         | Milvus              | Qdrant        |
| ------------ | ------------------- | ------------- |
| **性能**     | ⭐⭐⭐⭐⭐ 极高          | ⭐⭐⭐⭐ 高       |
| **扩展性**   | ⭐⭐⭐⭐⭐ 分布式        | ⭐⭐⭐ 单机/集群 |
| **易用性**   | ⭐⭐⭐ 中等            | ⭐⭐⭐⭐⭐ 简单    |
| **管理界面** | ✅ Attu              | ✅ Web UI      |
| **索引类型** | 多种（HNSW、IVF等） | HNSW          |
| **企业功能** | ✅ 完善              | ⭐⭐⭐ 基础      |
| **社区**     | ⭐⭐⭐⭐⭐ 活跃          | ⭐⭐⭐⭐ 活跃     |

---

## 🚀 快速开始

### 1. 启动 Milvus 服务

```bash
# 1. 确保环境变量配置
cat >> .env << EOF
VECTOR_DB_TYPE=milvus
MILVUS_HOST=milvus
MILVUS_PORT=19530
MILVUS_COLLECTION=memomate_dev
EOF

# 2. 启动 Milvus 和依赖服务
docker-compose -f docker-compose.yml -f docker-compose.milvus.yml up -d

# 3. 等待服务启动（约 30-60 秒）
docker logs -f memomate-milvus-1

# 4. 验证服务状态
curl http://localhost:9091/healthz
```

### 2. 访问 Attu 管理界面

打开浏览器访问：http://localhost:8001

- **Milvus 地址**：milvus:19530
- 可以查看集合、索引、数据等

### 3. 使用 Milvus

代码中使用方式与 Qdrant 相同：

```python
from app.rag.embedding import get_embeddings
from app.rag.embedding.embedding_db.custom_milvus import MilvusVectorStore

embeddings = get_embeddings()

# 创建向量存储
vector_store = MilvusVectorStore(
    collection_name="my_collection",
    embeddings=embeddings,
    connection_args={
        "host": "localhost",
        "port": 19530
    }
)

# 添加文档
vector_store.add_texts([
    "这是第一个文档",
    "这是第二个文档"
])

# 搜索
results = vector_store.similarity_search("查询文本", k=4)
```

---

## 📦 架构说明

### 组件

Milvus Standalone 包含以下组件：

```
┌─────────────┐
│   Milvus    │  ← 向量搜索引擎
└──────┬──────┘
       │
   ┌───┴────┬────────┐
   │        │        │
┌──┴──┐  ┌──┴──┐  ┌──┴──┐
│Etcd │  │MinIO│  │Attu │
└─────┘  └─────┘  └─────┘
 元数据    对象存储   管理界面
```

- **Milvus**：向量搜索引擎（端口 19530）
- **Etcd**：元数据存储
- **MinIO**：对象存储（端口 9000/9001）
- **Attu**：Web 管理界面（端口 8001）

### 数据持久化

```
data/
├── milvus/
│   ├── etcd/      # Etcd 数据
│   ├── minio/     # MinIO 数据
│   └── milvus/    # Milvus 数据
```

---

## ⚙️ 配置说明

### 环境变量

```bash
# 向量数据库类型
VECTOR_DB_TYPE=milvus

# Milvus 连接配置
MILVUS_HOST=milvus          # 服务地址
MILVUS_PORT=19530           # gRPC 端口
MILVUS_USER=                # 用户名（可选）
MILVUS_PASSWORD=            # 密码（可选）
MILVUS_DB_NAME=default      # 数据库名称
MILVUS_COLLECTION=memomate_dev  # 集合名称
```

### Milvus 配置文件

编辑 `milvus-config.yaml` 调整性能参数：

```yaml
# 内存限制（根据实际情况调整）
queryNode:
  cache:
    enabled: true
    memoryLimit: 2147483648  # 2GB

# 段大小配置
dataCoord:
  segment:
    maxSize: 512    # MB

# gRPC 配置
grpc:
  serverMaxRecvSize: 268435456  # 256MB
  serverMaxSendSize: 268435456  # 256MB
```

---

## 🎯 索引配置

### 常用索引类型

#### 1. HNSW（推荐，默认）

```python
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {
        "M": 8,              # 连接数（4-64）
        "efConstruction": 64  # 构建参数（8-512）
    }
}
```

- **优点**：高精度、高性能
- **缺点**：内存占用大
- **适用**：小到中等规模（< 10M 向量）

#### 2. IVF_FLAT

```python
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {
        "nlist": 1024  # 聚类中心数
    }
}
```

- **优点**：精度高、内存占用中等
- **缺点**：构建较慢
- **适用**：中等规模（1M-10M 向量）

#### 3. IVF_PQ

```python
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_PQ",
    "params": {
        "nlist": 1024,  # 聚类中心数
        "m": 8,         # PQ 分段数
        "nbits": 8      # 每个分段的位数
    }
}
```

- **优点**：内存占用小
- **缺点**：精度较低
- **适用**：大规模（> 10M 向量）

### 距离度量

```python
# 余弦相似度（推荐）
metric_type = "COSINE"

# 欧几里得距离
metric_type = "L2"

# 内积
metric_type = "IP"
```

---

## 📊 性能调优

### 1. 搜索参数

```python
search_params = {
    "metric_type": "COSINE",
    "params": {
        "ef": 64  # HNSW 搜索参数（越大越精确但越慢）
        # 或 "nprobe": 16  # IVF 搜索参数
    }
}
```

### 2. 内存优化

```yaml
# milvus-config.yaml
queryNode:
  cache:
    enabled: true
    memoryLimit: 4294967296  # 4GB（根据实际情况调整）
```

### 3. 批量插入

```python
# 批量插入性能更好
batch_size = 1000
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    vector_store.add_texts(batch_texts)
```

---

## 🔧 运维管理

### 1. 查看服务状态

```bash
# 检查所有服务
docker-compose -f docker-compose.milvus.yml ps

# 查看 Milvus 日志
docker logs -f memomate-milvus-1

# 健康检查
curl http://localhost:9091/healthz
curl http://localhost:9091/metrics  # Prometheus 指标
```

### 2. 数据备份

```bash
# 备份数据目录
tar -czf milvus-backup-$(date +%Y%m%d).tar.gz \
  data/milvus/etcd \
  data/milvus/minio \
  data/milvus/milvus

# 恢复
tar -xzf milvus-backup-20240101.tar.gz
```

### 3. 性能监控

访问 Prometheus 指标：http://localhost:9091/metrics

关键指标：
- `milvus_querynode_search_latency_bucket` - 搜索延迟
- `milvus_datanode_consume_bytes_count` - 数据消费
- `process_resident_memory_bytes` - 内存使用

### 4. 集合管理

使用 Attu 界面（http://localhost:8001）：
- 查看集合列表
- 查看索引状态
- 查看数据统计
- 执行搜索测试

或使用 Python 代码：

```python
from pymilvus import utility, connections

# 连接
connections.connect(host="localhost", port=19530)

# 列出所有集合
collections = utility.list_collections()

# 查看集合信息
utility.get_collection_stats("memomate_dev")

# 删除集合
utility.drop_collection("old_collection")
```

---

## 🐛 故障排查

### Q1: Milvus 启动失败

**检查：**
```bash
# 查看日志
docker logs memomate-milvus-1

# 检查依赖服务
docker logs memomate-etcd-1
docker logs memomate-minio-1
```

**常见原因：**
- Etcd 或 MinIO 未就绪
- 端口冲突（19530、9000、9001）
- 内存不足

**解决：**
```bash
# 重启服务
docker-compose -f docker-compose.milvus.yml restart

# 检查端口占用
lsof -i :19530
```

### Q2: 连接超时

**检查网络：**
```bash
# 测试连接
docker exec -it memomate-backend-1 ping milvus

# 检查端口
docker exec -it memomate-backend-1 telnet milvus 19530
```

### Q3: 内存不足

**调整配置：**
```yaml
# milvus-config.yaml
queryNode:
  cache:
    memoryLimit: 1073741824  # 减少到 1GB
```

### Q4: 搜索性能慢

**优化：**
1. 调整搜索参数（降低 ef/nprobe）
2. 使用更快的索引类型
3. 增加内存限制
4. 使用 GPU 加速（需要 GPU 版本）

---

## 📚 参考资源

- **官方文档**：https://milvus.io/docs
- **Attu 文档**：https://github.com/zilliztech/attu
- **性能调优**：https://milvus.io/docs/performance_faq.md
- **索引选择**：https://milvus.io/docs/index.md

---

## 🔄 从 Qdrant 迁移

### 1. 数据导出（Qdrant）

```python
from app.rag.embedding.embedding_db.custom_qdrant import QdrantVectorStore

# 连接 Qdrant
qdrant_store = QdrantVectorStore(
    collection_name="memomate_dev",
    embeddings=embeddings,
    url="http://localhost:6333"
)

# 导出所有数据
documents = qdrant_store.get_all()
```

### 2. 数据导入（Milvus）

```python
from app.rag.embedding.embedding_db.custom_milvus import MilvusVectorStore

# 连接 Milvus
milvus_store = MilvusVectorStore(
    collection_name="memomate_dev",
    embeddings=embeddings,
    connection_args={"host": "localhost", "port": 19530}
)

# 批量导入
texts = [doc.page_content for doc in documents]
metadatas = [doc.metadata for doc in documents]
milvus_store.add_texts(texts, metadatas=metadatas)
```

### 3. 更新配置

```bash
# 停止 Qdrant
docker-compose down qdrant

# 更新环境变量
echo "VECTOR_DB_TYPE=milvus" >> .env

# 启动 Milvus
docker-compose -f docker-compose.milvus.yml up -d
```

---

## 💡 最佳实践

### 开发环境

```bash
# 使用 Qdrant（更简单）
docker-compose up -d qdrant
```

### 生产环境

```bash
# 使用 Milvus（更强大）
docker-compose -f docker-compose.milvus.yml up -d

# 配置监控
# 配置定期备份
# 配置资源限制
```

### 大规模数据

- **< 1M 向量**：Qdrant 或 Milvus 都可以
- **1M-10M 向量**：推荐 Milvus + HNSW
- **> 10M 向量**：推荐 Milvus + IVF_PQ

---

## 🎉 总结

✅ **高性能**：支持大规模向量搜索
✅ **企业级**：完善的分布式支持
✅ **易管理**：Attu 管理界面
✅ **灵活**：多种索引和配置选项
✅ **兼容**：与 Qdrant 接口类似，易于切换

**立即开始使用 Milvus！** 🚀
