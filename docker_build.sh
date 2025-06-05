#!/bin/bash
image_name=backend
image_tag=$(git rev-parse --short HEAD)
DOCKER_REGISTRY_ADDRESS="localhost:5009"

docker buildx build --platform linux/arm64 \
  -t ${DOCKER_REGISTRY_ADDRESS}/easyfin/${image_name}:${image_tag} \
  --push \
  -f Dockerfile .

# 打印镜像信息
echo "镜像名称: ${DOCKER_REGISTRY_ADDRESS}/easyfin/${image_name}:${image_tag}"
echo "镜像标签: ${image_tag}"
# 打印推送是否成功
if [ $? -eq 0 ]; then
  echo "镜像推送成功"
else
  echo "镜像推送失败"
fi