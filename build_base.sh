#!/bin/bash
image_name=backend
# 使用uv.lock的md5作为镜像标签，截取前8位
image_tag=$(md5sum uv.lock | awk '{print $1}' | cut -c 1-8)
DOCKER_REGISTRY_ADDRESS="localhost:5009"

docker buildx build --platform linux/arm64 \
  -t ${DOCKER_REGISTRY_ADDRESS}/easyfin/${image_name}:${image_tag} \
  --push \
  -f Dockerfile.base .

# 打印镜像信息
echo "镜像名称: ${DOCKER_REGISTRY_ADDRESS}/easyfin/${image_name}:${image_tag}"
echo "镜像标签: ${image_tag}"
# 打印推送是否成功
if [ $? -eq 0 ]; then
  echo "镜像推送成功"
else
  echo "镜像推送失败"
fi