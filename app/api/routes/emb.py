import uuid

from fastapi import APIRouter, HTTPException, status

from app.core.log_adapter import logger
from app.schemas.emb import (
    AddVectorRequest,
    AddVectorResponse,
    BatchAddVectorsRequest,
    BatchAddVectorsResponse,
    CollectionListResponse,
    CreateCollectionRequest,
    CreateCollectionResponse,
    DeleteCollectionResponse,
    DeleteVectorRequest,
    DeleteVectorResponse,
    DocumentModel,
    SearchVectorRequest,
    SearchVectorResponse,
)
from app.utils.emb import convert_dict_to_qdrant_filter, get_vector_store

router = APIRouter()


@router.get(
    "/collections",
    response_model=CollectionListResponse,
    summary="获取集合列表",
    description="获取向量库中的所有集合",
)
def list_collections() -> CollectionListResponse:
    try:
        vector_store = get_vector_store("_temp")
        collections = vector_store.list_collections()
        return CollectionListResponse(success=True, collections=collections)
    except Exception as e:
        logger.exception("获取集合列表失败", exc_info=e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"获取集合列表失败: {str(e)}")


@router.post(
    "/collections",
    response_model=CreateCollectionResponse,
    summary="创建集合",
    description="在向量库中创建新集合",
)
def create_collection(request: CreateCollectionRequest) -> CreateCollectionResponse:
    try:
        vector_store = get_vector_store(request.collection_name)
        vector_store.create_collection_if_not_exists(request.collection_name, force=request.force)
        return CreateCollectionResponse(success=True, message=f"成功创建集合: {request.collection_name}")
    except Exception as e:
        logger.exception(f"创建集合失败: {request.collection_name}", exc_info=e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"创建集合失败: {str(e)}")


@router.delete(
    "/collections/{collection_name}",
    response_model=DeleteCollectionResponse,
    summary="删除集合",
    description="删除向量库中的指定集合",
)
def delete_collection(collection_name: str) -> DeleteCollectionResponse:
    try:
        vector_store = get_vector_store("_temp")
        success = vector_store.delete_collection(collection_name)
        if success:
            return DeleteCollectionResponse(success=True, message=f"成功删除集合: {collection_name}")
        else:
            return DeleteCollectionResponse(success=False, message=f"删除集合失败: {collection_name}")
    except Exception as e:
        logger.exception(f"删除集合失败: {collection_name}", exc_info=e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"删除集合失败: {str(e)}")


@router.post(
    "/vectors",
    response_model=AddVectorResponse,
    summary="添加向量",
    description="向指定集合中添加文本向量",
)
def add_vector(request: AddVectorRequest) -> AddVectorResponse:
    try:
        vector_store = get_vector_store(request.collection_name)

        # 转换元数据格式
        metadata = {}
        if request.metadata:
            metadata = {item.key: item.value for item in request.metadata}

        # 生成ID或使用提供的ID
        vector_id = request.id or str(uuid.uuid4())

        # 添加文本到向量库
        ids = vector_store.add_texts(texts=[request.text], metadatas=[metadata] if metadata else None, ids=[vector_id])

        if ids and len(ids) > 0:
            return AddVectorResponse(success=True, message="成功添加向量", vector_id=ids[0])
        else:
            return AddVectorResponse(success=False, message="向量添加失败，未返回ID", vector_id=None)
    except Exception as e:
        logger.exception("添加向量失败", exc_info=e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"添加向量失败: {str(e)}")


@router.post(
    "/vectors/batch",
    response_model=BatchAddVectorsResponse,
    summary="批量添加向量",
    description="向指定集合中批量添加文本向量",
)
def batch_add_vectors(request: BatchAddVectorsRequest) -> BatchAddVectorsResponse:
    try:
        vector_store = get_vector_store(request.collection_name)

        # 处理ID，如果未提供则生成
        ids = request.ids
        if not ids:
            ids = [str(uuid.uuid4()) for _ in request.texts]

        # 添加文本到向量库
        result_ids = vector_store.add_texts(texts=request.texts, metadatas=request.metadatas, ids=ids)

        return BatchAddVectorsResponse(
            success=True, message=f"成功添加 {len(result_ids)} 个向量", vector_ids=result_ids
        )
    except Exception as e:
        logger.exception("批量添加向量失败", exc_info=e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"批量添加向量失败: {str(e)}")


@router.delete(
    "/vectors",
    response_model=DeleteVectorResponse,
    summary="删除向量",
    description="从指定集合中删除向量",
)
def delete_vectors(request: DeleteVectorRequest) -> DeleteVectorResponse:
    try:
        vector_store = get_vector_store(request.collection_name)
        # 将list[str]转换为list[int | str]类型，实际使用时这两种类型是兼容的
        success = vector_store.delete_vectors(request.collection_name, request.vector_ids)

        if success:
            return DeleteVectorResponse(success=True, message=f"成功删除 {len(request.vector_ids)} 个向量")
        else:
            return DeleteVectorResponse(success=False, message="删除向量失败")
    except Exception as e:
        logger.exception("删除向量失败", exc_info=e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"删除向量失败: {str(e)}")


@router.post(
    "/search",
    response_model=SearchVectorResponse,
    summary="向量搜索",
    description="在指定集合中搜索相似文本",
)
def search_vectors(request: SearchVectorRequest) -> SearchVectorResponse:
    try:
        vector_store = get_vector_store(request.collection_name)
        # 转换过滤条件
        qdrant_filter = convert_dict_to_qdrant_filter(request.filter)

        results = []
        if request.search_type == "similarity":
            docs = vector_store.similarity_search(query=request.query, k=request.top_k, filter=qdrant_filter)

            # 转换结果
            for doc in docs:
                score = doc.metadata.get("_score")
                results.append(DocumentModel(page_content=doc.page_content, metadata=doc.metadata, score=score))

        elif request.search_type == "mmr":
            docs = vector_store.max_marginal_relevance_search(
                query=request.query, k=request.top_k, filter=qdrant_filter
            )

            # 转换结果
            for doc in docs:
                score = doc.metadata.get("_score")
                results.append(DocumentModel(page_content=doc.page_content, metadata=doc.metadata, score=score))

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"不支持的搜索类型: {request.search_type}"
            )

        return SearchVectorResponse(success=True, message=f"搜索成功，找到 {len(results)} 条结果", results=results)
    except Exception as e:
        logger.exception("向量搜索失败", exc_info=e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"向量搜索失败: {str(e)}")


@router.get(
    "/vectors/{collection_name}/{vector_id}",
    response_model=DocumentModel,
    summary="获取向量",
    description="根据ID获取向量对应的文档",
)
def get_vector(collection_name: str, vector_id: str) -> DocumentModel:
    try:
        vector_store = get_vector_store(collection_name)
        docs = vector_store.get_by_ids([vector_id])

        if not docs or len(docs) == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"未找到向量: {vector_id}")

        doc = docs[0]
        score = doc.metadata.get("_score")

        return DocumentModel(page_content=doc.page_content, metadata=doc.metadata, score=score)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"获取向量失败: {vector_id}", exc_info=e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"获取向量失败: {str(e)}")
