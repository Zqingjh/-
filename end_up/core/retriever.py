# retriever.py - RAG 检索模块
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from core.config import config
import os

def init_retriever():
    try:
        embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
        vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=embeddings,
            persist_directory=config.vector_db_dir
        )
        return vector_store.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"[配置化] 知识库加载失败: {e}")
        return None


def format_docs(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", config.domain_name + "规章")
        source = os.path.basename(source)
        parts.append(f"[来源{i}: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)
