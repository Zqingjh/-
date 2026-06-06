import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["NO_PROXY"] = "localhost, 127.0.0.1, ::1"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from core.config import config


def build_vector_database():
    print(f"知识库路径: {config.raw_file}")
    loader = TextLoader(config.raw_file, encoding="utf-8")
    documents = loader.load()
    print(f"加载完成，{len(documents)} 个文档，开始分块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    print(f"拆分为 {len(docs)} 个文本块")
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    print("加载 embedding 完成，构建向量库...")
    vector_store = Chroma.from_documents(
        docs, embeddings,
        collection_name=config.collection_name,
        persist_directory=config.vector_db_dir
    )
    print(f"构建完成！已保存至 {config.vector_db_dir}")


if __name__ == "__main__":
    build_vector_database()
