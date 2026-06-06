import os

# 1. 强制设置国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 引入 HuggingFace 的下载工具
from huggingface_hub import snapshot_download

print("🚀 开始从国内镜像下载 bert-base-chinese 模型...")
print("（模型约 400MB，请耐心等待）")

# 3. 下载到指定的本地缓存目录（例如 ./models/bert-base-chinese-cache）
#    或者不指定 local_dir，让它下载到默认缓存目录（~/.cache/huggingface/hub）
snapshot_download(
    repo_id="bert-base-chinese",           # 改成你要的模型（如BAAI/bge-small-zh-v1.5）
    local_dir="./models/bert-base-chinese", # 可选：直接下载到本地文件夹（如./models/bge-small-zh-v1.5）
    resume_download=True,
    max_workers=4
)

print("✅ bert-base-chinese 下载完成！")