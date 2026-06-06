# config.py
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
if os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")
if os.getenv("NO_PROXY"):
    os.environ["NO_PROXY"] = os.getenv("NO_PROXY")
class Config:
    def __init__(self, config_path: str = None):
        path = Path(config_path) if config_path else ROOT_DIR / "config.yaml"
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        domain = raw.get("domain", {})
        self.domain_name = domain.get("name", "校园")
        self.assistant_name = domain.get("assistant_name", "助手")
        model = raw.get("model", {})
        self.finetuned_model = model.get("finetuned", "college_bot")
        self.base_model = model.get("base", "qcwind/qwen2.5-7B-instruct-Q4_K_M")
        self.ollama_host = model.get("ollama_host", "http://127.0.0.1:11434")
        kb = raw.get("knowledge_base", {})
        self.raw_file = str(ROOT_DIR / kb.get("raw_file", "./knowledge_base/manual.txt"))
        self.vector_db_dir = str(ROOT_DIR / kb.get("vector_db_dir", "./chroma_data"))
        self.collection_name = kb.get("collection_name", "domain_handbook")
        self.chunk_size = kb.get("chunk_size", 500)
        self.chunk_overlap = kb.get("chunk_overlap", 50)
        emb = raw.get("embedding", {})
        self.embedding_model = emb.get("model_name", "./models/bge-small-zh-v1.5")
        prompt = raw.get("prompt", {})
        self.system_template = prompt.get("system_template", "")
        eval_cfg = raw.get("evaluation", {})
        self.test_cases = str(ROOT_DIR / eval_cfg.get("test_cases", "./test_cases.json"))
        self.consistency_runs = eval_cfg.get("consistency_runs", 3)
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or ""
        ui_cfg = raw.get("ui", {})
        self.ui_port = ui_cfg.get("port", 8080)
        self.ui_title = ui_cfg.get("title", "数字辅导员")
        self.avatar_user = "https://api.dicebear.com/7.x/avataaars/svg?seed=User&backgroundColor=b6e3f4"
        self.avatar_bot = "https://api.dicebear.com/7.x/bottts/svg?seed=Ollama&backgroundColor=c0aede"
config = Config()
