# evaluate.py — 问答系统消融实验与多指标评估脚本
# [配置化] 敏感信息从 .env 读取，模型/路径从 config.yaml 读取

import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# [配置化] 从 core.config 读取领域配置
from core.config import config

# [配置化] 原来硬编码的绝对路径改为相对路径
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_BASE_DIR, "models")
os.environ["HF_HOME"] = os.path.join(_BASE_DIR, "models")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import time
import statistics
from datetime import datetime
from pathlib import Path
import transformers
transformers.logging.set_verbosity_error()
import jieba
import logging
jieba.setLogLevel(logging.ERROR)

import ollama
from openai import OpenAI
from rouge import Rouge
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# [配置化] 模型名称和 host 从 config 读取
FINETUNED_MODEL = config.finetuned_model
BASE_MODEL = config.base_model
OLLAMA_HOST = config.ollama_host
JUDGE_MODEL = "deepseek-r1"
LOCAL_BERT_PATH = os.path.join(_BASE_DIR, "models", "bert-base-chinese")
DEEPSEEK_API_KEY = config.deepseek_api_key  # [配置化] 从 .env 读取
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
TEST_CASES_FILE = config.test_cases  # [配置化] 从 config 读取
OUTPUT_DIR = Path("results")
CONSISTENCY_RUNS = config.consistency_runs
CHROMA_PERSIST_DIR = config.vector_db_dir  # [配置化] 从 config 读取
COLLECTION_NAME = config.collection_name   # [配置化] 从 config 读取

print("\n[系统初始化] 开始加载各项服务，预加载模型可能需要十几秒...")
try:
    _bert_scorer = BERTScorer(model_type=LOCAL_BERT_PATH, num_layers=9, lang="zh")
    print("  ? BERTScore 模型已常驻内存")
except Exception as e:
    print(f"  ? BERTScore 加载失败: {e}")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    _embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    _vectorstore = Chroma(collection_name=COLLECTION_NAME, embedding_function=_embeddings, persist_directory=CHROMA_PERSIST_DIR)
    _retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})
    RAG_AVAILABLE = True
    print("  ? RAG 知识库加载成功\n")
except Exception as e:
    RAG_AVAILABLE = False
    print(f"  ? RAG 知识库加载失败: {e}\n")

def retrieve_context(question: str) -> str:
    if not RAG_AVAILABLE: return ""
    try:
        docs = _retriever.invoke(question)
        return "\n\n".join([f"[来源{i+1}: {os.path.basename(d.metadata.get('source', '规章'))}]\n{d.page_content}" for i, d in enumerate(docs)])
    except Exception:
        return ""

def build_prompt(question: str, use_rag: bool, context: str) -> list[dict]:
    template = config.system_template if config.system_template else (
        "你是一个智能且贴心的{assistant_name}。\n"
        "请结合你自身的{domain}知识储备，以及以下提供的【参考最新规章】，综合为学生解答。\n\n"
        "【参考最新规章】：\n{context}\n\n"
        "排版与回答要求：\n"
        "1. 知识融合策略：\n"
        "   - 优先查阅【参考最新规章】。如规章中包含相关政策，必须以此为准。\n"
        "   - 如果参考规章未提及、或信息不全，请直接调用你自身内化的知识进行详细解答。\n"
        "   - 绝不胡编乱造。如果连你自己的知识库里都没有确切答案，请如实告知并引导学生咨询具体学院的辅导员。\n"
        "2. 必须严格遵循以下三段式排版结构输出：\n"
        "   - 第一段：贴心的问候与核心结论。\n"
        "   - 第二段：具体的解答细节（如需分点，必须使用 1. 2. 3. 且每点之间独立换行）。\n"
        "   - 第三段：总结、建议或鼓励的话语（必须另起两行，绝对不能和上面的分点内容挤在一起！）。\n"
    )
    context_text = context if (use_rag and context) else ""
    filled = template.format(assistant_name=config.assistant_name, domain=config.domain_name, context=context_text if context_text else "（当前无相关的实时规章补充，请直接调用你的记忆回答）")
    return [{"role": "system", "content": filled}, {"role": "user", "content": question}]

def query_model(model: str, messages: list[dict], retries: int = 3) -> str:
    ollama_client = ollama.Client(host=OLLAMA_HOST)
    for attempt in range(retries):
        try:
            resp = ollama_client.chat(model=model, messages=messages)
            return resp["message"]["content"].strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"[ERROR] {e}"

def calc_bleu(hypothesis: str, reference: str) -> float:
    try:
        if not hypothesis.strip() or not reference.strip(): return 0.0
        hyp_words = list(jieba.cut(hypothesis))
        ref_words = list(jieba.cut(reference))
        smooth = SmoothingFunction().method1
        return round(sentence_bleu([ref_words], hyp_words, smoothing_function=smooth), 4)
    except Exception:
        return 0.0

def calc_rouge_l(hypothesis: str, reference: str) -> float:
    try:
        if not hypothesis.strip() or not reference.strip(): return 0.0
        hyp_spaced = " ".join(jieba.cut(hypothesis))
        ref_spaced = " ".join(jieba.cut(reference))
        scores = Rouge().get_scores(hyp_spaced, ref_spaced)
        return round(scores[0]["rouge-l"]["f"], 4)
    except Exception:
        return 0.0

def calc_bert_score(hypothesis: str, reference: str) -> float:
    try:
        if not hypothesis.strip() or not reference.strip(): return 0.0
        P, R, F1 = _bert_scorer.score([hypothesis], [reference])
        return round(F1[0].item(), 4)
    except Exception:
        return 0.0

def calc_llm_judge(question: str, reference: str, answer: str) -> dict:
    prompt = f"""你是一位严格的评价专家。你的任务是对下面系统的回答进行打分。
【学生问题】：{question}
【标准答案】：{reference}
【系统回答】：{answer}
请严格根据以下三个维度，各给出0.0到1.0的评分：
1. 事实准确性：系统回答是否准确覆盖了标准答案的核心信息？
2. 服务态度：系统回答是否具备贴心、有温度的服务语态？
3. 结构排版：系统回答是否结构清晰、合理使用分点？
输出严格的JSON格式，例如：{{"事实准确性": {{"分数": 0.85}}, "服务态度": {{"分数": 0.90}}, "结构排版": {{"分数": 0.80}}}}"""
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        response = client.chat.completions.create(model=DEEPSEEK_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        import re
        match = re.search(r"\{[\s\S]*\}", response.choices[0].message.content)
        if match:
            result = json.loads(match.group())
            scores = {"准确性": result.get("事实准确性", {}).get("分数", 0), "相关性": result.get("服务态度", {}).get("分数", 0), "语态自然度": result.get("结构排版", {}).get("分数", 0)}
            scores["综合均分"] = round(sum(scores.values()) / len(scores), 2)
            return scores
    except Exception: pass
    return {"准确性": 0, "相关性": 0, "语态自然度": 0, "综合均分": 0}

def calc_consistency(model: str, messages: list[dict], runs: int = 3) -> float:
    answers = [query_model(model, messages) for _ in range(runs)]
    scores = [calc_rouge_l(answers[i], answers[j]) for i in range(len(answers)) for j in range(i + 1, len(answers))]
    return round(statistics.mean(scores), 4) if scores else 0.0

EXPERIMENTS = [
    {"id": "A", "label": "基座模型（无微调，无RAG）", "model": BASE_MODEL, "use_rag": False},
    {"id": "B", "label": "微调模型（有微调，无RAG）", "model": FINETUNED_MODEL, "use_rag": False},
    {"id": "C", "label": "基座+RAG（无微调，有RAG）", "model": BASE_MODEL, "use_rag": True},
    {"id": "D", "label": "微调+RAG（完整系统）", "model": FINETUNED_MODEL, "use_rag": True},
]

def run_evaluation():
    base_dir = Path("results")
    base_dir.mkdir(exist_ok=True)
    run_idx = 1
    while (base_dir / f"run_{run_idx}").exists():
        run_idx += 1
    output_dir = base_dir / f"run_{run_idx}"
    output_dir.mkdir()
    print(f"{'='*60}\n 领域问答系统消融实验评估\n 测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" 本次数据将实时保存在：{output_dir}\n{'='*60}\n")
    with open(TEST_CASES_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    all_results = {}
    for exp in EXPERIMENTS:
        exp_id, exp_label, model, use_rag = exp["id"], exp["label"], exp["model"], exp["use_rag"]
        print(f"\n{'='*50}\n| 实验组{exp_id}：{exp_label}\n  模型：{model} | RAG：{'开启' if use_rag else '关闭'}\n{'='*50}")
        exp_results = []
        for i, case in enumerate(test_cases, 1):
            q_id, category, question, reference = case["id"], case.get("category", "未分类"), case["question"], case["reference_answer"]
            retrieved_context = retrieve_context(question) if use_rag else ""
            messages = build_prompt(question, use_rag, retrieved_context)
            answer = query_model(model, messages)
            bleu = calc_bleu(answer, reference)
            rouge_l = calc_rouge_l(answer, reference)
            bert_f1 = calc_bert_score(answer, reference)
            judge = calc_llm_judge(question, reference, answer)
            consistency = calc_consistency(model, messages, CONSISTENCY_RUNS)
            print(f"  [{i:02d}/{len(test_cases)}] Q{q_id}（{category}）")
            print(f"    BLEU={bleu:.4f}  ROUGE-L={rouge_l:.4f}  BERTScore={bert_f1:.4f}  LLM综合={judge['综合均分']:.2f}  一致性={consistency:.4f}")
            exp_results.append({"q_id": q_id, "category": category, "question": question, "reference": reference, "answer": answer, "bleu": bleu, "rouge_l": rouge_l, "bert_score": bert_f1, "consistency": consistency, "judge_准确性": judge.get("准确性", 0), "judge_相关性": judge.get("相关性", 0), "judge_语态自然度": judge.get("语态自然度", 0), "judge_综合均分": judge.get("综合均分", 0)})
            all_results[exp_id] = exp_results
            with open(output_dir / f"exp_{exp_id}_detail.json", "w", encoding="utf-8") as f:
                json.dump(exp_results, f, ensure_ascii=False, indent=2)
            with open(output_dir / "all_results.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n所有实验完成，数据已完整保存至 {output_dir}。")

if __name__ == "__main__":
    run_evaluation()
