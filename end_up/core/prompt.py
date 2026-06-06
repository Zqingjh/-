# prompt.py - Prompt 构建模块
from core.config import config

DEFAULT_TEMPLATE = (
    "你是一个智能且贴心的{assistant_name}。\n"
    "请结合你自身的{domain}知识储备，以及以下提供的【参考最新规章】，综合解答。\n\n"
    "【参考最新规章】：\n"
    "{context}\n\n"
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

def build_system_prompt(context_text: str = "") -> str:
    template = config.system_template if config.system_template else DEFAULT_TEMPLATE
    context = context_text if context_text else "（当前无相关的实时规章补充，请直接调用你的记忆回答）"
    return template.format(
        assistant_name=config.assistant_name,
        domain=config.domain_name,
        context=context
    )

def build_messages(user_text: str, context_text: str, history: list = None) -> list:
    sys_prompt = build_system_prompt(context_text)
    messages = [{"role": "system", "content": sys_prompt}]
    if history:
        messages.extend([{"role": m["role"], "content": m["content"]} for m in history[-3:]])
    messages.append({"role": "user", "content": user_text})
    return messages
