# 通用垂直领域问答系统框架

基于大模型微调 + RAG 检索增强 + NiceGUI Web 界面的通用垂直领域问答系统。

## 快速开始

### 1. 环境准备

\`\`\`bash
pip install nicegui ollama langchain-huggingface langchain-chroma pyyaml python-dotenv rouge-score bert-score nltk jieba matplotlib pynvml openai
\`\`\`

### 2. 配置领域

编辑 \`config.yaml\`，主要改三个字段：

\`\`\`yaml
domain:
  name: "校园"                     # 改为你的领域名称
  assistant_name: "校园事务辅导员"  # 改为你的助手名称
knowledge_base:
  raw_file: "./knowledge_base/manual.txt"  # 指向你的领域知识文档
\`\`\`

### 3. 设置密钥（可选，仅评估时需要）

\`\`\`bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
\`\`\`

### 4. 构建知识库

\`\`\`bash
python build_db.py
\`\`\`

### 5. 启动对话系统

\`\`\`bash
python main.py
# 访问 http://localhost:8080
\`\`\`

### 6. 运行评估

\`\`\`bash
python evaluate.py
python report.py
\`\`\`

## 项目结构

\`\`\`
├── config.yaml              # 领域配置总闸
├── main.py                  # Web 对话界面
├── evaluate.py              # 评估流水线
├── monitor.py               # GPU 监控
├── report.py                # 评估报告
├── build_db.py              # 知识库构建
├── core/                    # 通用逻辑模块
│   ├── config.py            # 配置读取
│   ├── retriever.py         # RAG 检索
│   ├── prompt.py            # Prompt 构建
│   └── session.py           # 会话管理
├── knowledge_base/          # 领域知识文档
├── test_cases.json          # 测试用例
└── .env                     # 敏感信息（不提交）
\`\`\`

## 换领域指南

1. 准备领域知识文档，放到 \`knowledge_base/\`
2. 修改 \`config.yaml\` 中的 \`domain.name\`、\`domain.assistant_name\`、\`knowledge_base.raw_file\`
3. 修改 \`test_cases.json\` 为对应领域的测试用例
4. 重新运行 \`python build_db.py\`

## 许可证

MIT
