# 垂直领域问答系统框架

> **本项目仅提供系统框架**，不包含隐私数据、微调后的模型文件及完整模型权重。
> 使用者需自行准备领域数据、完成模型微调与量化部署。
> 本地配置参考：RTX3050Ti(4GB)+16GB内存

---

## 完整流程概览

```
① 数据集构建（隐私，不上传）
    ↓
② AutoDL LoRA 微调 + 量化（模型过大，不上传）
    ↓
③ Ollama 部署量化模型 & 基座模型
    ↓
④ 运行系统代码（本框架提供）
```

---

## ① 数据集构建

> 因涉及领域隐私数据，**数据集不随本项目上传**。

构建流程参考：
- 收集领域相关的规章制度、FAQ、手册等原始文档
- 整理为问答对格式（JSON），字段包含 `question`、`reference_answer`、`category`
- 参考 `test_cases.json` 的格式结构自行构建
- 数据规模根据领域复杂度自行决定

---

## ② LoRA 微调 + 量化（AutoDL）

> 微调后的模型文件较大，**不随本项目上传**。上传至 GitHub 的仅含框架代码。

推荐流程：
1. 在 AutoDL 等云平台申请 GPU 实例
2. 基于 Qwen2.5 等基座模型，使用 LoRA 进行领域微调
3. 微调完成后导出为 Ollama 支持的格式（GGUF）
4. 使用 `llama.cpp` 或 `AutoDL` 内置工具进行量化（Q4_K_M 等）

**关键产出物**：
- 微调后导出的 GGUF 模型文件 → 部署到 Ollama（如 `college_bot`）
- 基座模型同样通过 Ollama 部署（如 `qcwind/qwen2.5-7B-instruct-Q4_K_M`）

---

## ③ Ollama 部署

确保本地已安装 [Ollama](https://ollama.com/)，然后部署两个模型：

```bash
# 导入量化后的微调模型
ollama create college_bot -f ./college_bot/Modelfile

# 拉取基座模型（用于对比测试）
ollama pull qcwind/qwen2.5-7B-instruct-Q4_K_M
```

部署完成后验证：

```bash
ollama list
```

应能看到两个模型均在列表中。

---

## ④ 运行系统代码

所有命令在 `end_up/` 目录下执行：

```bash
cd end_up
```

### 4.1 配置

```bash
# 复制配置模板
cp config.yaml.example config.yaml
# 编辑 config.yaml，按需修改领域名称、助理名称、知识库路径等

# 复制环境变量模板（仅评估需要 API Key）
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY（如需使用 LLM 裁判评分）
```

配置文件说明（`config.yaml`）：

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `domain.name` | 领域名称 | `"校园"` |
| `model.finetuned` | 微调模型名（Ollama 中的名称） | `"college_bot"` |
| `model.base` | 基座模型名 | `"qcwind/qwen2.5-7B-instruct-Q4_K_M"` |
| `knowledge_base.raw_file` | 领域知识文档路径 | `"./knowledge_base/manual.txt"` |
| `knowledge_base.vector_db_dir` | 向量库存储目录 | `"./chroma_data"` |
| `embedding.model_name` | 嵌入模型路径 | `"./models/bge-small-zh-v1.5"` |
| `evaluation.test_cases` | 测试用例路径 | `"./test_cases.json"` |
| `ui.port` | Web 界面端口 | `8080` |

### 4.2 下载嵌入模型

```bash
python download_model_2.py
```

此脚本会从 HuggingFace 镜像下载 `bert-base-chinese` 等嵌入模型到 `./models/` 目录。

### 4.3 构建向量知识库

```bash
python build_db.py
```

此脚本读取 `config.yaml` 中 `knowledge_base.raw_file` 指向的领域文档，分块后构建 ChromaDB 向量库，保存至 `knowledge_base.vector_db_dir`。

### 4.4 启动对话系统

```bash
python main.py
```

访问 `http://localhost:8080` 打开 Web 界面。

左侧面板支持：
- 新建/切换/删除对话
- 切换模型（微调模型 vs 基座模型）
- 启用/关闭 RAG 检索
- 调节生成参数（温度、Top-P、最大 Token 数）

### 4.5 运行评估

```bash
python evaluate.py
```

评估流程自动执行 A/B/C/D 四组消融实验：

| 组别 | 模型 | RAG |
|------|------|-----|
| A | 基座模型 | 关闭 |
| B | 微调模型 | 关闭 |
| C | 基座模型 | 开启 |
| D | 微调模型 | 开启 |

评估指标包括：
- **BLEU**：字面精确度
- **ROUGE-L**：内容覆盖率
- **BERTScore**：语义相似度
- **LLM 裁判评分**：综合质量（需配置 `DEEPSEEK_API_KEY`）
- **一致性评分**：模型稳定性

结果保存在 `results/run_N/` 目录下。

> `report.py` 会自动读取最新的 `results/run_N/` 目录生成汇总图表（`ablation_chart.png`），无需手动运行。

### 4.6 GPU 监控（可选）

```bash
python monitor.py
```

实时监测 GPU 显存占用与利用率，配合对话测试后按 `Ctrl+C` 停止，自动生成 `evaluation_results.png`。

---

## 项目结构

```
.
├── end_up/                      # 系统代码（工作目录）
│   ├── core/
│   │   ├── config.py            # 配置读取
│   │   ├── retriever.py         # RAG 检索
│   │   ├── prompt.py            # Prompt 构建
│   │   └── session.py           # 会话管理
│   ├── main.py                  # Web 对话界面
│   ├── evaluate.py              # 消融实验评估
│   ├── report.py                # 评估报告生成（自动）
│   ├── monitor.py               # GPU 性能监控
│   ├── build_db.py              # 向量库构建
│   ├── download_model_2.py      # 嵌入模型下载
│   ├── config.yaml              # 领域配置文件
│   ├── config.yaml.example      # 配置模板
│   ├── .env                     # 敏感信息（不提交）
│   └── .env.example             # 环境变量模板
├── .gitignore
└── README.md
```

---

## 环境依赖

```bash
pip install nicegui ollama langchain-huggingface langchain-chroma \
            pyyaml python-dotenv rouge-score bert-score nltk \
            jieba matplotlib pynvml openai
```

---

## 换领域指引

1. 准备领域知识文档，放入 `knowledge_base/`
2. 编辑 `config.yaml`，修改 `domain.name`、`domain.assistant_name`、`knowledge_base.raw_file`
3. 构建领域测试用例 `test_cases.json`
4. 在 AutoDL 等平台用领域数据进行 LoRA 微调
5. 量化后部署到 Ollama
6. 重新运行 `python build_db.py` 构建向量库
7. 启动 `python main.py`

---

## 许可

MIT
