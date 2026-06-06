"""
report.py — 自动生成评估报告与柱状图

读取最新的 results/run_X/all_results.json，输出：
  1. 控制台汇总表格
  2. run_X/summary_table.csv     （表格数据）
  3. run_X/ablation_chart.png    （可直接贴论文的柱状图）
"""

import json
import csv
import statistics
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# 实验标签与指标配置
EXP_LABELS = {
    "A": "A: 基座",
    "B": "B: 微调",
    "C": "C: 基座+RAG",
    "D": "D: 微调+RAG",
}

# (JSON中的键名, 图表显示的图注名称)
METRICS = [
    ("bleu", "BLEU(字面匹配)"),
    ("rouge_l", "ROUGE-L(覆盖率)"),
    ("bert_score", "BERTScore(语义)"),
    ("consistency", "一致性评分(稳定性)"),
    ("judge_综合均分", "LLM裁判(综合表现)"),
]


def avg(values):
    valid = [v for v in values if v is not None and v >= 0]
    return round(statistics.mean(valid), 4) if valid else 0.0


def get_latest_run_dir() -> Path:
    """自动获取序号最大的 run_ 文件夹"""
    base_dir = Path("results")
    if not base_dir.exists():
        print("❌ 找不到 results 文件夹，请先运行 evaluate.py")
        exit(1)

    runs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not runs:
        print("❌ 找不到任何 run_x 文件夹，请先运行 evaluate.py")
        exit(1)

    latest_run = max(runs, key=lambda d: int(d.name.split("_")[1]))
    return latest_run


def compute_summary(all_results: dict) -> dict:
    """计算每组实验各指标的均值"""
    summary = {}
    for exp_id, results in all_results.items():
        summary[exp_id] = {}
        for key, _ in METRICS:
            values = [r.get(key, -1) for r in results]
            val = avg(values)
            # 如果之前的旧数据 LLM 评分还是 5 分制的，这里做个兼容保护（自动除以5）
            if "judge" in key and val > 1.0:
                val = round(val / 5.0, 4)
            summary[exp_id][key] = val
    return summary


def generate_bar_chart(summary: dict, output_dir: Path):
    """生成学术风的分组柱状图"""
    # 设置中文字体 (解决图表中文显示方块的问题)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 提取存在的实验组（适应中途停止时只有部分组的情况）
    exp_ids = sorted(summary.keys())
    group_labels = [EXP_LABELS.get(eid, eid) for eid in exp_ids]

    # 准备绘图数据
    x = np.arange(len(group_labels))  # 横坐标组别的位置
    width = 0.15  # 每个柱子的宽度
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

    # 针对每一个指标，画一组相对偏置的柱子
    for key, legend_label in METRICS:
        # 获取该指标在各组的分数
        scores = [summary[eid].get(key, 0.0) for eid in exp_ids]
        scores = [round(s, 3) for s in scores]

        offset = width * multiplier
        rects = ax.bar(x + offset, scores, width, label=legend_label)
        # 在柱子顶端显示数值
        ax.bar_label(rects, padding=3, fontsize=9)
        multiplier += 1

    # 美化图表
    ax.set_ylabel('评估得分 (0~1分)', fontsize=12)
    ax.set_title('校园问答机器人系统消融实验指标对比图', fontsize=14, pad=15)

    # 将 X 轴刻度设置在每一组柱子的正中间
    ax.set_xticks(x + width * (len(METRICS) - 1) / 2)
    ax.set_xticklabels(group_labels, fontsize=11)

    # 图例设置在图表外部上方，避免遮挡数据
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10)

    # Y 轴范围强制设为 0 到 1.1（留出数值标签的空间）
    ax.set_ylim(0, 1.1)

    # 添加隐隐的横向网格线辅助阅读
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    # 保存图片
    chart_path = output_dir / "ablation_chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存至: {chart_path}")


def main():
    latest_dir = get_latest_run_dir()
    print(f"📂 正在分析最新数据目录: {latest_dir}")

    path = latest_dir / "all_results.json"
    if not path.exists():
        print(f"❌ 目录下找不到 all_results.json")
        exit(1)

    with open(path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    summary = compute_summary(all_results)

    # 生成柱状图
    generate_bar_chart(summary, latest_dir)


if __name__ == "__main__":
    main()