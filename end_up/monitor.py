# monitor.py
import pynvml
import time
import csv
import os
import matplotlib.pyplot as plt
from datetime import datetime

# 初始化 NVIDIA 管理库
pynvml.nvmlInit()
# 假设你只有一张显卡，索引为 0 
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

GPU_LOG_FILE = "metrics_gpu.csv"
TTFT_LOG_FILE = "metrics_ttft.log"


def start_monitoring():
    print("🚀 开始监测系统资源...")
    print("💡 提示：去 Web 界面问几个问题。问完后，在这里按 Ctrl+C 停止监测，系统会自动为你画图！")

    # 清空旧的历史数据
    if os.path.exists(GPU_LOG_FILE): os.remove(GPU_LOG_FILE)
    if os.path.exists(TTFT_LOG_FILE): os.remove(TTFT_LOG_FILE)

    with open(GPU_LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'vram_gb', 'utilization_percent'])

        try:
            while True:
                # 获取显存信息
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_used_gb = mem_info.used / (1024 ** 3)

                # 获取 GPU 算力利用率
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util_info.gpu

                # 记录时间戳和数据
                current_time = time.time()
                writer.writerow([current_time, vram_used_gb, gpu_util])
                f.flush()

                # 如果显卡利用率飙升，说明模型正在推理
                if gpu_util > 10:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] ⚡ 模型推理中... 显存: {vram_used_gb:.2f} GB, 利用率: {gpu_util}%")

                time.sleep(0.5)  # 每 0.5 秒采样一次

        except KeyboardInterrupt:
            print("\n🛑 监测已停止。正在为你生成学术图表...")
            generate_plots()


def generate_plots():
    timestamps_gpu = []
    vram_usage = []

    # 读取 GPU 数据
    try:
        with open(GPU_LOG_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps_gpu.append(float(row['timestamp']))
                vram_usage.append(float(row['vram_gb']))
    except FileNotFoundError:
        print("未找到 GPU 数据。")
        return

    # 读取 TTFT 数据
    timestamps_ttft = []
    ttft_values = []
    try:
        with open(TTFT_LOG_FILE, 'r') as f:
            for line in f:
                ts, val = line.strip().split(',')
                timestamps_ttft.append(float(ts))
                ttft_values.append(float(val))
    except FileNotFoundError:
        print("未找到 TTFT 数据。可能你没有在网页里提问？")

    if not timestamps_gpu:
        return

    # 规范化时间轴（从 0 秒开始算起）
    start_t = timestamps_gpu[0]
    time_axis_gpu = [t - start_t for t in timestamps_gpu]
    time_axis_ttft = [t - start_t for t in timestamps_ttft]

    # 设置中文字体（根据你的电脑系统修改，这里默认用黑体）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # === 图 1：显存占用趋势 ===
    ax1.plot(time_axis_gpu, vram_usage, color='#1f77b4', linewidth=2, label='实时显存占用')
    ax1.axhline(y=4.0, color='red', linestyle='--', label='硬件上限 (4.0 GB)')
    ax1.set_title('系统负载测试 - 显存占用动态', fontsize=14)
    ax1.set_xlabel('运行时间 (秒)', fontsize=12)
    ax1.set_ylabel('显存占用 (GB)', fontsize=12)
    ax1.set_ylim(0, 4.5)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend()

    # === 图 2：TTFT 散点图 ===
    if time_axis_ttft:
        # 用散点图表示每一次对话的首字延迟
        ax2.scatter(time_axis_ttft, ttft_values, color='#ff7f0e', s=80, edgecolors='black', label='首字响应时间 (TTFT)')
        # 画一条平均线
        avg_ttft = sum(ttft_values) / len(ttft_values)
        ax2.axhline(y=avg_ttft, color='green', linestyle='-.', label=f'平均延迟: {avg_ttft:.2f}s')

    ax2.set_title('系统负载测试 - 首字响应延迟', fontsize=14)
    ax2.set_xlabel('运行时间 (秒)', fontsize=12)
    ax2.set_ylabel('延迟时间 (秒)', fontsize=12)
    ax2.set_ylim(0, max(ttft_values + [2.5]) * 1.2 if ttft_values else 2.5)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plot_filename = 'evaluation_results.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"✅ 图表已保存为 {plot_filename}，可以直接插入论文中！")
    plt.show()


if __name__ == "__main__":
    start_monitoring()