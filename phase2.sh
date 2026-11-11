#!/bin/bash
# phase2.sh —— 第二轮补充实验，一条命令启动三条并行队列（在 garment_type 目录下运行）
#   bash phase2.sh
# 每条队列内部依次执行；某一步出错不影响后面的步骤；所有结果都可断点续跑。
# 三条队列把 ConvNeXt-Large 全部放在同一条里，保证同一时间最多只有一个 Large，避免显存不足。

cd "$(dirname "$0")"

# ── 队列 1：ConvNeXt-T/S/B 和 EfficientNet-B3/B5 的配对，补种子 123、2024 ──────────
nohup bash -c '
for s in 123 2024; do
  python capacity_sweep.py --seed $s --out capacity_seeds.json \
    --runs tiny_norare tiny_rare small_norare small_rare base_norare base_rare \
           effb3_norare effb3_rare effb5_norare effb5_rare
done
' > log_q1_seeds.txt 2>&1 &
echo "队列1 已启动（日志 log_q1_seeds.txt）"

# ── 队列 2：Swin-T / Swin-B 配对（3 个种子），然后自动接 DeepFashion1 ─────────────
nohup bash -c '
for s in 42 123 2024; do
  python capacity_sweep.py --seed $s --out capacity_swin.json \
    --runs swint_norare swint_rare swinb_norare swinb_rare
done
echo "Swin 完成，等待 DeepFashion1 数据（crops_df1/meta.json）..."
while [ ! -f crops_df1/meta.json ]; do sleep 600; done
python df1_sweep.py --seeds 42 123 2024
' > log_q2_swin_df1.txt 2>&1 &
echo "队列2 已启动（日志 log_q2_swin_df1.txt）"

# ── 队列 3：所有 ConvNeXt-Large（CB×采样 补种子 + 不加 rare aug 补种子）─────────────
nohup bash -c '
for s in 123 2024; do
  python interaction_runs.py --seed $s --runs large_ce_uniform large_cbls_uniform
done
for s in 123 2024; do
  python capacity_sweep.py --seed $s --out capacity_large.json --runs large_norare
done
' > log_q3_large.txt 2>&1 &
echo "队列3 已启动（日志 log_q3_large.txt）"
