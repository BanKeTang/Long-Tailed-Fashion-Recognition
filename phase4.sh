#!/bin/bash
# phase4.sh —— 第四轮补充实验（回应第二次模拟审稿），3 条队列并行
#   bash phase4.sh
# 需要先放入 garment_type 目录：capacity_sweep.py（新版）、df1_sweep.py（新版）、phase4_diag.py
# 结果：DeepFashion2 → phase4_results.json；DeepFashion1 → df1_results.json（新键带 _uni）；诊断 → phase4_diag.json
# 中断后重新执行本脚本即可续跑。

cd "$(dirname "$0")"

# 队列 1：ConvNeXt-Large，CE-LS + 均匀采样 + 不加稀少类增强（3 种子），然后 DeepFashion1 均匀采样（有/无增强）
nohup bash -c '
for s in 42 123 2024; do
  python capacity_sweep.py --seed $s --out phase4_results.json --runs large_norare_ce_uni
done
python df1_sweep.py --sizes tiny --sampler uniform --seeds 42 123 2024
' > log_p4_q1.txt 2>&1 &
echo "队列1 已启动（log_p4_q1.txt）"

# 队列 2：ConvNeXt-Large，CB-LS + 均匀采样 + 不加稀少类增强（3 种子），然后 Tiny 标准类别均衡采样
nohup bash -c '
for s in 42 123 2024; do
  python capacity_sweep.py --seed $s --out phase4_results.json --runs large_norare_cb_uni
done
for s in 42 123 2024; do
  python capacity_sweep.py --seed $s --out phase4_results.json --runs tiny_norare_ce_cbs
done
' > log_p4_q2.txt 2>&1 &
echo "队列2 已启动（log_p4_q2.txt）"

# 队列 3：Tiny 干净基线与强增强（每个种子单独保存 checkpoint，用于多种子诊断和 cRT），logit-adjusted 损失；最后跑诊断
nohup bash -c '
for s in 42 123 2024; do
  python capacity_sweep.py --seed $s --out phase4_results.json --runs tiny_norare_ce_uni tiny_rare_ce_uni
done
for s in 42 123 2024; do
  python capacity_sweep.py --seed $s --out phase4_results.json --runs tiny_norare_la_uni
done
python phase4_diag.py
' > log_p4_q3.txt 2>&1 &
echo "队列3 已启动（log_p4_q3.txt）"
