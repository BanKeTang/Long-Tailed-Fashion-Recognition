#!/bin/bash
# phase4_resume.sh —— 服务器重启后续跑第四轮（已完成的组自动跳过；中断的那组从头重跑）
#   bash phase4_resume.sh
# 日志写到新的 log_p4r_q*.txt，不覆盖重启前的旧日志。

cd "$(dirname "$0")"

# 队列 1：Large CE-LS + 均匀（种子 123、2024），然后 DeepFashion1 均匀采样（种子 42、123）
nohup bash -c '
for s in 123 2024; do
  python capacity_sweep.py --seed $s --out phase4_results.json --runs large_norare_ce_uni
done
python df1_sweep.py --sizes tiny --sampler uniform --seeds 42 123
' > log_p4r_q1.txt 2>&1 &
echo "队列1 已启动（log_p4r_q1.txt）"

# 队列 2：Large CB-LS + 均匀（种子 123、2024），然后 DeepFashion1 均匀采样（种子 2024）
nohup bash -c '
for s in 123 2024; do
  python capacity_sweep.py --seed $s --out phase4_results.json --runs large_norare_cb_uni
done
python df1_sweep.py --sizes tiny --sampler uniform --seeds 2024
' > log_p4r_q2.txt 2>&1 &
echo "队列2 已启动（log_p4r_q2.txt）"

# 队列 3：Tiny 剩下的组（种子 2024 的干净/强增强一对、logit-adjusted 损失、类别均衡采样），最后跑诊断
nohup bash -c '
python capacity_sweep.py --seed 2024 --out phase4_results.json --runs tiny_norare_ce_uni tiny_rare_ce_uni
for s in 42 123 2024; do
  python capacity_sweep.py --seed $s --out phase4_results.json --runs tiny_norare_la_uni
done
for s in 42 123 2024; do
  python capacity_sweep.py --seed $s --out phase4_results.json --runs tiny_norare_ce_cbs
done
python phase4_diag.py
' > log_p4r_q3.txt 2>&1 &
echo "队列3 已启动（log_p4r_q3.txt）"
