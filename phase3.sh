#!/bin/bash
# phase3.sh —— 第三轮补充实验（回应模拟审稿意见），全部用 ConvNeXt-Tiny，3 条队列 = 3 个种子并行
#   bash phase3.sh
# 先需要把 capacity_sweep.py（新版）、rare_policies.py、count_supports.py 放进 garment_type 目录。
# 每条队列按优先级顺序执行，结果全部写入 phase3_results.json，可断点续跑（重新执行本脚本即可）。
#
# 优先级（每个种子都按这个顺序）：
#   ① 不加稀少类增强的 CB × 采样 2×2（另一格 = 已有的 tiny_norare）            3 组
#   ② 加稀少类增强的 CB × 采样 2×2（另一格 = 已有的 tiny_rare）                3 组
#   ③ 增强拆分：只加强几何 / 只加强颜色                                         2 组
#   ④ 剂量：轻度 / 中度（强度 = 已有的 tiny_rare，零 = 已有的 tiny_norare）       2 组

cd "$(dirname "$0")"
python count_supports.py > log_supports.txt 2>&1

RUNS="tiny_norare_cb_uni tiny_norare_ce_uni tiny_norare_cb \
      tiny_rare_ce_uni tiny_rare_cb_uni tiny_rare_cb \
      tiny_geom tiny_photo \
      tiny_mild tiny_medium"

for s in 42 123 2024; do
  nohup python capacity_sweep.py --seed $s --out phase3_results.json --runs $RUNS \
    > log_p3_s$s.txt 2>&1 &
  echo "种子 $s 队列已启动（日志 log_p3_s$s.txt）"
done
