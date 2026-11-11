# count_supports.py —— 统计每个类别的样本数（审稿意见 M7：报告稀少类测试样本数）
# 用法（在 garment_type 目录下）：python count_supports.py   → 写入 supports.json，几秒钟
import json, os
from pathlib import Path

out = {}
for name, root in [("df2_train", "crops/train"), ("df2_val", "crops/validation"),
                   ("df1_train", "crops_df1/train"), ("df1_val", "crops_df1/val"), ("df1_test", "crops_df1/test")]:
    p = Path(root)
    if not p.exists():
        print(f"跳过 {root}（不存在）"); continue
    out[name] = {d.name: len(list(d.glob("*.jpg"))) for d in sorted(p.iterdir()) if d.is_dir()}
    print(f"{name}: {len(out[name])} 类, {sum(out[name].values())} 张")
json.dump(out, open("supports.json", "w"), indent=2, ensure_ascii=False)
print("已写入 supports.json")
