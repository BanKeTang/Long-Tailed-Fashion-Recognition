# analyze_v3.py — all numbers for the revised manuscript (sample SD, ddof=1)
import json, numpy as np
from scipy import stats
R = lambda f: json.load(open("results/" + f))
cap, seeds, swin, large, inter, df1, re_ = (R("capacity_results.json"), R("capacity_seeds.json"), R("capacity_swin.json"),
    R("capacity_large.json"), R("interaction_results.json"), R("df1_results.json"), R("reeval_results.json"))
p3, diag = R("phase3_results.json"), R("diag_results.json")
cifar, cifw = R("cifar_lt_results.json"), R("cifar_lt_capacity_results.json")
S = [42, 123, 2024]
sd = lambda x: float(np.std(x, ddof=1))
OUT = {}

def get(size, aug, s):
    k = f"{size}_{aug}"
    if size == "large" and aug == "rare": return re_[f"checkpoints/repeat_seed{s}_best.pth"]
    if size.startswith("swin"): return swin[k if s == 42 else f"{k}_s{s}"]
    if size == "large": return cap[k] if s == 42 else large[f"{k}_s{s}"]
    return cap[k] if s == 42 else seeds[f"{k}_s{s}"]
def last5(r, m):
    h = r.get("history")
    return None if not h or m not in h[-1] else float(np.mean([e[m] for e in h[-5:]]))

# ---------- Study 1 DF2 ----------
print("=== T2 DF2 rare aug x backbone ===")
BB = [("effb3", "EfficientNet-B3", 11.5), ("tiny", "ConvNeXt-Tiny", 28.2), ("swint", "Swin-T", 28.3), ("effb5", "EfficientNet-B5", 29.4),
      ("small", "ConvNeXt-Small", 49.9), ("swinb", "Swin-B", 87.8), ("base", "ConvNeXt-Base", 88.1), ("large", "ConvNeXt-Large", 197.0)]
pairs, bbmeans = [], []
pc_rare = {c: [] for c in ["short_sleeve_outwear", "sling", "long_sleeve_dress", "sling_dress"]}
pc_common = []
for size, name, P in BB:
    n = [get(size, "norare", s) for s in S]; a = [get(size, "rare", s) for s in S]
    v = lambda L, m: np.array([x[m] * 100 for x in L])
    N, A = v(n, "rare_f1"), v(a, "rare_f1"); d = A - N
    dm, dall = v(a, "macro_f1") - v(n, "macro_f1"), v(a, "acc_all") - v(n, "acc_all")
    p = stats.ttest_rel(A, N).pvalue
    l5d = [ (last5(a[i], "rare_f1") - last5(n[i], "rare_f1")) * 100 for i in range(3) if last5(a[i], "rare_f1") and last5(n[i], "rare_f1")]
    print(f"{name:16s} {P:6.1f}M | off {N.mean():.1f}±{sd(N):.1f} on {A.mean():.1f}±{sd(A):.1f} | Δ {d.mean():+.1f}±{sd(d):.1f} p={p:.3f} | Δmacro {dm.mean():+.1f}±{sd(dm):.1f} | Δall {dall.mean():+.1f}±{sd(dall):.1f} | last5 {np.mean(l5d) if l5d else float('nan'):+.1f} (n={len(l5d)}) | per-seed {np.round(d,1)}")
    pairs += [(np.log10(P), x) for x in d]; bbmeans.append((np.log10(P), d.mean()))
    for i in range(3):
        for c in pc_rare: pc_rare[c].append((a[i]["per_class_f1"][c] - n[i]["per_class_f1"][c]) * 100)
        pc_common.append(np.mean([(a[i]["per_class_f1"][c] - n[i]["per_class_f1"][c]) * 100 for c in a[i]["per_class_f1"] if c not in pc_rare]))
x, y = np.array(pairs).T
print("pairs", len(y), "all negative:", (y < 0).all(), "mean", y.mean().round(2))
r = stats.pearsonr(x, y); print(f"Pearson (24 pairs) r={r[0]:.2f} p={r[1]:.2f}")
bx, by = np.array(bbmeans).T
lr = stats.linregress(bx, by); tcrit = stats.t.ppf(0.975, len(bx) - 2)
print(f"backbone-mean regression (n=8): slope {lr.slope:+.2f} pts/decade, 95% CI [{lr.slope - tcrit*lr.stderr:+.2f}, {lr.slope + tcrit*lr.stderr:+.2f}], r={lr.rvalue:.2f}, p={lr.pvalue:.2f}")
print("per-class Δ:", {c: round(np.mean(v), 1) for c, v in pc_rare.items()}, "neg counts", {c: int((np.array(v) < 0).sum()) for c, v in pc_rare.items()}, "common mean", round(np.mean(pc_common), 2))

# ---------- DF1 ----------
print("\n=== T3 DF1 ===")
for size in ["tiny", "small", "base"]:
    row = [size]
    for tag in ["test_best", "test_final"]:
        for m in ["acc", "macro_f1", "rare_f1", "common_f1"]:
            N = np.array([df1[f"{size}_norare_s{s}"][tag][m] for s in S]) * 100; A = np.array([df1[f"{size}_rare_s{s}"][tag][m] for s in S]) * 100; d = A - N
            row.append(f"{tag[5:]}:{m} off {N.mean():.1f} Δ{d.mean():+.1f}±{sd(d):.1f} p={stats.ttest_rel(A,N).pvalue:.3f}")
    print(" | ".join(row))
sup = R("supports.json")
df1_rare = sorted(sup["df1_train"], key=lambda c: sup["df1_train"][c])[:14]
print("DF1 rare classes:", [(c, sup["df1_train"][c], sup["df1_test"][c]) for c in df1_rare], "test total", sum(sup["df1_test"][c] for c in df1_rare))
print("DF2 val rare supports:", {c: sup["df2_val"][c] for c in pc_rare})

# ---------- CIFAR width ----------
print("\n=== T4 CIFAR width ===")
for k in [1, 2, 4, 8]:
    N = {m: np.array([cifw[f"k{k}_noaug_s{s}"][m] for s in S]) * 100 for m in ["acc", "many", "medium", "few"]}
    A = {m: np.array([cifw[f"k{k}_aug_s{s}"][m] for s in S]) * 100 for m in ["acc", "many", "medium", "few"]}
    print(f"x{k}: off acc {N['acc'].mean():.1f} few {N['few'].mean():.1f} | " + " ".join(f"Δ{m} {(A[m]-N[m]).mean():+.1f}±{sd(A[m]-N[m]):.1f}" for m in ["acc", "many", "medium", "few"]))

# ---------- Study 2: Tiny 2x2x2 ----------
print("\n=== T5 Tiny loss x sampler x rare aug ===")
def tcell(name, s):
    if name in ("tiny_norare", "tiny_rare"): return get("tiny", name.split("_")[1], s)
    return p3[name if s == 42 else f"{name}_s{s}"]
cells = {("CE-LS", "Uniform", "off"): "tiny_norare_ce_uni", ("CE-LS", "3D-W", "off"): "tiny_norare",
         ("CB-LS", "Uniform", "off"): "tiny_norare_cb_uni", ("CB-LS", "3D-W", "off"): "tiny_norare_cb",
         ("CE-LS", "Uniform", "on"): "tiny_rare_ce_uni", ("CE-LS", "3D-W", "on"): "tiny_rare",
         ("CB-LS", "Uniform", "on"): "tiny_rare_cb_uni", ("CB-LS", "3D-W", "on"): "tiny_rare_cb"}
V = {}
for c, nme in cells.items():
    V[c] = {m: np.array([tcell(nme, s)[m] * 100 for s in S]) for m in ["acc_all", "macro_f1", "rare_f1", "acc_shop", "acc_user"]}
    V[c]["rare_l5"] = np.array([last5(tcell(nme, s), "rare_f1") * 100 for s in S])
    print(c, " ".join(f"{m} {V[c][m].mean():.1f}±{sd(V[c][m]):.1f}" for m in ["acc_all", "macro_f1", "rare_f1", "rare_l5"]))
for m in ["rare_f1", "rare_l5", "macro_f1", "acc_all"]:
    for aug in ["off", "on"]:
        rs = V["CE-LS", "3D-W", aug][m] - V["CE-LS", "Uniform", aug][m]; rsb = V["CB-LS", "3D-W", aug][m] - V["CB-LS", "Uniform", aug][m]
        cb = V["CB-LS", "Uniform", aug][m] - V["CE-LS", "Uniform", aug][m]; I = rsb - rs
        print(f" {m} aug {aug}: RS|CE {rs.mean():+.1f}±{sd(rs):.1f}  RS|CB {rsb.mean():+.1f}±{sd(rsb):.1f}  CB|uni {cb.mean():+.1f}±{sd(cb):.1f}  I {I.mean():+.1f}±{sd(I):.1f} (seeds {np.round(I,1)})")
    for l in ["CE-LS", "CB-LS"]:
        for smp in ["Uniform", "3D-W"]:
            d = V[l, smp, "on"][m] - V[l, smp, "off"][m]
            print(f"   aug effect {l} {smp}: {d.mean():+.1f}±{sd(d):.1f} seeds {np.round(d,1)}")
    I3 = ((V["CB-LS","3D-W","on"][m]-V["CB-LS","Uniform","on"][m])-(V["CE-LS","3D-W","on"][m]-V["CE-LS","Uniform","on"][m])) - \
         ((V["CB-LS","3D-W","off"][m]-V["CB-LS","Uniform","off"][m])-(V["CE-LS","3D-W","off"][m]-V["CE-LS","Uniform","off"][m]))
    Iar = (V["CE-LS","3D-W","on"][m]-V["CE-LS","3D-W","off"][m]) - (V["CE-LS","Uniform","on"][m]-V["CE-LS","Uniform","off"][m])
    print(f"   I(aug x RS | CE) {Iar.mean():+.1f}±{sd(Iar):.1f}; three-way {I3.mean():+.1f}±{sd(I3):.1f}")

# Large 2x2 (with rare aug), sample SD
print("\n=== Large 2x2 (rare aug on) ===")
L = {"CE-LS Uniform": [inter["large_ce_uniform"]] + [inter[f"large_ce_uniform_s{s}"] for s in [123, 2024]],
     "CB-LS Uniform": [inter["large_cbls_uniform"]] + [inter[f"large_cbls_uniform_s{s}"] for s in [123, 2024]],
     "CE-LS 3D-W": [re_[f"checkpoints/repeat_seed{s}_best.pth"] for s in S],
     "CB-LS 3D-W": [re_[f"checkpoints/combined_seed{s}_best.pth"] for s in S]}
LM = {k: {m: np.array([x[m] * 100 for x in v]) for m in ["acc_all", "macro_f1", "rare_f1"]} for k, v in L.items()}
for k in LM: print(k, " ".join(f"{m} {LM[k][m].mean():.1f}±{sd(LM[k][m]):.1f}" for m in LM[k]))
for m in ["acc_all", "macro_f1", "rare_f1"]:
    I = (LM["CB-LS 3D-W"][m] - LM["CB-LS Uniform"][m]) - (LM["CE-LS 3D-W"][m] - LM["CE-LS Uniform"][m])
    print(f" I {m} {I.mean():+.1f}±{sd(I):.1f}")
print(" Large CE-LS 3D-W off (Study 1):", np.round([get('large','norare',s)['rare_f1']*100 for s in S],1))

# ---------- Study 3: dose & decomposition ----------
print("\n=== T6 dose ===")
base = V["CE-LS", "3D-W", "off"]
for nme, lab in [("tiny_norare", "off"), ("tiny_mild", "mild"), ("tiny_medium", "medium"), ("tiny_rare", "strong"), ("tiny_geom", "geom"), ("tiny_photo", "photo")]:
    X = {m: np.array([tcell(nme, s)[m] * 100 for s in S]) for m in ["rare_f1", "macro_f1", "acc_all"]}
    X["rare_l5"] = np.array([last5(tcell(nme, s), "rare_f1") * 100 for s in S])
    line = f"{lab:7s}"
    for m in ["rare_f1", "rare_l5", "macro_f1", "acc_all"]:
        b = base[m]; d = X[m] - b
        p = stats.ttest_rel(X[m], b).pvalue if lab != "off" else float("nan")
        line += f" | {m} {X[m].mean():.1f}±{sd(X[m]):.1f} Δ{d.mean():+.1f}±{sd(d):.1f} p={p:.3f}"
    print(line)
# linear trend in magnitude m (0,1/3,2/3,1) per seed
mags = [0, 1/3, 2/3, 1]; names = ["tiny_norare", "tiny_mild", "tiny_medium", "tiny_rare"]
sl = [stats.linregress(mags, [tcell(n, s)["rare_f1"] * 100 for n in names]).slope for s in S]
print("dose slope per seed (pts per unit magnitude):", np.round(sl, 1), "mean", round(np.mean(sl), 1), "±", round(sd(sl), 1), "one-sample p", round(stats.ttest_1samp(sl, 0).pvalue, 3))

# ---------- diagnostics ----------
print("\n=== T7 diagnostics (one seed per cell) ===")
order = ["tiny_norare_ce_uni", "tiny_rare_ce_uni", "tiny_norare", "tiny_rare", "tiny_norare_cb_uni", "tiny_rare_cb_uni", "tiny_norare_cb", "tiny_rare_cb", "tiny_mild", "tiny_medium", "tiny_geom", "tiny_photo"]
for nme in order:
    k = [k for k in diag if diag[k]["run"] == nme][0]; dd = diag[k]
    la = dd["logit_adjust"]
    print(f"{nme:20s} s{dd['seed']:<5d} argmax {dd['argmax']['rare_f1']*100:.1f} | pred/true {np.mean(list(dd['rare_pred_over_true'].values())):.2f} | recall {np.mean(list(dd['rare_recall'].values()))*100:.1f} prec {np.mean(list(dd['rare_precision'].values()))*100:.1f} | wnorm {dd['weight_norm_rare_over_common']:.2f} | LA tau {la['tau']}: {la['report_half_tau0']['rare_f1']*100:.1f}->{la['report_half_tau_best']['rare_f1']*100:.1f} | NCM {dd['ncm']['rare_f1']*100:.1f} (acc {dd['ncm']['acc_all']*100:.1f})")

# ---------- CIFAR conflicts ----------
print("\n=== T8 CIFAR conflicts ===")
C = {e: {m: np.array([cifar[f"{e}_s{s}"][m] for s in S]) * 100 for m in ["acc", "many", "medium", "few"]} for e in ["CE", "CE+RS", "CB", "CB+RS", "CE+Mixup", "Focal", "Focal+Mixup"]}
for e in C: print(e, " ".join(f"{m} {C[e][m].mean():.1f}±{sd(C[e][m]):.1f}" for m in C[e]))
for m in ["acc", "many", "medium", "few"]:
    I1 = (C["CB+RS"][m] - C["CB"][m]) - (C["CE+RS"][m] - C["CE"][m]); I2 = (C["Focal+Mixup"][m] - C["Focal"][m]) - (C["CE+Mixup"][m] - C["CE"][m])
    print(f" {m}: I CBxRS {I1.mean():+.1f}±{sd(I1):.1f} | I focal x mixup {I2.mean():+.1f}±{sd(I2):.1f}")

# DF2 focal (one seed)
print("\n=== DF2 focal x mixup (B3, 1 seed) ===")
for k, x in [("CE nomix", inter["b3_ce_nomix"]), ("NFL nomix", inter["b3_nfl_nomix"]), ("CE mix", re_["checkpoints/ablation_exp4_best.pth"]), ("NFL mix", re_["checkpoints/ablation_exp5_best.pth"])]:
    print(k, round(x["acc_all"]*100, 1), round(x["rare_f1"]*100, 1))
