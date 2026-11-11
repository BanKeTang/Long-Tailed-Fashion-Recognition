import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd5"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.dpi": 600, "figure.dpi": 150, "legend.frameon": False,
})
R = lambda f: json.load(open("results/" + f))
cap, seeds, swin, large, inter, df1, re = (R(x) for x in ["capacity_results.json", "capacity_seeds.json",
    "capacity_swin.json", "capacity_large.json", "interaction_results.json", "df1_results.json", "reeval_results.json"])
cifar, cifcap = R("cifar_lt_results.json"), R("cifar_lt_capacity_results.json")
S = [42, 123, 2024]

def get(size, aug, s):
    k = f"{size}_{aug}"
    if size == "large" and aug == "rare": return re[f"checkpoints/repeat_seed{s}_best.pth"]
    if size.startswith("swin"): return swin[k if s == 42 else f"{k}_s{s}"]
    if size == "large": return cap[k] if s == 42 else large[f"{k}_s{s}"]
    return cap[k] if s == 42 else seeds[f"{k}_s{s}"]

def style(ax, ygrid=True):
    if ygrid: ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6)

# ---------------- Figure 1: class distributions ----------------
df2 = {"short sleeve top": 71644, "trousers": 55386, "shorts": 36616, "long sleeve top": 36064, "skirt": 30835,
       "vest dress": 17949, "short sleeve dress": 17211, "vest": 16095, "long sleeve outwear": 13457,
       "long sleeve dress": 7907, "sling dress": 6492, "sling": 1985, "short sleeve outwear": 543}
df2_rare = {"long sleeve dress", "sling dress", "sling", "short sleeve outwear"}
df1c = [52138, 26653, 17752, 14183, 11204, 10785, 9606, 9517, 7548, 7270, 5425, 5408, 5126, 4464, 3570, 3259, 2910,
        2224, 1637, 1539, 1177, 781, 579, 534, 521, 491, 443, 374, 359, 287, 243, 229, 224, 121, 107, 99, 98, 63, 57,
        47, 38, 35, 32, 18, 13, 11]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.6]})
ax = axs[0]
names, vals = list(df2), list(df2.values())
cols = [ORANGE if n in df2_rare else BLUE for n in names]
ax.bar(range(len(vals)), vals, color=cols, width=0.72)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
ax.set_ylabel("Training instances"); ax.set_title("(a) DeepFashion2 (13 classes, 132:1)", fontsize=9, loc="left")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{int(v/1000)}k" if v else "0"))
style(ax)
ax = axs[1]
cols = [ORANGE if i >= len(df1c) - 14 else BLUE for i in range(len(df1c))]
ax.bar(range(len(df1c)), df1c, color=cols, width=0.72); ax.set_yscale("log")
ax.set_xticks([]); ax.set_xlabel("46 categories, sorted by frequency")
ax.set_title("(b) DeepFashion1 (46 classes, 4,740:1)", fontsize=9, loc="left"); style(ax)
from matplotlib.patches import Patch
axs[1].legend(handles=[Patch(color=BLUE, label="Common classes"), Patch(color=ORANGE, label="Rare classes")],
              loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig("fig1_distributions.png"); plt.close(fig)

# ---------------- Figure 2: study design ----------------
fig, ax = plt.subplots(figsize=(7.2, 3.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 52)
def box(x, y, w, h, title, lines, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=8.2, weight="bold", color=INK)
    for i, l in enumerate(lines):
        ax.text(x + 1.0, y + h - 7.0 - i * 3.75, l, ha="left", va="top", fontsize=6.8, color=INK2)
box(0.8, 1, 23.6, 50, "Study 1", [
    "Augmentation × capacity", "Factor: rare-class policy", "  off vs on", "Fixed: sampler (3D-W),", "  loss (CE-LS), Mixup",
    "DeepFashion2:", "  8 backbones, 11–197M", "DeepFashion1:", "  ConvNeXt-T/S/B",
    "CIFAR-100-LT:", "  ResNet-32 ×1–×8", "3 seeds, paired"], BLUE)
box(25.9, 1, 23.6, 50, "Study 2", [
    "Rebalancing × augmentation",
    "2 × 2 × 2 factorial:", "  loss: CE-LS vs CB-LS", "  sampler: uniform vs 3D-W",
    "  rare policy: off vs on", "DeepFashion2:", "  ConvNeXt-Tiny (all cells)", "  ConvNeXt-Large (policy on)",
    "CIFAR-100-LT:", "  CE/CB × uniform/RS", "3 seeds per cell"], ORANGE)
box(51.0, 1, 23.6, 50, "Study 3", [
    "Dose, components, mechanism",
    "Rare-policy magnitude:", "  0, 1/3, 2/3, 1", "Decomposition:", "  geometric only,", "  photometric only",
    "Diagnostics:", "  NCM on features,", "  logit adjustment,", "  classifier norms",
    "DeepFashion2, ConvNeXt-T", "3 seeds (diag.: 1)"], "#7a5bc7")
box(76.1, 1, 23.1, 50, "Study 4", [
    "Focal loss × Mixup",
    "2 × 2 factorial:", "  loss: CE vs focal", "  Mixup: off vs on",
    "CIFAR-100-LT:", "  ResNet-32, 3 seeds", "DeepFashion2:", "  EfficientNet-B3,", "  1 seed (supplementary)"], AQUA)
fig.tight_layout(); fig.savefig("fig2_design.png"); plt.close(fig)

# ---------------- Figure 3: Δ Rare F1 vs capacity (DF2, DF1) ----------------
fam = {"effb3": ("EfficientNet", 11.5, "B3"), "effb5": ("EfficientNet", 29.4, "B5"),
       "tiny": ("ConvNeXt", 28.2, "T"), "small": ("ConvNeXt", 49.9, "S"), "base": ("ConvNeXt", 88.1, "B"),
       "large": ("ConvNeXt", 197.0, "L"), "swint": ("Swin", 28.3, "Swin-T"), "swinb": ("Swin", 87.8, "Swin-B")}
FS = {"ConvNeXt": (BLUE, "o"), "EfficientNet": (ORANGE, "s"), "Swin": (AQUA, "^")}
jit = {"ConvNeXt": 1.0, "EfficientNet": 0.94, "Swin": 1.06}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
ax = axs[0]
for k, (f, P, lab) in fam.items():
    d = np.array([get(k, "rare", s)["rare_f1"] - get(k, "norare", s)["rare_f1"] for s in S]) * 100
    c, m = FS[f]; x = P * jit[f]
    ax.scatter([x] * 3, d, s=14, color=c, alpha=0.35, marker=m, lw=0)
    ax.errorbar(x, d.mean(), yerr=d.std(ddof=1), fmt=m, color=c, ms=7, mec="white", mew=0.8, capsize=2.5, lw=1.2)
    off = {"Swin": (7, 4), "EfficientNet": (-8, 6) if k == "effb5" else (7, -3)}.get(f, (7, -3))
    ax.annotate(lab, (x, d.mean()), textcoords="offset points", xytext=off, fontsize=7, color=INK2,
                ha="right" if off[0] < 0 else "left")
ax.axhline(0, color=INK2, lw=0.8); ax.set_xscale("log")
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([10, 20, 50, 100, 200]); ax.set_xticklabels(["10M", "20M", "50M", "100M", "200M"])
ax.set_xlabel("Parameters (log scale)"); ax.set_ylabel("Δ Rare-class F1 (pp)\naugmentation on − off")
ax.set_title("(a) DeepFashion2 (validation)", fontsize=9, loc="left"); style(ax)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=FS[f][0], marker=FS[f][1], ls="", ms=6, label=f) for f in FS],
          loc="lower right", fontsize=7.5)
ax = axs[1]
for k, P, lab in [("tiny", 28.2, "T"), ("small", 49.9, "S"), ("base", 88.1, "B")]:
    d = np.array([df1[f"{k}_rare_s{s}"]["test_best"]["rare_f1"] - df1[f"{k}_norare_s{s}"]["test_best"]["rare_f1"] for s in S]) * 100
    ax.scatter([P] * 3, d, s=14, color=BLUE, alpha=0.35, lw=0)
    ax.errorbar(P, d.mean(), yerr=d.std(ddof=1), fmt="o", color=BLUE, ms=7, mec="white", mew=0.8, capsize=2.5, lw=1.2)
    ax.annotate(lab, (P, d.mean()), textcoords="offset points", xytext=(6, -3), fontsize=7, color=INK2)
ax.axhline(0, color=INK2, lw=0.8); ax.set_xscale("log")
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([20, 50, 100]); ax.set_xticklabels(["20M", "50M", "100M"]); ax.set_xlim(18, 140)
ax.set_xlabel("Parameters (log scale)"); ax.set_title("(b) DeepFashion1 (test, ConvNeXt)", fontsize=9, loc="left")
style(ax); axs[0].set_ylim(-19, 2)
fig.tight_layout(); fig.savefig("fig3_rare_aug.png"); plt.close(fig)

# ---------------- Figure 4: CIFAR width ----------------
fig, ax = plt.subplots(figsize=(3.6, 2.8))
P = {1: 0.47, 2: 1.87, 4: 7.45, 8: 29.73}
for met, c, m, lab, off in [("few", ORANGE, "s", "Few-shot classes", 0.96), ("acc", BLUE, "o", "All classes", 1.04)]:
    mu, sd = [], []
    for k in P:
        d = np.array([cifcap[f"k{k}_aug_s{s}"][met] - cifcap[f"k{k}_noaug_s{s}"][met] for s in S]) * 100
        mu.append(d.mean()); sd.append(d.std(ddof=1))
    xs = [P[k] * off for k in P]
    ax.errorbar(xs, mu, yerr=sd, color=c, marker=m, ms=6, mec="white", mew=0.8, capsize=2.5, lw=1.6, label=lab)
ax.axhline(0, color=INK2, lw=0.8); ax.set_xscale("log")
ax.set_xticks(list(P.values())); ax.set_xticklabels(["0.47M\n(x1)", "1.9M\n(x2)", "7.5M\n(x4)", "30M\n(x8)"])
ax.set_xlabel("ResNet-32 width (parameters)"); ax.set_ylabel("Δ Accuracy (pp)\naugmentation on − off")
ax.legend(fontsize=7.5, loc="lower right"); style(ax)
fig.tight_layout(); fig.savefig("fig4_cifar_width.png"); plt.close(fig)

# ---------------- Figure 5: Study 2 factorial on DeepFashion2 ----------------
p3 = R("phase3_results.json")
def tc(name, s):
    if name in ("tiny_norare", "tiny_rare"): return get("tiny", name.split("_")[1], s)
    return p3[name if s == 42 else f"{name}_s{s}"]
def iplot(ax, xl, series, ylabel, title, ylim=None, dy={}):
    for lab, vals, c, m, ls in series:
        v = np.array(vals) * 100; mu = v.mean(axis=1); sd = v.std(axis=1, ddof=1)
        ax.errorbar([0, 1], mu, yerr=sd, color=c, marker=m, ms=6, mec="white", mew=0.8, capsize=2.5, lw=1.6, ls=ls, label=lab)
        ax.annotate(lab, (1, mu[1]), textcoords="offset points", xytext=(6, -3 + dy.get(lab, 0)), fontsize=7.5, color=c)
    ax.set_xticks([0, 1]); ax.set_xticklabels(xl); ax.set_xlim(-0.3, 1.8)
    if ylabel: ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, loc="left"); style(ax)
    if ylim: ax.set_ylim(*ylim)
g = lambda L: [x["rare_f1"] for x in L]
T = lambda n: g([tc(n, s) for s in S])
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
iplot(axs[0], ["Uniform", "3D-W"], [("CE-LS", [T("tiny_norare_ce_uni"), T("tiny_norare")], BLUE, "o", "-"),
      ("CB-LS", [T("tiny_norare_cb_uni"), T("tiny_norare_cb")], ORANGE, "s", "-")], "Rare-class F1 (%)",
      "(a) Tiny, rare policy off", (40, 66))
iplot(axs[1], ["Uniform", "3D-W"], [("CE-LS", [T("tiny_rare_ce_uni"), T("tiny_rare")], BLUE, "o", "--"),
      ("CB-LS", [T("tiny_rare_cb_uni"), T("tiny_rare_cb")], ORANGE, "s", "--")], None, "(b) Tiny, rare policy on", dy={"CE-LS": 8, "CB-LS": -8})
ce_u = [inter["large_ce_uniform"]] + [inter[f"large_ce_uniform_s{s}"] for s in [123, 2024]]
cb_u = [inter["large_cbls_uniform"]] + [inter[f"large_cbls_uniform_s{s}"] for s in [123, 2024]]
ce_w = [re[f"checkpoints/repeat_seed{s}_best.pth"] for s in S]
cb_w = [re[f"checkpoints/combined_seed{s}_best.pth"] for s in S]
p4 = R("phase4_results.json")
P4 = lambda n: [p4[n if s_ == 42 else f"{n}_s{s_}"]["rare_f1"] for s_ in S]
L_off_w = [get("large", "norare", s_)["rare_f1"] for s_ in S]
iplot(axs[2], ["Uniform", "3D-W"], [("CE-LS on", [g(ce_u), g(ce_w)], BLUE, "o", "--"), ("CB-LS on", [g(cb_u), g(cb_w)], ORANGE, "s", "--"),
      ("CE-LS off", [P4("large_norare_ce_uni"), L_off_w], BLUE, "o", "-")], None, "(c) Large, policy off / on",
      dy={"CE-LS on": 2, "CB-LS on": -9, "CE-LS off": 0})
v = np.array(P4("large_norare_cb_uni")) * 100
axs[2].errorbar([0], [v.mean()], yerr=[v.std(ddof=1)], color=ORANGE, marker="s", ms=6, mec="white", mew=0.8, capsize=2.5)
axs[2].annotate("CB-LS off", (0, v.mean()), textcoords="offset points", xytext=(-4, 7), fontsize=7.5, color=ORANGE, ha="center")
axs[0].set_ylim(40, 72)
fig.tight_layout(); fig.savefig("fig5_factorial.png"); plt.close(fig)

# ---------------- Figure 6: dose-response and diagnostics ----------------
diag = R("diag_results.json")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
mags = [0, 1/3, 2/3, 1]; names = ["tiny_norare", "tiny_mild", "tiny_medium", "tiny_rare"]
V = np.array([[tc(n, s)["rare_f1"] * 100 for n in names] for s in S])
for row in V: ax.plot(mags, row, color=BLUE, alpha=0.3, lw=0.9, marker="o", ms=3)
ax.errorbar(mags, V.mean(0), yerr=V.std(0, ddof=1), color=BLUE, marker="o", ms=6, mec="white", mew=0.8, capsize=2.5, lw=1.8, label="Joint magnitude")
for n, x, c, m, lab in [("tiny_geom", 1.18, ORANGE, "s", "Geometric only"), ("tiny_photo", 1.30, AQUA, "^", "Photometric only")]:
    v = np.array([tc(n, s)["rare_f1"] * 100 for s in S])
    ax.scatter([x] * 3, v, color=c, alpha=0.35, s=12, marker=m, lw=0)
    ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), color=c, marker=m, ms=6, mec="white", mew=0.8, capsize=2.5, lw=1.2, label=lab)
ax.set_xticks([0, 1/3, 2/3, 1, 1.18, 1.30]); ax.set_xticklabels(["0\n(off)", "1/3", "2/3", "1\n(full)", "G", "P"])
ax.set_xlim(-0.08, 1.4); ax.set_xlabel("Rare-policy magnitude"); ax.set_ylabel("Rare-class F1 (%)")
ax.set_title("(a) Dose–response (ConvNeXt-Tiny, 3D-W)", fontsize=9, loc="left"); ax.legend(fontsize=7, loc="lower left"); style(ax)
ax = axs[1]
d4 = R("phase4_diag.json")
keys = [("Trained", lambda r: r["argmax"]["rare_f1"]), ("Post-hoc LA", lambda r: r["logit_adjust"]["report_half_tau_best"]["rare_f1"]),
        ("cRT", lambda r: r["crt"]["rare_f1"]), ("NCM", lambda r: r["ncm"]["rare_f1"])]
for j, (run, c, lab) in enumerate([("tiny_norare_ce_uni", BLUE, "Policy off"), ("tiny_rare_ce_uni", ORANGE, "Policy on")]):
    mu = [np.mean([f(d4[f"{run}_s{s_}"]) * 100 for s_ in S]) for _, f in keys]
    sd_ = [np.std([f(d4[f"{run}_s{s_}"]) * 100 for s_ in S], ddof=1) for _, f in keys]
    xs = np.arange(4) + (j - 0.5) * 0.36
    ax.bar(xs, mu, width=0.33, color=c, yerr=sd_, capsize=2, error_kw={"lw": 0.8}, label=lab)
    for x_, m_ in zip(xs, mu): ax.text(x_, m_ + 1.0, f"{m_:.1f}", ha="center", fontsize=6.3, color=INK2)
ax.set_xticks(range(4)); ax.set_xticklabels([k for k, _ in keys], fontsize=7.5); ax.set_ylim(30, 78); ax.set_yticks([30, 40, 50, 60, 70])
ax.set_ylabel("Rare-class F1 (%)"); ax.set_title("(b) Diagnostics, CE-LS + uniform, 3 seeds", fontsize=9, loc="left")
ax.legend(fontsize=7, loc="upper center", ncol=2); style(ax)
fig.tight_layout(); fig.savefig("fig7_dose_mechanism.png"); plt.close(fig)

# ---------------- Figure 7: CIFAR-100-LT interactions ----------------
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.7))
c = lambda e, k: [cifar[f"{e}_s{s}"][k] for s in S]
iplot(axs[0], ["Uniform", "Re-sampling"], [("CE", [c("CE", "acc"), c("CE+RS", "acc")], BLUE, "o", "-"),
      ("CB", [c("CB", "acc"), c("CB+RS", "acc")], ORANGE, "s", "-")], "Overall accuracy (%)", "(a) CB × re-sampling")
iplot(axs[1], ["No Mixup", "Mixup"], [("CE", [c("CE", "acc"), c("CE+Mixup", "acc")], BLUE, "o", "-"),
      ("Focal", [c("Focal", "acc"), c("Focal+Mixup", "acc")], AQUA, "^", "-")], "Overall accuracy (%)", "(b) Focal × Mixup")
fig.tight_layout(); fig.savefig("fig6_cifar_interactions.png"); plt.close(fig)
print("ok")
