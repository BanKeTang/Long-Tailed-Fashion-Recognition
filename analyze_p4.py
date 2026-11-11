# analyze_p4.py — numbers for phase-4 additions (sample SD)
import json, numpy as np
from scipy import stats
exec(open('analyze_v3.py').read().split('# ---------- Study 1 DF2')[0])
p4, d4 = R("phase4_results.json"), R("phase4_diag.json")
def ci(x):
    x = np.array(x); h = stats.t.ppf(.975, len(x)-1) * sd(x) / np.sqrt(len(x))
    return f"{x.mean():+.1f} ± {sd(x):.1f} [{x.mean()-h:+.1f}, {x.mean()+h:+.1f}] p={stats.ttest_1samp(x,0).pvalue:.3f} seeds {np.round(x,1)}"
ms = lambda x: f"{np.mean(x):.1f} ± {sd(x):.1f}"
g4 = lambda n, s, m="rare_f1": p4[n if s == 42 else f"{n}_s{s}"][m] * 100
def tcell(name, s):
    if name in ("tiny_norare", "tiny_rare"): return get("tiny", name.split("_")[1], s)
    return p3[name if s == 42 else f"{name}_s{s}"]
print("== reproducibility Tiny (phase3 vs phase4)")
for n in ["tiny_norare_ce_uni", "tiny_rare_ce_uni"]:
    print(n, [round(g4(n, s) - tcell(n, s)["rare_f1"] * 100, 4) for s in S], [round(g4(n, s, "acc_all") - tcell(n, s)["acc_all"] * 100, 4) for s in S])
print("== Large cells")
L = {}
L["CE uni off"] = {m: np.array([g4("large_norare_ce_uni", s, m) for s in S]) for m in ["acc_all", "macro_f1", "rare_f1", "acc_user"]}
L["CB uni off"] = {m: np.array([g4("large_norare_cb_uni", s, m) for s in S]) for m in ["acc_all", "macro_f1", "rare_f1", "acc_user"]}
L["CE 3DW off"] = {m: np.array([get("large", "norare", s)[m] * 100 for s in S]) for m in ["acc_all", "macro_f1", "rare_f1", "acc_user"]}
on = {"CE uni on": [inter["large_ce_uniform"]] + [inter[f"large_ce_uniform_s{s}"] for s in [123, 2024]],
      "CB uni on": [inter["large_cbls_uniform"]] + [inter[f"large_cbls_uniform_s{s}"] for s in [123, 2024]],
      "CE 3DW on": [re_[f"checkpoints/repeat_seed{s}_best.pth"] for s in S],
      "CB 3DW on": [re_[f"checkpoints/combined_seed{s}_best.pth"] for s in S]}
for k, v in on.items(): L[k] = {m: np.array([x[m] * 100 for x in v]) for m in ["acc_all", "macro_f1", "rare_f1", "acc_user"]}
for k in L: print(f"{k:12s}", " ".join(f"{m} {ms(L[k][m])}" for m in L[k]))
for n in ["large_norare_ce_uni", "large_norare_cb_uni"]:
    print(n, "last5", ms([np.mean([h["rare_f1"] for h in p4[n if s == 42 else f'{n}_s{s}']["history"][-5:]]) * 100 for s in S]))
print("RS|CE off", ci(L["CE 3DW off"]["rare_f1"] - L["CE uni off"]["rare_f1"]))
print("CB|uni off", ci(L["CB uni off"]["rare_f1"] - L["CE uni off"]["rare_f1"]))
print("CB|uni off macro", ci(L["CB uni off"]["macro_f1"] - L["CE uni off"]["macro_f1"]))
print("aug CE uni", ci(L["CE uni on"]["rare_f1"] - L["CE uni off"]["rare_f1"]))
print("aug CB uni", ci(L["CB uni on"]["rare_f1"] - L["CB uni off"]["rare_f1"]))
print("aug CE 3DW", ci(L["CE 3DW on"]["rare_f1"] - L["CE 3DW off"]["rare_f1"]))
print("best vs stacked Large", {m: round(L["CB uni off"][m].mean() - L["CE 3DW on"][m].mean(), 1) for m in ["rare_f1", "macro_f1", "acc_all", "acc_user"]})
print("best Tiny vs stacked", {m: round(np.mean([tcell('tiny_norare_ce_uni', s)[m] for s in S])*100 - np.mean([tcell('tiny_rare', s)[m] for s in S])*100, 1) for m in ["rare_f1", "macro_f1", "acc_all", "acc_user"]})
print("== Tiny remedies (policy off)")
for n in ["tiny_norare_ce_uni", "tiny_norare_la_uni", "tiny_norare_ce_cbs"]:
    print(n, " ".join(f"{m} {ms([g4(n, s, m) for s in S])}" for m in ["rare_f1", "macro_f1", "acc_all", "acc_user"]),
          "last5", ms([np.mean([h["rare_f1"] for h in p4[n if s == 42 else f'{n}_s{s}']["history"][-5:]]) * 100 for s in S]))
for n, b in [("tiny_norare_la_uni", "tiny_norare_ce_uni"), ("tiny_norare_ce_cbs", "tiny_norare_ce_uni")]:
    for m in ["rare_f1", "macro_f1", "acc_all"]:
        print(f"  {n} - CE uni {m}:", ci([g4(n, s, m) - g4(b, s, m) for s in S]))
print("== diag (3 seeds)")
D = lambda n, f: np.array([f(d4[f"{n}_s{s}"]) * (100 if "x100" else 1) for s in S])
for n in ["tiny_norare_ce_uni", "tiny_rare_ce_uni", "tiny_norare_la_uni", "tiny_norare_ce_cbs"]:
    x = {k: np.array([fn(d4[f"{n}_s{s}"]) for s in S]) for k, fn in {
        "trained_rare": lambda r: r["argmax"]["rare_f1"] * 100, "trained_bal": lambda r: r["argmax"]["balanced_acc"] * 100,
        "trained_macro": lambda r: r["argmax"]["macro_f1"] * 100, "trained_acc": lambda r: r["argmax"]["acc_all"] * 100,
        "pred_true": lambda r: r["rare_pred_over_true"], "wnorm": lambda r: r["weight_norm_rare_over_common"],
        "la_tau": lambda r: r["logit_adjust"]["tau"], "la0_rare": lambda r: r["logit_adjust"]["report_half_tau0"]["rare_f1"] * 100,
        "la_rare": lambda r: r["logit_adjust"]["report_half_tau_best"]["rare_f1"] * 100,
        "la0_macro": lambda r: r["logit_adjust"]["report_half_tau0"]["macro_f1"] * 100, "la_macro": lambda r: r["logit_adjust"]["report_half_tau_best"]["macro_f1"] * 100,
        "la0_acc": lambda r: r["logit_adjust"]["report_half_tau0"]["acc_all"] * 100, "la_acc": lambda r: r["logit_adjust"]["report_half_tau_best"]["acc_all"] * 100,
        "la0_bal": lambda r: r["logit_adjust"]["report_half_tau0"]["balanced_acc"] * 100, "la_bal": lambda r: r["logit_adjust"]["report_half_tau_best"]["balanced_acc"] * 100,
        "ncm_rare": lambda r: r["ncm"]["rare_f1"] * 100, "ncm_acc": lambda r: r["ncm"]["acc_all"] * 100,
        "crt_rare": lambda r: r["crt"]["rare_f1"] * 100, "crt_macro": lambda r: r["crt"]["macro_f1"] * 100,
        "crt_acc": lambda r: r["crt"]["acc_all"] * 100, "crt_bal": lambda r: r["crt"]["balanced_acc"] * 100}.items()}
    globals()["X_" + n] = x
    print(n); print("   " + " | ".join(f"{k} {x[k].mean():.2f}±{sd(x[k]):.2f}" for k in x))
A, B = X_tiny_norare_ce_uni, X_tiny_rare_ce_uni
for k in ["trained_rare", "ncm_rare", "crt_rare", "la_rare", "pred_true", "wnorm", "crt_bal"]:
    print(f" on-off {k}:", ci(B[k] - A[k]))
print(" cRT gain off:", ci(A["crt_rare"] - A["trained_rare"]), " on:", ci(B["crt_rare"] - B["trained_rare"]))
print(" LA gain off:", ci(A["la_rare"] - A["la0_rare"]), " on:", ci(B["la_rare"] - B["la0_rare"]))
print(" cRT bal gain off:", ci(A["crt_bal"] - A["trained_bal"]))
print(" LA bal gain off:", ci(A["la_bal"] - A["la0_bal"]), "macro", ci(A["la_macro"] - A["la0_macro"]), "acc", ci(A["la_acc"] - A["la0_acc"]))
print(" cRT macro off", ci(A["crt_macro"]-A["trained_macro"]), "acc", ci(A["crt_acc"]-A["trained_acc"]))
print(" LA-loss bal vs CE:", ci(X_tiny_norare_la_uni["trained_bal"] - A["trained_bal"]))
print(" CBS bal vs CE:", ci(X_tiny_norare_ce_cbs["trained_bal"] - A["trained_bal"]))
print(" NCM: on-off ncm_acc", ci(B["ncm_acc"] - A["ncm_acc"]))
# fraction of loss recovered by cRT
loss = A["trained_rare"] - B["trained_rare"]; rec = B["crt_rare"] - B["trained_rare"]
print(" policy loss", ci(-loss), " recovered by cRT", ci(rec), " remaining after cRT (on cRT - off cRT)", ci(B["crt_rare"] - A["crt_rare"]))
print("== DF1 uniform")
for tag in ["test_best", "test_final"]:
    for m in ["acc", "macro_f1", "rare_f1", "common_f1"]:
        V = {k: np.array([df1[f"tiny_{k}_s{s}"][tag][m] * 100 for s in S]) for k in ["norare", "rare", "norare_uni", "rare_uni"]}
        print(tag, m, " ".join(f"{k} {ms(v)}" for k, v in V.items()))
        print("    RS(off)", ci(V["norare"] - V["norare_uni"]), "| aug@uni", ci(V["rare_uni"] - V["norare_uni"]))
