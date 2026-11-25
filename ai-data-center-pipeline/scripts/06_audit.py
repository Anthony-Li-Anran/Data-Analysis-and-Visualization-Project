# -*- coding: utf-8 -*-
"""
06_audit.py —— 质量审计与敏感性分析

产出（全部写入 results/）：
    audit.json                    数据集不变量检查 + 输入文件指纹
    row_audit.csv                 567 行逐行标记：观测/预测、是否投运、规则命中、是否定义承诺上限
    state_audit.csv               零投运站点的状态原文 + 分类依据（供人工抽检）
    definition_sensitivity.csv    "承诺容量" 5 种定义下的兑现率
    cutoff_sensitivity.csv        冻结日前后滑动 ±30/±60 天，兑现率如何漂移

这些文件的作用：让"22% 兑现率"这个结论可以被逐条挑战而不倒。
"""
from __future__ import print_function
import hashlib
import io
import json
import os

import numpy as np
import pandas as pd

import _paths
import _stage

BASE = _paths.BASE
DAT = _paths.RESULTS
_paths.ensure(DAT)

CUTOFF = pd.Timestamp("2026-10-21")
H100 = "H100 equivalents"
CUR = "Current H100 equivalents (from Data center)"
ITMW = "IT power (MW)"
MW = "Power (MW)"
OPS = "Buildings operational"
M_DAYS = 30.44


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load():
    tl = pd.read_csv(os.path.join(BASE, "data_center_timelines.csv"))
    tl["Date"] = pd.to_datetime(tl["Date"], errors="coerce")
    tl = tl.sort_values(["Data center", "Date"]).reset_index(drop=True)
    tl["is_projection"] = tl["Date"] > CUTOFF
    dc = pd.read_csv(os.path.join(BASE, "data_centers.csv"))
    return tl, dc


# ---------------------------------------------------------------- 1. 不变量
def build_audit(tl, dc, out, review_stats=None):
    checks = []

    def chk(name, passed, detail, expect=None):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail, "expect": expect})

    chk("行数 = 567", len(tl) == 567, len(tl), 567)
    chk("站点数 = 95", tl["Data center"].nunique() == 95,
        tl["Data center"].nunique(), 95)
    chk("两表站点集合一致",
        set(tl["Data center"]) == set(dc["Name"]),
        {"only_timelines": sorted(set(tl["Data center"]) - set(dc["Name"])),
         "only_snapshot": sorted(set(dc["Name"]) - set(tl["Data center"]))},
        "双向差集为空")
    dup = int(tl.duplicated(subset=["Data center", "Date"]).sum())
    chk("无重复(站点,日期)", dup == 0, dup, 0)

    n_act = int((~tl["is_projection"]).sum())
    n_proj = int(tl["is_projection"].sum())
    chk("实测行 = 469", n_act == 469, n_act, 469)
    chk("预测行 = 98", n_proj == 98, n_proj, 98)

    # actual <= planned 全站成立
    act_max = tl[~tl["is_projection"]].groupby("Data center")[H100].max()
    plan_max = tl.groupby("Data center")[H100].max()
    bad = [(k, float(act_max.get(k, 0)), float(v))
           for k, v in plan_max.items() if float(act_max.get(k, 0)) > float(v) + 1e-6]
    chk("actual <= planned 全站成立", not bad, bad[:5], "无越界站点")

    # 快照值 = 时间线最大实测值
    snap = dc.set_index("Name")["Current H100 equivalents"]
    pair = pd.DataFrame({"snap": snap, "act": act_max}).dropna()
    diff = (pair["snap"] - pair["act"]).abs()
    chk("快照值 = 最大实测值", int((diff < 1e-6).sum()) == len(pair),
        {"equal": int((diff < 1e-6).sum()), "total": len(pair)},
        "95/95 相等")

    miss = tl[~tl["is_projection"] & tl[H100].isna()]
    chk("实测行 H100e 缺失行数（已知 3 行，聚合时按 NaN 处理）",
        len(miss) == 3,
        {"rows": int(len(miss)),
         "sites": sorted(miss["Data center"].unique().tolist()),
         "dates": [str(d.date()) for d in miss["Date"]]},
        "3 行缺失，已知且不影响站点级 max")

    both = tl[(tl[ITMW] > 0) & (tl[MW] > 0)]
    ratio = (both[MW] / both[ITMW])
    chk("Power >= IT power 的行占比 > 60%",
        float((ratio >= 0.999).mean()) > 0.6,
        round(float((ratio >= 0.999).mean()), 4), "> 0.6")
    chk("Power/IT 比值在 1.0-1.6 之间（PUE 合理区间）",
        bool(ratio.between(1.0, 1.6).mean() > 0.95),
        {"median": round(float(ratio.median()), 3),
         "share_in_range": round(float(ratio.between(1.0, 1.6).mean()), 4)},
        "中位约 1.30")

    water_missing = float(tl["Water use (MGD)"].isna().mean())
    chk("水耗字段缺失率 > 80%（不可用，仅记录）", water_missing > 0.8,
        round(water_missing, 4), "> 0.8")

    doc = {
        "generated_for": "AI 数据中心产能管线与实际兑现",
        "cutoff": str(CUTOFF.date()),
        "inputs": {
            "data_center_timelines.csv": {
                "md5": md5(os.path.join(BASE, "data_center_timelines.csv")),
                "rows": int(len(tl))},
            "data_centers.csv": {
                "md5": md5(os.path.join(BASE, "data_centers.csv")),
                "rows": int(len(dc))},
        },
        "state_classification_review": review_stats,
        "checks": checks,
        "summary": {
            "passed": int(sum(1 for c in checks if c["passed"])),
            "failed": int(sum(1 for c in checks if not c["passed"])),
            "failed_names": [c["check"] for c in checks if not c["passed"]],
        },
    }
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, ensure_ascii=False, indent=2))
    print("audit: %d 项检查, %d 通过, %d 未通过%s"
          % (len(checks), doc["summary"]["passed"], doc["summary"]["failed"],
             "" if not doc["summary"]["failed_names"] else " -> " + ", ".join(doc["summary"]["failed_names"])))
    return doc


# ---------------------------------------------------------------- 2. 逐行审计
def build_row_audit(tl):
    rows = []
    for name, sub in tl.groupby("Data center"):
        sub = sub.sort_values("Date")
        act = sub[~sub["is_projection"]]
        first_obs = act["Date"].iloc[0] if len(act) else pd.NaT
        op = act[act[H100] > 0]
        first_op = op["Date"].iloc[0] if len(op) else pd.NaT
        plan = sub[H100].max()
        stage = _stage.classify(act, plan)
        for _, r in sub.iterrows():
            hit = [k for k in _stage.BUILD_KW if k in str(r["Construction status"]).lower()]
            rows.append({
                "site": name,
                "date": r["Date"].date(),
                "is_projection": bool(r["is_projection"]),
                "h100e": r[H100],
                "it_power_mw": r[ITMW],
                "power_mw": r[MW],
                "buildings_operational": r[OPS],
                "is_first_actual_obs": bool(pd.notna(first_obs) and r["Date"] == first_obs and not r["is_projection"]),
                "is_first_operational": bool(pd.notna(first_op) and r["Date"] == first_op and not r["is_projection"]),
                "defines_planned_max": bool(abs(float(r[H100] or 0) - float(plan or 0)) < 1e-6),
                "excluded": False,
                "exclude_reason": "",
                "site_stage": stage,
                "status_keywords_hit": "|".join(hit[:4]),
                "status_len": len(str(r["Construction status"])),
            })
    df = pd.DataFrame(rows).sort_values(["site", "date"])
    return df


# ---------------------------------------------------------------- 3. 分类抽检
def build_state_audit(tl):
    rows = []
    for name, sub in tl.groupby("Data center"):
        sub = sub.sort_values("Date")
        act = sub[~sub["is_projection"]]
        plan = sub[H100].max()
        if len(act) and act[H100].max() > 0:
            continue                                  # 只看零投运站点
        last = act.iloc[-1] if len(act) else None
        stage = _stage.classify(act, plan)
        text = str(last["Construction status"]) if last is not None else ""
        rows.append({
            "site": name,
            "assigned_stage": stage,
            "plan_h100e": plan,
            "obs_span_months": round(((act["Date"].max() - act["Date"].min()).days / M_DAYS)
                                     if len(act) > 1 else np.nan, 1),
            "last_obs": last["Date"].date() if last is not None else None,
            "it_power_mw": last[ITMW] if last is not None else None,
            "buildings_operational": last[OPS] if last is not None else None,
            "keywords_hit": "|".join([k for k in _stage.BUILD_KW if k in text.lower()]),
            "status_text": text[:300],
            "manual_verdict": "",                          # 留给人工填：agree / disagree
        })
    return pd.DataFrame(rows).sort_values("plan_h100e", ascending=False)


# ---------------------------------------------------------------- 3b. 人工判读与精细阶段
def load_review(dat):
    """读取人工抽检表（存在则合并）。"""
    f = os.path.join(dat, "state_review_manual.csv")
    if not os.path.isfile(f):
        return None
    return pd.read_csv(f)


def refine_stage(site_row, review_map):
    """把'在建（土建阶段）'按人工判读拆成土建/机电两档。"""
    st = site_row["stage"]
    m = review_map.get(site_row["site"])
    if m and st == "在建（土建阶段）":
        return m
    return st


def build_stage_refined(dat, review):
    """5 档阶段汇总（人工判读优先，其余沿用自动分类）。"""
    f = os.path.join(dat, "site_stage.csv")
    if not os.path.isfile(f):
        return None
    site = pd.read_csv(f)
    review_map = dict(zip(review["site"], review["manual_stage"])) if review is not None else {}
    site["stage_refined"] = site.apply(lambda r: refine_stage(r, review_map), axis=1)
    g = (site.groupby("stage_refined")
             .agg(站点数=("site", "size"),
                  承诺容量_M=("planned_h100e", lambda s: s.sum() / 1e6),
                  已投运_M=("actual_h100e", lambda s: s.sum() / 1e6),
                  观察跨度中位月=("span_months", "median"))
             .reset_index())
    order = ["已投运（达产）", "部分投运", "在建（机电安装）", "在建（土建阶段）", "前期/土地平整"]
    g["_o"] = g["stage_refined"].map({k: i for i, k in enumerate(order)}).fillna(99)
    g = g.sort_values("_o").drop(columns="_o")
    g["容量占比"] = (g["承诺容量_M"] / g["承诺容量_M"].sum()).round(4)
    return g


# ---------------------------------------------------------------- 4. 定义敏感性
def compute_rates(tl, cutoff):
    act = tl[tl["Date"] <= cutoff]
    proj = tl[tl["Date"] > cutoff]
    act_max = act.groupby("Data center")[H100].max()
    plan_all = tl.groupby("Data center")[H100].max()
    has_proj = set(proj["Data center"])
    return act, act_max, plan_all, has_proj


def build_definition_sensitivity(tl, dc):
    act, act_max, plan_all, has_proj = compute_rates(tl, CUTOFF)
    rows = []

    def add(name, a, p, note):
        rows.append({"definition": name,
                     "actual_M": round(a / 1e6, 3),
                     "planned_M": round(p / 1e6, 3),
                     "rate_pct": round(100.0 * a / p, 2) if p else np.nan,
                     "note": note})

    # A 基线：全部 95 站，planned = 全站最高累计
    add("A 基线（95 站，全行最高）", act_max.sum(), plan_all.sum(),
        "论文口径；含 41 个无承诺站点")

    # B 只算有预测行的 54 站
    keep = [s for s in plan_all.index if s in has_proj]
    add("B 仅有扩张承诺的站点", act_max.reindex(keep).sum(),
        plan_all.reindex(keep).sum(), "剔除 planned==actual 的构造性 100%")

    # C planned 只取预测行最高值
    proj_max = tl[tl["Date"] > CUTOFF].groupby("Data center")[H100].max()
    add("C planned 仅取预测行", act_max.reindex(proj_max.index).sum(),
        proj_max.sum(), "与 B 等价 -> 说明最高规模总出现在预测行，从不来自实测行")

    # G 只算已投运的 74 站（剔除 21 个从未上线站点）
    op_sites = act_max[act_max > 0].index
    add("G 只算已投运站点（%d 站）" % len(op_sites),
        act_max.reindex(op_sites).sum(), plan_all.reindex(op_sites).sum(),
        "剔除从未上线站点后的兑现率")

    # D 站点级兑现率中位数（54 站）
    ratio = (act_max.reindex(keep) / plan_all.reindex(keep)).dropna()
    rows.append({"definition": "D 站点兑现率的中位数（54 站）", "actual_M": np.nan,
                 "planned_M": np.nan, "rate_pct": round(100.0 * float(ratio.median()), 2),
                 "note": "不含容量加权"})

    # E IT 功率口径
    act_pw = act.groupby("Data center")[ITMW].max().sum()
    plan_pw = tl.groupby("Data center")[ITMW].max().sum()
    rows.append({"definition": "E IT 功率口径（95 站）", "actual_M": round(act_pw / 1e3, 3),
                 "planned_M": round(plan_pw / 1e3, 3),
                 "rate_pct": round(100.0 * act_pw / plan_pw, 2),
                 "note": "单位 GW；与 README 的 2.69x 对应"})

    # F 总功率口径
    act_pw2 = act.groupby("Data center")[MW].max().sum()
    plan_pw2 = tl.groupby("Data center")[MW].max().sum()
    rows.append({"definition": "F 总功率口径（95 站）", "actual_M": round(act_pw2 / 1e3, 3),
                 "planned_M": round(plan_pw2 / 1e3, 3),
                 "rate_pct": round(100.0 * act_pw2 / plan_pw2, 2),
                 "note": "单位 GW；含 PUE 开销"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5. 冻结日敏感性
def build_cutoff_sensitivity(tl):
    rows = []
    plan_all = tl.groupby("Data center")[H100].max()
    for shift in [-60, -30, 0, 30, 60]:
        cut = CUTOFF + pd.Timedelta(days=shift)
        act = tl[tl["Date"] <= cut]
        proj = tl[tl["Date"] > cut]
        a = act.groupby("Data center")[H100].max().sum()
        rows.append({
            "cutoff_shift_days": shift,
            "cutoff": str(cut.date()),
            "actual_rows": int(len(act)),
            "projection_rows": int(len(proj)),
            "actual_M": round(a / 1e6, 3),
            "planned_M": round(plan_all.sum() / 1e6, 3),
            "rate_pct": round(100.0 * a / plan_all.sum(), 2),
        })
    return pd.DataFrame(rows)


def build_caliber_table(tl, dc):
    """算力 vs 功率 三口径对照 + 芯片密度假设拆解。"""
    act = tl[tl["Date"] <= CUTOFF]
    a_h = act.groupby("Data center")[H100].max().sum()
    p_h = tl.groupby("Data center")[H100].max().sum()
    a_it = act.groupby("Data center")[ITMW].max().sum()
    p_it = tl.groupby("Data center")[ITMW].max().sum()
    a_pw = act.groupby("Data center")[MW].max().sum()
    p_pw = tl.groupby("Data center")[MW].max().sum()
    return pd.DataFrame([
        {"caliber": "H100e", "actual": round(a_h / 1e6, 3), "planned": round(p_h / 1e6, 3),
         "unit": "M H100e", "rate_pct": round(100 * a_h / p_h, 2),
         "multiplier": round(p_h / a_h, 3),
         "density_per_it_mw_actual": round(a_h / a_it, 1),
         "density_per_it_mw_planned": round(p_h / p_it, 1),
         "density_per_total_mw_actual": round(a_h / a_pw, 1),
         "density_per_total_mw_planned": round(p_h / p_pw, 1)},
        {"caliber": "IT 功率", "actual": round(a_it / 1e3, 3), "planned": round(p_it / 1e3, 3),
         "unit": "GW", "rate_pct": round(100 * a_it / p_it, 2),
         "multiplier": round(p_it / a_it, 3),
         "density_per_it_mw_actual": np.nan, "density_per_it_mw_planned": np.nan,
         "density_per_total_mw_actual": np.nan, "density_per_total_mw_planned": np.nan},
        {"caliber": "总功率", "actual": round(a_pw / 1e3, 3), "planned": round(p_pw / 1e3, 3),
         "unit": "GW", "rate_pct": round(100 * a_pw / p_pw, 2),
         "multiplier": round(p_pw / a_pw, 3),
         "density_per_it_mw_actual": np.nan, "density_per_it_mw_planned": np.nan,
         "density_per_total_mw_actual": np.nan, "density_per_total_mw_planned": np.nan},
    ])


def main():
    tl, dc = load()

    # 1
    _rev = load_review(DAT)
    _stats = None
    if _rev is not None:
        _stats = {
            "reviewed": int(len(_rev)),
            "coarse_agree": int(_rev["coarse_agree"].sum()),
            "fine_agree": int(_rev["fine_agree"].sum()),
            "coarse_accuracy_pct": round(100 * float(_rev["coarse_agree"].mean()), 1),
            "fine_accuracy_pct": round(100 * float(_rev["fine_agree"].mean()), 1),
            "misclassified": _rev.loc[~_rev["fine_agree"], ["site", "auto_stage", "manual_stage"]]
                                .to_dict(orient="records"),
            "note": "粗分类=在建/前期；细分类=土建阶段/机电安装。误判集中在把机电安装标成土建。",
        }
    build_audit(tl, dc, os.path.join(DAT, "audit.json"), _stats)

    # 2
    ra = build_row_audit(tl)
    ra.to_csv(os.path.join(DAT, "row_audit.csv"), index=False, encoding="utf-8-sig")
    print("row_audit: %d 行" % len(ra))

    # 3
    sa = build_state_audit(tl)
    review = load_review(DAT)
    if review is not None:
        sa = sa.drop(columns=["manual_verdict"]).merge(
            review[["site", "manual_stage", "coarse_agree", "fine_agree", "reason"]],
            on="site", how="left")
    sa.to_csv(os.path.join(DAT, "state_audit.csv"), index=False, encoding="utf-8-sig")
    if review is not None:
        print("state_audit: %d 站，人工已判读 %d 站（粗分类一致 %.0f%%，细分类一致 %.0f%%）"
              % (len(sa), len(review),
                 100 * review["coarse_agree"].mean(), 100 * review["fine_agree"].mean()))
    else:
        print("state_audit: %d 个零投运站点（待人工抽检）" % len(sa))

    # 3b 精细阶段
    refined = build_stage_refined(DAT, review)
    if refined is not None:
        refined.to_csv(os.path.join(DAT, "stage_counts_refined.csv"),
                       index=False, encoding="utf-8-sig")
        print("stage_counts_refined:")
        print(refined.to_string(index=False))

    # 4
    ds = build_definition_sensitivity(tl, dc)
    ds.to_csv(os.path.join(DAT, "definition_sensitivity.csv"), index=False, encoding="utf-8-sig")
    print("definition_sensitivity:")
    print(ds[["definition", "rate_pct"]].to_string(index=False))

    # 5
    cs = build_cutoff_sensitivity(tl)
    cs.to_csv(os.path.join(DAT, "cutoff_sensitivity.csv"), index=False, encoding="utf-8-sig")
    print("cutoff_sensitivity: %s" % " / ".join(
        "%+d天 %.2f%%" % (r["cutoff_shift_days"], r["rate_pct"]) for _, r in cs.iterrows()))

    # 6
    ct = build_caliber_table(tl, dc)
    ct.to_csv(os.path.join(DAT, "caliber_comparison.csv"), index=False, encoding="utf-8-sig")
    print("caliber_comparison:")
    print(ct[["caliber", "actual", "planned", "unit", "rate_pct", "multiplier"]].to_string(index=False))


if __name__ == "__main__":
    main()
