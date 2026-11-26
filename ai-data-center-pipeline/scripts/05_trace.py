# -*- coding: utf-8 -*-
"""
05_trace.py —— 交互式追踪：把某个站点的"原始行 -> 清洗 -> 指标"全过程打出来。

用法:
    python 05_trace.py "Colossus"            # 模糊匹配站点名
    python 05_trace.py "Stargate New Mexico"
    python 05_trace.py --list                # 列出全部站点名

它复现 02/04 的全部判定，只是把中间结果打印出来，用于核对和答辩自证。
"""
from __future__ import print_function
import os, sys, argparse
import numpy as np
import pandas as pd

import _paths

BASE = _paths.BASE
CUTOFF = pd.Timestamp("2026-10-21")
H100 = "H100 equivalents"
CUR = "Current H100 equivalents (from Data center)"
ITMW = "IT power (MW)"
MW = "Power (MW)"
OPS = "Buildings operational"
M_DAYS = 30.44

BUILD_KW = ["construct", "foundation", "steel", "roof", "shell", "chiller",
            "cooling", "generator", "building", "permit", "land clearing",
            "excavat", "concrete", "grading", "earthwork"]
EARLY_KW = ["land clearing begins", "first signs of development"]


def load():
    tl = pd.read_csv(os.path.join(BASE, "data_center_timelines.csv"))
    tl["Date"] = pd.to_datetime(tl["Date"], errors="coerce")
    tl = tl.dropna(subset=["Date"]).sort_values(["Data center", "Date"])
    tl["proj"] = tl["Date"] > CUTOFF
    dc = pd.read_csv(os.path.join(BASE, "data_centers.csv"))
    return tl, dc


def classify(act, plan):
    if len(act) == 0:
        return "无观测"
    a = act[H100].max()
    if a > 0:
        return "已投运（达产）" if a >= 0.999 * plan else "部分投运"
    if act[ITMW].max() > 0 or act[OPS].max() > 0:
        return "在建（已有机电/建筑）"
    last = str(act["Construction status"].iloc[-1]).lower()
    if any(k in last for k in BUILD_KW) and not any(k in last for k in EARLY_KW):
        return "在建（土建阶段）"
    return "前期/土地平整"


def trace(name, tl, dc):
    site = tl[tl["Data center"] == name]
    if len(site) == 0:
        print("没有这个站点")
        return
    print("=" * 78)
    print("站点:", name)
    row = dc[dc["Name"] == name]
    if len(row):
        r = row.iloc[0]
        print("快照表: 当前 H100e = %.0f | 当前功率 = %.0f MW | 国别 = %s | 业主 = %s"
              % (r["Current H100 equivalents"], r["Current power (MW)"],
                 r["Country"], r["Owner"]))
    print("=" * 78)

    print("\n[第1步] 时间线原始行（%d 行）" % len(site))
    print("%-12s %10s %9s %8s %6s  %s" %
          ("Date", "H100e", "IT_MW", "Power_MW", "楼数", "类型"))
    print("-" * 78)
    for _, s in site.iterrows():
        print("%-12s %10.0f %9.0f %8.0f %6.0f  %s"
              % (s["Date"].date(), s[H100], s[ITMW], s[MW], s[OPS],
                 "projection" if s["proj"] else "actual"))

    act = site[~site["proj"]]
    proj = site[site["proj"]]
    print("\n[第2步] 按冻结日 2026-10-21 切分")
    print("  actual 行 = %d, projection 行 = %d" % (len(act), len(proj)))
    print("  观测不可能发生在未来 -> projection 行不参与'已投运'判定")

    a = act[H100].max() if len(act) else np.nan
    p = site[H100].max()
    print("\n[第3步] 站点级压缩：多行累计值 -> 两个数")
    print("  actual_h100e  = max(actual 行的 H100e)     = %.0f" % a)
    print("  planned_h100e = max(全部行的 H100e)        = %.0f" % p)
    print("  兑现率 = %.0f / %.0f = %.1f%%" % (a, p, 100 * a / p if p else np.nan))

    op = act[act[H100] > 0]
    first_obs = act["Date"].iloc[0] if len(act) else pd.NaT
    last_obs = act["Date"].iloc[-1] if len(act) else pd.NaT
    first_op = op["Date"].iloc[0] if len(op) else pd.NaT
    print("\n[第4步] 时间线与滞后")
    print("  first_obs（首次看到施工） = %s" % (first_obs.date() if pd.notna(first_obs) else "—"))
    print("  first_op （首次 H100e>0） = %s" % (first_op.date() if pd.notna(first_op) else "—"))
    print("  last_obs （最后一条实测） = %s" % (last_obs.date() if pd.notna(last_obs) else "—"))
    if pd.notna(first_op):
        print("  lag_months = (first_op - first_obs) = %.1f 个月"
              % ((first_op - first_obs).days / M_DAYS))
    else:
        print("  lag_months = 不可计算（尚未投运）, 改用观察跨度:")
    print("  span_months = (last_obs - first_obs) = %.1f 个月"
          % ((last_obs - first_obs).days / M_DAYS if pd.notna(first_obs) else np.nan))

    print("\n[第5步] 建设状态分类")
    print("  规则: H100e>0 且达承诺 -> 已投运(达产); H100e>0 -> 部分投运;")
    print("        否则看 IT功率/楼数 -> 在建(机电); 否则看状态文本 -> 在建(土建)/前期")
    print("  结果 =", classify(act, p))
    if len(act):
        print("  依据: 末条实际观测状态文本 = \"%s\""
              % str(act["Construction status"].iloc[-1])[:90])

    print("\n[第6步] 该站点对年度序列的贡献（累计值按年末快照、逐年结转）")
    for y in sorted(set(act["Date"].dt.year) | set(proj["Date"].dt.year)):
        av = act[act["Date"].dt.year == y][H100].max()
        pv = proj[proj["Date"].dt.year == y][H100].max()
        print("  %d: actual=%-12s projection=%s"
              % (y, "%.0f" % av if pd.notna(av) else "—",
                 "%.0f" % pv if pd.notna(pv) else "—"))

    print("\n[第7步] 该站点进入哪些图")
    print("  fig1_gap    : 通过年度累计/承诺贡献")
    print("  fig2_sites  : %s" % ("是（承诺前18）" if p >= 2.0e6 else "否"))
    print("  fig3_lag    : %s" % ("是" if pd.notna(first_op) else "否（零投运）"))
    print("  fig4_timeline: %s" % ("是（承诺前22）" if p >= 1.4e6 else "否"))
    print("  fig5_owner  : 通过业主分类")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    tl, dc = load()
    if a.list or not a.name:
        for n in sorted(tl["Data center"].unique()):
            print(n)
        return
    hits = [n for n in tl["Data center"].unique()
            if a.name.lower() in n.lower()]
    if not hits:
        print("没匹配到站点，用 --list 看全部名称")
        return
    for h in hits[:3]:
        trace(h, tl, dc)
        print()


if __name__ == "__main__":
    main()
