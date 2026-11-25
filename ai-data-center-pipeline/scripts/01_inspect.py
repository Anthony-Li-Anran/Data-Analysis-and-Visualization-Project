# -*- coding: utf-8 -*-
"""
00_inspect.py —— 数据表体检：在写任何清洗逻辑之前，先看清表长什么样。

用法:
    python 00_inspect.py "路径\\到\\epoch_datacenters.csv"
    python 00_inspect.py "D:\\data\\xxx.xlsx" --sheet 0

输出:
    - 文件编码 / 分隔符 / 行数
    - 每列: 类型、非空率、去重数、前 3 个取值样例
    - 疑似字段自动归位: 状态 / 日期 / 容量 / 功率 / 项目名 / 公司 / 地区
    - 抽 5 行完整样本
    - 列名猜测结果写盘 _schema_report.txt, 供后续脚本引用
"""
from __future__ import print_function
import io, os, sys, json, argparse

import pandas as pd

ENC_TRY = ["utf-8", "utf-8-sig", "gb18030", "latin-1"]

# 字段语义猜测用到的关键词(小写匹配)
SEMANTIC = {
    "status": ["status", "state", "stage", "phase", "stage of"],
    "date": ["date", "year", "month", "announce", "operation", "online",
             "commission", "ready", "start", "end", "timeline", "q1", "q2"],
    "capacity_compute": ["h100", "h200", "flop", "compute", "ai capacity", "h100e"],
    "capacity_power": ["mw", "gw", "kw", "power", "watt", "it power", "total power",
                       "facility"],
    "cost": ["cost", "usd", "$", "price", "capex"],
    "name": ["name", "site", "project", "facility", "datacenter", "data center"],
    "company": ["company", "owner", "operator", "developer", "firm", "tenant", "user"],
    "geo": ["country", "region", "state", "city", "location", "country code"],
}


def load(path, sheet=0):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet)
    last = None
    for enc in ENC_TRY:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception as e:                              # noqa: BLE001
            last = e
    raise last


def guess(cols):
    """按关键词给每列打语义标签, 输出 {语义: [(列名, 命中词)]}"""
    hits = {}
    for c in cols:
        low = str(c).lower()
        for sem, kws in SEMANTIC.items():
            for kw in kws:
                if kw in low:
                    hits.setdefault(sem, []).append((c, kw))
                    break
    return hits


def describe(df):
    lines = []
    lines.append("行数: %d   列数: %d" % (len(df), len(df.columns)))
    lines.append("")
    lines.append("%-34s %-10s %8s %8s  %s" %
                 ("列名", "dtype", "非空率", "去重数", "取值样例"))
    lines.append("-" * 110)
    for c in df.columns:
        s = df[c]
        nn = 1.0 - s.isna().mean()
        try:
            nun = s.nunique(dropna=True)
        except Exception:                                   # noqa: BLE001
            nun = -1
        vals = [str(v)[:28] for v in s.dropna().unique()[:3]]
        lines.append("%-34s %-10s %7.0f%% %8d  %s" %
                     (str(c)[:34], str(s.dtype)[:10], nn * 100, nun,
                      " | ".join(vals)))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--out", default=None, help="报告输出目录, 默认脚本同级 reports/")
    a = ap.parse_args()

    df = load(a.path, a.sheet)
    out = []
    out.append("# 文件: %s" % a.path)
    out.append("")
    out.append(describe(df))
    out.append("")

    hits = guess(df.columns)
    out.append("## 字段语义命中")
    for sem in ["name", "company", "geo", "status", "date", "capacity_power",
                "capacity_compute", "cost"]:
        if sem in hits:
            out.append("- %-18s -> %s" % (sem, ", ".join("%s(%s)" % h for h in hits[sem])))
        else:
            out.append("- %-18s -> 未命中, 需人工指定" % sem)
    out.append("")
    out.append("## 未归类的列")
    used = set(c for v in hits.values() for c, _ in v)
    out.append(", ".join([str(c) for c in df.columns if c not in used]) or "(无)")
    out.append("")
    out.append("## 样本行")
    with pd.option_context("display.max_columns", None, "display.width", 250):
        out.append(df.head(a.rows).to_string())

    # 状态字段取值全览: 这是"建设状态分类"的基础, 必须人工确认
    for c, _ in hits.get("status", []):
        out.append("")
        out.append("## 状态字段 `%s` 的全部取值" % c)
        vc = df[c].value_counts(dropna=False)
        for k, v in vc.items():
            out.append("  %-40s %d" % (str(k)[:40], v))

    text = "\n".join(out)
    dst_dir = a.out or os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "reports"))
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    dst = os.path.join(dst_dir, "_schema_report.txt")
    with io.open(dst, "w", encoding="utf-8") as f:
        f.write(text)
    sys.stdout.write(text + "\n\n[已写入] " + dst + "\n")


if __name__ == "__main__":
    main()
