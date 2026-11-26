# -*- coding: utf-8 -*-
"""路径与出图约定（自包含版）。

    feng-datacenter-pipeline/
    ├── data/       输入副本（随分支走，clone 即跑）
    ├── results/    派生表 + 审计 + 敏感性
    ├── figures/    01-09 每张 PNG + SVG
    ├── docs/       文档与讲稿
    └── scripts/    脚本

默认从本目录的 data/ 读输入；需要换数据时设 DCP_BASE。
所有写入都落在本目录内，不动仓库其它位置。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # feng-datacenter-pipeline/
DATA_IN = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
DOCS = os.path.join(ROOT, "docs")
REPORTS = os.path.join(RESULTS, "reports")

BASE = os.environ.get("DCP_BASE") or DATA_IN      # 原始 CSV 所在目录

# 图编号 -> 文件名主体（与团队 01..09 的编号习惯对齐）
FIGS = {
    "01": "gap-cumulative",
    "02": "site-realization",
    "03": "lag-distribution",
    "04": "site-timeline",
    "05": "owner-split",
    "06": "country-focus",
    "07": "scale-vs-realization",
    "08": "cohort-censoring",
    "09": "caliber-sensitivity",
}


def ensure(*dirs):
    for d in dirs or (RESULTS, FIGURES, REPORTS):
        if not os.path.isdir(d):
            os.makedirs(d)


def figure_path(num):
    """返回不带扩展名的图路径，例如 figures/01-gap-cumulative。"""
    return os.path.join(FIGURES, "%s-%s" % (num, FIGS[num]))


def save(fig, num, dpi=150):
    """同时导出 PNG 与 SVG，返回两个路径。"""
    if not os.path.isdir(FIGURES):
        os.makedirs(FIGURES)
    base = figure_path(num)
    png, svg = base + ".png", base + ".svg"
    fig.savefig(png, dpi=dpi)
    fig.savefig(svg)
    return [png, svg]
