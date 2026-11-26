# -*- coding: utf-8 -*-
"""
03_charts.py —— 三张核心图(开题汇报用, 供组长拼 PPT)
    fig1_gap.png        累计算力: 已兑现 vs 已承诺, 以及两者之间的缺口
    fig2_sites.png      站点级: 规划规模 vs 已投运规模(双对数 + 45 度线)
    fig3_lag.png        从首次观测到首次投运的滞后分布 + 零投运站点的观察跨度

统一风格: 白底、无顶右框、大字号(PPT 投影可读)、图注写在下方。
"""
from __future__ import print_function
import io, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

import _paths

DAT = _paths.RESULTS
_paths.ensure(DAT)

# ---- 中文字体
for cand in ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DengXian", "Source Han Serif SC"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 12,
    "figure.dpi": 150,
})

INK = "#1a1a1a"
BLUE = "#2b6cb0"
RED = "#c0392b"
GRAY = "#9aa0a6"
LIGHT = "#e8eef5"

CN = dict(
    actual="已投运（实际观测）",
    committed="已承诺（含公司排期与估计）",
    gap="兑现缺口",
)


def style_ax(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, color="#e6e6e6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def wan(x, pos=None):
    """亿/万 级中文刻度"""
    if x >= 1e8:
        return "%.0f亿" % (x / 1e8)
    if x >= 1e4:
        return "%.0f万" % (x / 1e4)
    return "%.0f" % x


def footer(fig, text):
    fig.text(0.008, 0.012, text, fontsize=9, color="#666666", ha="left", va="bottom")


# ============================================================ 图1
def fig1(ann, summ):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = ann["year"]
    a, p = ann["cum_actual_h100e"] / 1e6, ann["cum_committed_h100e"] / 1e6

    ax.fill_between(x, a, p, where=p >= a, color="#f6d5d1", alpha=0.9,
                    interpolate=True, zorder=1, label=CN["gap"])
    ax.plot(x, p, "-o", color=RED, lw=2.6, ms=6, zorder=4, label=CN["committed"])
    ax.plot(x, a, "-o", color=BLUE, lw=2.6, ms=6, zorder=5, label=CN["actual"])

    # 采集日分界: 左侧是历史, 右侧是承诺
    split = 2026
    ax.axvline(split + 0.5, color=GRAY, ls="--", lw=1.4, zorder=2)
    ax.text(split + 0.58, a.max() * 0.55, "截至采集日\n（2026-10-21）", fontsize=10.5,
            color="#555555", va="center")

    i26 = ann.index[ann["year"] == 2026][0]
    i28 = ann.index[ann["year"] == 2028][0]
    ax.annotate("已投运 %.1fM H100e" % a.iloc[i26], xy=(2026, a.iloc[i26]),
                xytext=(2023.1, a.iloc[i26] * 2.4), fontsize=11.5, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.0))
    ax.annotate("承诺总量 %.1fM H100e" % p.iloc[i28], xy=(2028, p.iloc[i28]),
                xytext=(2024.6, p.iloc[i28] * 0.72), fontsize=11.5, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=1.0))
    gap8 = p.iloc[i28] - a.iloc[i28]
    ax.annotate("缺口 %.1fM H100e\n（兑现率 %.0f%%）" % (gap8, 100 * summ["realization_rate"]),
                xy=(2028, (p.iloc[i28] + a.iloc[i28]) / 2),
                xytext=(2028.35, (p.iloc[i28] + a.iloc[i28]) / 2), fontsize=12,
                color="#8e2b20", va="center",
                arrowprops=dict(arrowstyle="->", color="#8e2b20", lw=1.2))

    ax.set_xlim(2020.6, 2031.2)
    ax.set_ylim(0, p.max() * 1.12)
    ax.set_xlabel("年份")
    ax.set_ylabel("累计 AI 算力（百万 H100 等效）")
    ax.set_title("AI 数据中心算力：已兑现 vs 已承诺", loc="left", color=INK, pad=14)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p_: "%.0fM" % v))
    style_ax(ax)
    ax.legend(loc="upper left", frameon=False)
    footer(fig, "数据：Epoch AI, AI Data Centers / Data Center Timelines（数据截至 {}）\n"
                "已投运 = 采集日前有卫星或许可证据的实际累计容量；"
                "已承诺 = 叠加 Epoch 对未来里程碑的估计。".format(summ["cutoff"]))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.90, bottom=0.155)
    out = _paths.save(fig, "01")
    plt.close(fig)
    return out


# ============================================================ 图2
def fig2(site, topn=18):
    d = site[site["planned_h100e"] > 0].nlargest(topn, "planned_h100e")
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(11.2, 7.4))
    y = np.arange(len(d))
    act = d["actual_h100e"].fillna(0) / 1e6
    pend = d["pending_h100e"].clip(lower=0) / 1e6

    ax.barh(y, act, height=0.7, color=BLUE, zorder=3, label="已投运")
    ax.barh(y, pend, left=act, height=0.7, color="#f3bdb7", zorder=2,
            label="承诺但尚未投运")

    labels = [s if len(s) <= 24 else s[:23] + "…" for s in d["site"]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for i, (a, p) in enumerate(zip(act, pend)):
        tot = a + p
        rate = a / tot if tot else 0
        ax.text(tot + 0.06, i, "%.0f%%" % (rate * 100), va="center", fontsize=11,
                color=RED if rate < 0.15 else "#333333",
                fontweight="bold" if rate < 0.15 else "normal")

    ax.set_xlabel("容量（百万 H100 等效）")
    ax.set_title("规模最大的站点里，多少已经真正上线", loc="left", color=INK, pad=14)
    ax.set_xlim(0, (act + pend).max() * 1.13)
    style_ax(ax, grid_axis="x")
    ax.legend(loc="lower right", frameon=False)
    n_zero = int((site["actual_h100e"].fillna(0) <= 0).sum())
    footer(fig, "数据：Epoch AI。右侧百分比 = 已投运 / 承诺容量；红色表示兑现率低于 15%%。\n"
                "全部 95 个站点中有 %d 个至今零投运（图中兑现率 0%% 的条形），"
                "其余站点的兑现率中位为 %.0f%%。"
           % (n_zero, 100 * site["realization"].median()))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.925, bottom=0.135)
    out = _paths.save(fig, "02")
    plt.close(fig)
    return out


# ============================================================ 图3
def fig3(site):
    op = site[site["is_operational"] & site["lag_months"].notna()]
    no = site[~site["is_operational"]]
    fig, ax = plt.subplots(figsize=(10.5, 6))
    bins = np.arange(0, 72, 6)
    ax.hist(op["lag_months"], bins=bins, color=BLUE, alpha=0.75,
            edgecolor="white", lw=1.2, zorder=3, label="已投运站点（n=%d）" % len(op))
    med = op["lag_months"].median()
    ax.axvline(med, color=RED, lw=2.2, zorder=5)
    ax.text(med - 1.5, ax.get_ylim()[1] * 0.94, "中位 %.0f 个月" % med, color=RED,
            fontsize=12.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

    ax.set_xlabel("从首次观测到首次投运的月数（= 滞后下限）")
    ax.set_ylabel("站点数")
    ax.set_title("算力上线有多慢：从开工迹象到真正投运", loc="left", color=INK, pad=14)
    style_ax(ax)
    ax.legend(loc="upper right", frameon=False)

    span = no["span_months"].median()
    txt = ("另有 %d 个站点至今零投运，\n观察跨度中位仅 %.0f 个月——\n"
           "它们的'未兑现'里，有相当部分\n只是观察窗还没走完（右删失）。"
           % (len(no), span))
    ax.text(0.985, 0.42, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=10.8, color="#333333",
            bbox=dict(boxstyle="round,pad=0.55", fc=LIGHT, ec="#c5d3e2", lw=1))
    footer(fig, "数据：Epoch AI Data Center Timelines。滞后以'首次观测到开工/土地平整'为起点，\n"
                "因此是真实滞后的下限（真正的'宣布'更早）。")
    fig.subplots_adjust(left=0.085, right=0.975, top=0.92, bottom=0.155)
    out = _paths.save(fig, "03")
    plt.close(fig)
    return out


def main():
    site = pd.read_csv(os.path.join(DAT, "site_level.csv"))
    ann = pd.read_csv(os.path.join(DAT, "annual.csv"))
    summ = json.load(io.open(os.path.join(DAT, "summary.json"), encoding="utf-8"))
    outs = [fig1(ann, summ), fig2(site), fig3(site)]
    for o in outs:
        print("[ok]", o)


if __name__ == "__main__":
    main()
