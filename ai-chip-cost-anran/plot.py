"""Render the available report and supplementary figures from the analysis CSVs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
INK, TEAL, ORANGE, GRAY = "#182C2C", "#00736A", "#B85A32", "#536764"
FONT_PATHS = (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
)
for font in FONT_PATHS:
    if font.exists():
        fm.fontManager.addfont(str(font))
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=str(font)).get_name()
        break
else:
    matplotlib.rcParams["font.family"] = ["Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC"]
matplotlib.rcParams.update({
    "font.size": 11, "text.color": INK, "axes.labelcolor": GRAY, "axes.titlecolor": INK,
    "axes.unicode_minus": False, "axes.edgecolor": "#AEBCB9",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "xtick.color": GRAY, "ytick.color": GRAY, "xtick.major.size": 0, "ytick.major.size": 0,
    "svg.fonttype": "path",
})


def read(name):
    return pd.read_csv(ROOT / "results" / f"{name}.csv")


def grid(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DFE7E5", lw=0.8)


def heading(fig, title, subtitle):
    fig.text(.09, .95, title, va="top", fontsize=20, weight="bold")
    fig.text(.09, .879, subtitle, va="top", fontsize=11, color=GRAY)


def save(fig, name, note):
    fig.text(.09, .03, note, va="bottom", fontsize=9, color=GRAY)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, facecolor="white")
    plt.close(fig)


def trend(q):
    fig, (cost, supply) = plt.subplots(2, 1, sharex=True, figsize=(10, 6.25))
    fig.subplots_adjust(left=.09, right=.94, top=.79, bottom=.15, hspace=.43)
    heading(fig, "季度新增规格算力上升，单位芯片购置成本下降",
            "NVIDIA 样本 · 19 个季度 · 2022 Q1 至 2026 Q3")
    x = list(range(len(q)))
    cost.plot(x, q.usd_per_h100e / 1000, color=TEAL, lw=2.6, marker="o", ms=3.7)
    cost.set_title("单位购置成本（千美元 / H100e，名义价格）", loc="left", fontsize=12, pad=9)
    cost.set_ylim(0, 45)
    cost.set_yticks([0, 10, 20, 30, 40])
    cost.annotate("38.1", (0, q.usd_per_h100e.iloc[0] / 1000), xytext=(7, 5),
                  textcoords="offset points", color=TEAL, weight="bold")
    cost.annotate("14.9", (18, q.usd_per_h100e.iloc[-1] / 1000), xytext=(7, 3),
                  textcoords="offset points", color=TEAL, weight="bold")
    output_index = q.h100e / q.h100e.iloc[0]
    spend_index = q.spend_usd / q.spend_usd.iloc[0]
    supply.plot(x, output_index, color=TEAL, lw=2.6, label="出货折算算力")
    supply.plot(x, spend_index, color=ORANGE, ls="--", lw=2, label="估计购置总额")
    supply.set_title("算力出货量与购置总额（2022 Q1 = 1）", loc="left", fontsize=12, pad=9)
    supply.set_ylim(0, 90)
    supply.set_yticks([0, 20, 40, 60, 80])
    supply.legend(loc="upper left", frameon=False, ncol=2, fontsize=10)
    for series, color in [(output_index, TEAL), (spend_index, ORANGE)]:
        supply.annotate(f"{series.iloc[-1]:.1f} 倍", (18, series.iloc[-1]), xytext=(-4, 4),
                        textcoords="offset points", ha="right", color=color, weight="bold")
    supply.set_xticks([0, 4, 8, 12, 16, 18], ["22Q1", "23Q1", "24Q1", "25Q1", "26Q1", "26Q3"])
    for ax in (cost, supply):
        ax.set_xlim(-.65, 18.8)
        grid(ax)
    save(fig, "01-quarterly-trends", "数据取自 Epoch AI 芯片销售与规格表（2026-11-19 版）。H100e 基于密集 8 位峰值规格。")


def waterfall(change):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(left=.09, right=.94, top=.78, bottom=.18)
    heading(fig, "更强的出货组合抵消了单颗价格上升",
            "2024 Q4 至 2025 Q4 · Shapley 两因素分解 · 千美元 / H100e")
    base, price, spec, end = [change[k] / 1000 for k in
                              ["base_usd_per_h100e", "price_contribution",
                               "spec_contribution", "new_usd_per_h100e"]]
    ax.bar(range(4), [base, price, -spec, end], bottom=[0, base, end, 0],
           color=["#9DAEAB", ORANGE, TEAL, "#426360"], width=.58)
    for i, y in enumerate([base, base + price, end]):
        ax.hlines(y, i + .29, i + .71, color="#859794", ls="dashed", lw=1)
    for i, (y, value) in enumerate(zip([base, base + price, base + price, end], [base, price, spec, end])):
        label = f"{value:+.1f}" if i in (1, 2) else f"{value:.1f}"
        ax.text(i, y + .9, label, ha="center", weight="bold", fontsize=12)
    ax.set_xticks(range(4), ["2024 Q4", "平均单颗价格", "平均单颗算力", "2025 Q4"])
    ax.set_ylim(0, 40)
    grid(ax)
    save(fig, "02-cost-decomposition",
         "根据 Epoch AI 数据计算。贡献取两种替换顺序的平均值，显示值四舍五入。")


def precision(q):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(left=.09, right=.94, top=.75, bottom=.18)
    heading(fig, "同一季度，换精度口径后成本方向反转",
            "每单位 H100 峰值算力的芯片购置成本（千美元）")
    periods = q.set_index("quarter").loc[["2026Q2", "2026Q3"]]
    groups = [periods.usd_per_h100e / 1000, periods.usd_per_h100e16 / 1000]
    for i, bars in enumerate(groups):
        ax.bar([i - .16, i + .16], bars, color=["#A5B8B4", TEAL], width=.28)
        for xpos, value in zip([i - .16, i + .16], bars):
            ax.text(xpos, value + .32, f"{value:.2f}", ha="center", fontsize=11, weight="bold")
        ax.text(i, 19.75, f"环比 {(bars.iloc[1]/bars.iloc[0]-1):+.1%}", ha="center",
                fontsize=12, weight="bold", color=TEAL if i == 0 else ORANGE)
    ax.legend(handles=[Patch(color="#A5B8B4", label="2026 Q2"),
                       Patch(color=TEAL, label="2026 Q3")], ncol=2, loc="upper right",
              bbox_to_anchor=(1, 1.15), frameon=False)
    ax.set_xticks([0, 1], ["密集 8 位", "密集 16 位"])
    ax.set_ylim(0, 22)
    ax.set_yticks([0, 5, 10, 15, 20])
    grid(ax)
    save(fig, "03-precision-sensitivity",
         "根据 Epoch AI 数据计算。两种精度各用 H100 的对应峰值规格作基准。")


def benchmark_costs(cases):
    nebius = cases.loc[cases.provider.eq("Nebius")].reset_index(drop=True)
    labels = nebius.Model.map({"deepseek-r1": "DeepSeek-R1", "gpt-oss-120b": "GPT-OSS-120B"})
    labels = labels + " · " + nebius.Scenario
    saved = (1 - nebius.cost_ratio_b300_to_b200) * 100
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(left=.30, right=.93, top=.76, bottom=.20)
    heading(fig, "B300 时租高 11.8%，每百万输出 token 成本低 2% 至 9%",
            "Nebius 单机 8 卡实测吞吐 × 同品牌按需公开标价 · 满负载情景")
    ax.barh(range(len(saved)), saved, color=TEAL, height=.58)
    for i, value in enumerate(saved):
        ax.text(value + .16, i, f"{value:.1f}%", va="center", weight="bold")
    ax.set_yticks(range(len(saved)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 10.5)
    ax.set_xticks([0, 2, 4, 6, 8, 10], ["0%", "2%", "4%", "6%", "8%", "10%"])
    ax.set_xlabel("相对 B200 的单位租赁成本降幅")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#DFE7E5", lw=.8)
    save(fig, "04-cloud-cost-benchmark",
         "数据取自 MLPerf Inference v6.1 与 Nebius 公开报价（2026-11-24）。为情景测算，未计空闲等费用。")


def procurement_frontiers(candidates):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.96, top=.79, bottom=.19, wspace=.15)
    heading(fig, "同样的吞吐目标，批量与在线场景选择不同芯片组合",
            "DeepSeek-R1 · 每秒至少 45 万输出 token · 8 卡服务器整台采购")
    for ax, scenario, label in zip(axes, ("Offline", "Server"), ("批量 Offline", "在线 Server")):
        subset = candidates.loc[candidates.scenario.eq(scenario)]
        x = subset.cost_usd / 10_000
        y = subset.tdp_w / 1000
        efficient = subset.pareto
        ax.scatter(x[~efficient], y[~efficient], s=37, color="#A1AFAD", label="其他达标方案")
        ax.scatter(x[efficient], y[efficient], s=87, color=TEAL, marker="D",
                   zorder=3, label="帕累托最优")
        point = subset.loc[efficient].iloc[0]
        shift = (8, -11) if scenario == "Offline" else (8, 8)
        ax.annotate(f"{point.b200_servers} 台 B200 + {point.b300_servers} 台 B300",
                    (point.cost_usd / 10_000, point.tdp_w / 1000),
                    xytext=shift, textcoords="offset points", va="center", color=TEAL,
                    fontsize=10, weight="bold")
        ax.axhline(76.8, color=ORANGE, linestyle="--", lw=1.2)
        ax.set_title(label, loc="left", fontsize=13, weight="bold")
        ax.set_xlabel("估计芯片购置额（万美元）")
        ax.set_xticks([230, 245, 260, 275])
        ax.set_yticks([75, 80, 85, 90])
        ax.set_xlim(226, 285)
        ax.set_ylim(72, 94)
        grid(ax)
    axes[0].set_ylabel("合计芯片额定功率（kW）")
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)
    save(fig, "05-procurement-pareto",
         "虚线为 76.8 kW 示例额度。仅展示比最低购置额高不超过 40 万美元的达标方案。")


def open_weight_deployment(cases):
    samples = cases.loc[
        cases.gpu.eq("B300") & cases.input_per_output.eq(4)
        & cases.self_host_overhead_uplift.eq(0)
    ]
    base = samples.loc[samples.input_cache_share.eq(0)].set_index("provider")
    fig, (cost, threshold) = plt.subplots(1, 2, figsize=(10, 5.4))
    fig.subplots_adjust(left=.09, right=.97, top=.75, bottom=.19, wspace=.29)
    heading(fig, "同一个开放模型，自部署门槛随 API 报价和缓存变化",
            "gpt-oss-120b · 4 个输入 token / 1 个输出 token · Nebius B300 在线推理")
    usage = np.linspace(.1, 1, 200)
    full = base.self_host_full_load_usd_m_output.iloc[0]
    cost.plot(100 * usage, full / usage, lw=2.2, color=INK, label="自部署 GPU 租赁费")
    for provider, color, mark in [("Fireworks", ORANGE, "s"), ("DeepInfra", TEAL, "D")]:
        rate, point = base.loc[provider, [
            "api_usd_m_output_including_input", "break_even_utilization"
        ]]
        cost.axhline(rate, color=color, linestyle="--", lw=1.2, label=f"{provider} API")
        cost.scatter(100 * point, rate, color=color, marker=mark, s=26, zorder=3)
        cost.annotate(f"{point:.1%}", (100 * point, rate), xytext=(6, 5),
                      textcoords="offset points", color=color, weight="bold")
        here = samples.loc[samples.provider.eq(provider)].sort_values("input_cache_share")
        threshold.plot(100 * here.input_cache_share, 100 * here.break_even_utilization,
                       lw=1.7, marker=mark, color=color, label=provider)
    cost.set_title("a  无缓存时的成本交点", loc="left", fontsize=12, weight="bold")
    cost.set_xlabel("有效利用率（%）")
    cost.set_ylabel("每百万输出 token 成本（美元）")
    cost.set_ylim(0, 1.55)
    cost.set_xlim(10, 100)
    cost.set_xticks([10, 25, 50, 75, 100])
    threshold.set_title("b  输入缓存对自部署门槛的影响", loc="left", fontsize=12, weight="bold")
    threshold.set_xlabel("API 输入缓存命中比例（%）")
    threshold.set_ylabel("自部署保本利用率（%）")
    threshold.set_ylim(0, 100)
    threshold.set_xticks([0, 50, 100])
    cost.legend(loc="upper right", fontsize=8, frameon=False)
    threshold.legend(loc="upper left", fontsize=9, frameon=False)
    for ax in (cost, threshold):
        grid(ax)
    save(fig, "06-open-weight-deployment",
         "依据 MLPerf Inference v6.1 与 2026-11-24 公开标价。仅计 GPU 租赁费，吞吐按基准值固定。")


def capacity_replay(trace, curves, evaluation):
    filter_case = "profile == 'conversation' and provider == 'DeepInfra'"
    series = trace.query(filter_case).copy()
    bills = curves.query(filter_case).copy()
    chosen = evaluation.query(filter_case + " and setting == 'base'").iloc[0]
    span = max(chosen.fixed_servers + 2, 3)
    bills = bills.loc[bills.servers.le(span)]
    time = pd.to_datetime(series.minute, utc=True)
    x = (time - time.iloc[0]).dt.total_seconds() / 3600
    load = series.work_tokens / series.per_server_capacity
    cap = chosen.fixed_servers
    covered = np.minimum(load, cap)
    fig, (arrivals, expense) = plt.subplots(1, 2, figsize=(10, 5.3))
    fig.subplots_adjust(left=.09, right=.97, top=.74, bottom=.19, wspace=.28)
    heading(fig, "把流量高峰和低谷摊开，才能决定固定租多少",
            "Azure 对话请求 · DeepInfra 报价 · 用前段选台数，在后段计算费用")
    arrivals.fill_between(x, 0, covered, step="post", color="#83BBB2", label="租卡接住")
    arrivals.fill_between(x, covered, load, step="post", color="#E0A183", label="高峰按量购买")
    arrivals.step(x, load, where="post", color=INK, lw=.9)
    arrivals.axhline(cap, color=TEAL, lw=1, ls="--")
    arrivals.set_title("a  检验日的流量与固定容量", loc="left", fontsize=12, weight="bold")
    arrivals.set_xlabel("检验日起始后的小时")
    arrivals.set_ylabel("需求折合服务器台数")
    arrivals.legend(loc="upper left", frameon=False, fontsize=8)
    expense.plot(bills.servers, bills.rent_cost, "s--", color=TEAL, lw=1.6, label="固定租赁")
    expense.plot(bills.servers, bills.api_cost, "^--", color=ORANGE, lw=1.6, label="超量 API")
    expense.plot(bills.servers, bills.total_cost, "o-", color=INK, lw=2, label="合计")
    picked = bills.loc[bills.selected].iloc[0]
    expense.scatter(picked.servers, picked.total_cost, s=72,
                    facecolor="none", edgecolor=TEAL, linewidth=1.5, zorder=5)
    lowest = bills.loc[bills.total_cost.idxmin()]
    if lowest.servers != picked.servers:
        expense.annotate("后段最低", (lowest.servers, lowest.total_cost),
                         xytext=(5, -15), textcoords="offset points", fontsize=9, color=INK)
    expense.set_title("b  锁定不同台数的检验期费用", loc="left", fontsize=12, weight="bold")
    expense.set_xlabel("固定租用的 B300 服务器数")
    expense.set_ylabel(f"{chosen.holdout_hours:g} 小时费用（美元）")
    expense.set_xticks(bills.servers.to_list())
    expense.legend(loc="upper center", frameon=False, fontsize=8)
    for ax in (arrivals, expense):
        ax.set_ylim(bottom=0)
        grid(ax)
    save(fig, "07-capacity-replay",
         "来源为 Azure 2024 聚合流量、MLPerf 单机基准和公开费率。虚线为固定容量，空心圈为事先选定台数。")


def capacity_decisions(cases):
    colors = ["#EDF1F0", "#AED4CC", "#3E9289", "#00564F", "#173F57", "#805139"]
    vmax = int(cases.fixed_servers.max())
    palette = ListedColormap(colors[:vmax + 1] if vmax < len(colors)
                             else plt.cm.viridis(np.linspace(.1, .9, vmax + 1)))
    levels = np.arange(-.5, vmax + 1, 1)
    norm = BoundaryNorm(levels, palette.N)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.88, top=.75, bottom=.19, wspace=.15)
    heading(fig, "平均流量相同，合适的固定容量仍可能不同",
            "颜色表示根据前段流量选择的固定台数 · 其余请求按 API 标价付费")
    for ax, profile, title in zip(axes, ("conversation", "code"), ("a  对话流量", "b  代码流量")):
        data = cases.loc[cases.profile.eq(profile)]
        pivot = data.pivot(index="api_price_multiplier", columns="mean_load", values="fixed_servers")
        x, y = pivot.columns.to_numpy(), pivot.index.to_numpy()
        heat = ax.pcolormesh(x, y, pivot.to_numpy(), cmap=palette, norm=norm, shading="nearest")
        ax.scatter(1, 1, s=54, marker="o", facecolor="none", edgecolor=ORANGE, linewidth=1.5)
        ax.scatter(.3, 1, s=40, marker="D", facecolor="none", edgecolor=ORANGE, linewidth=1.2)
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.set_xlabel("平均需求折合服务器台数")
        ax.set_xticks([.5, 1, 1.5, 2])
    axes[0].set_ylabel("API 价格相对 DeepInfra 的倍数")
    bar = fig.colorbar(heat, cax=fig.add_axes([.90, .21, .023, .51]), ticks=np.arange(vmax + 1))
    bar.set_label("固定租用台数")
    save(fig, "08-capacity-decisions",
         "空心圈为平均需求相当于一台的基准情景。菱形为相当于 0.3 台的对照。两者均使用实际 DeepInfra 报价。")


def shipments_by_precision(chips, quarterly):
    order = ["H100", "B200", "B300", "R200"]
    chips = chips.set_index("sku").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.3), sharey=True)
    fig.subplots_adjust(left=.09, right=.97, top=.75, bottom=.18, wspace=.14)
    heading(fig, "R200 推高了两种算力，只有 8 位增幅超过支出",
            "2026 Q3 相对 Q2 的出货变化 · 芯片购置额同期增长 19.6%")
    spend = quarterly.set_index("quarter").spend_usd
    spend_growth = spend.loc["2026Q3"] / spend.loc["2026Q2"] - 1
    for ax, metric, name in zip(axes, ("h100e", "h100e16"), ("密集 8 位", "密集 16 位")):
        change = chips[f"delta_{metric}"].to_numpy() / 1e6
        previous = chips[f"{metric}_q2"].sum()
        growth = change.sum() * 1e6 / previous
        required = previous * spend_growth / 1e6
        running = np.r_[0, np.cumsum(change)]
        ax.bar(range(5), [*change, running[-1]], bottom=[*running[:-1], 0],
               color=[TEAL, GRAY, TEAL, ORANGE, INK], width=.57)
        for index, end in enumerate(running[1:]):
            ax.hlines(end, index + .285, index + .715, color="#859794", ls="dashed", lw=.9)
        for index, value in enumerate([*change, running[-1]]):
            y = running[index + 1] if index < 4 else running[-1]
            inside = value > .4 and 0 < required - y < .17
            ax.annotate(f"{value:+.2f}", (index, y),
                        xytext=(0, -5 if inside else 4 if value >= 0 else -6),
                        textcoords="offset points", ha="center",
                        va="bottom" if value >= 0 and not inside else "top",
                        color="white" if inside else INK, fontsize=9)
        ax.axhline(required, color="#645074", ls=":", lw=1.4)
        ax.annotate("单位成本不变线", (0, required), xytext=(-13, 4),
                    textcoords="offset points", color="#645074", fontsize=9)
        ax.set_title(f"{name} · 环比 {growth:+.1%}", loc="left", fontsize=13, weight="bold")
        ax.set_xticks(range(5), ["H100 组", "B200", "B300", "R200", "净增量"])
        ax.set_ylim(-.36, 1.62)
        ax.set_yticks([0, .4, .8, 1.2, 1.6])
        ax.axhline(0, lw=.8, color=GRAY)
        grid(ax)
    axes[0].set_ylabel("净增量（百万 H100 等效）")
    save(fig, "09-shipment-growth-by-precision",
         "紫色点线表示要抵消购置额涨幅所需的算力增量。数据取自 Epoch AI。标注值四舍五入。")


if __name__ == "__main__":
    quarterly = read("quarterly")
    trend(quarterly)
    waterfall(read("decomposition").query("comparison == 'blackwell_transition'").iloc[0])
    precision(quarterly)
    if (ROOT / "results" / "benchmark_cloud_cost_scenarios.csv").exists():
        benchmark_costs(read("benchmark_cloud_cost_scenarios"))
    if (ROOT / "results" / "procurement_figure_candidates.csv").exists():
        procurement_frontiers(read("procurement_figure_candidates"))
    if (ROOT / "results" / "effective_compute_api_break_even.csv").exists():
        open_weight_deployment(read("effective_compute_api_break_even"))
    if (ROOT / "results" / "capacity_policy_evaluation.csv").exists():
        capacity_replay(read("capacity_trace_day"), read("capacity_cost_curves"),
                        read("capacity_policy_evaluation"))
        capacity_decisions(read("capacity_policy_grid"))
    if (ROOT / "results" / "effective_compute_q3_chip_growth.csv").exists():
        shipments_by_precision(read("effective_compute_q3_chip_growth"), quarterly)
