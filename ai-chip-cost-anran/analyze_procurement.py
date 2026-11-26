"""Enumerate cost/power Pareto frontiers for benchmark-based 8-GPU replicas."""
from dataclasses import dataclass, fields
from decimal import Decimal
import csv
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
PRIMARY_TARGET = 450_000
TARGETS = range(400_000, 500_001, 5_000)
THROUGHPUT_FACTORS = (Decimal("0.95"), Decimal("1.00"), Decimal("1.05"))
GPUS_PER_SERVER = 8


@dataclass(frozen=True)
class Resources:
    cost_usd: int
    tdp_w: int


@dataclass(frozen=True)
class Plan:
    b200_servers: int
    b300_servers: int
    cost_usd: int
    tdp_w: int
    throughput: Decimal


def read_csv(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def possible_plans(resources, throughputs, max_target):
    # Any plan above this bound can remove a whole server and still meet max_target.
    max_servers = ceil(Decimal(max_target) / min(throughputs))
    b200, b300 = resources
    t200, t300 = throughputs
    return [
        Plan(a, b, a * b200.cost_usd + b * b300.cost_usd,
             a * b200.tdp_w + b * b300.tdp_w, a * t200 + b * t300)
        for a in range(max_servers + 1)
        for b in range(max_servers + 1 - a)
    ]


def frontier(plans, target):
    """Retain every equal-resource tie, plus all strictly nondominated points."""
    feasible = sorted((p for p in plans if p.throughput >= target),
                      key=lambda p: (p.cost_usd, p.tdp_w))
    efficient = []
    for plan in feasible:
        if not efficient or plan.tdp_w < efficient[-1].tdp_w:
            efficient.append(plan)
        elif (plan.cost_usd, plan.tdp_w) == (efficient[-1].cost_usd, efficient[-1].tdp_w):
            efficient.append(plan)
    return feasible, efficient


def write_csv(name, records, context_fields):
    columns = [field.name for field in fields(Plan)]
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*context_fields, *columns])
        for context, plan in records:
            writer.writerow([*context, *(getattr(plan, col) for col in columns)])


def main():
    specs = {row["sku"]: row for row in read_csv("chip_specs_and_thresholds.csv")}
    resources = tuple(
        Resources(GPUS_PER_SERVER * int(Decimal(specs[sku]["price"])),
                  GPUS_PER_SERVER * int(Decimal(specs[sku]["spec_tdp"])))
        for sku in ("B200", "B300")
    )
    benchmark = {
        row["Scenario"]: row for row in read_csv("mlperf_b200_b300_pairs.csv")
        if row["Organization"] == "Nebius" and row["Model"] == "deepseek-r1"
    }
    assert set(benchmark) == {"Offline", "Server"}
    all_frontiers, figure_candidates = [], []
    for scenario in ("Offline", "Server"):
        rates = tuple(Decimal(benchmark[scenario][col]) for col in ("Result_base", "Result_upgrade"))
        for factor in THROUGHPUT_FACTORS:
            plans = possible_plans(resources, (rates[0], rates[1] * factor), max(TARGETS))
            targets = TARGETS if factor == 1 else [PRIMARY_TARGET]
            for target in targets:
                feasible, efficient = frontier(plans, target)
                all_frontiers.extend(((scenario, factor, target), plan) for plan in efficient)
                if target == PRIMARY_TARGET and factor == 1:
                    for plan in feasible:
                        if plan.cost_usd <= min(p.cost_usd for p in efficient) + 400_000:
                            figure_candidates.append(((scenario, plan in efficient), plan))

    write_csv("procurement_pareto.csv", all_frontiers,
              ["scenario", "b300_throughput_factor", "target_tokens_s"])
    write_csv("procurement_figure_candidates.csv", figure_candidates, ["scenario", "pareto"])
    for (scenario, factor, target), plan in all_frontiers:
        if target == PRIMARY_TARGET:
            print(scenario, factor, target, plan)


if __name__ == "__main__":
    main()
