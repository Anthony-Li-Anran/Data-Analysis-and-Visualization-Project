"""Relate generation gains to service constraints and open-weight deployment cost."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
DATA = ROOT / "data"


def revenue_and_spec_paths():
    q = pd.read_csv(OUT / "quarterly.csv").set_index("quarter")
    windows = [
        ("2022Q1", "2026Q3"), ("2023Q1", "2026Q3"),
        ("2024Q1", "2026Q3"), ("2022Q1", "2026Q2"),
        ("2026Q2", "2026Q3"),
    ]
    results = []
    for start, end in windows:
        ratio = q.loc[end] / q.loc[start]
        assert np.isclose(ratio.spend_usd, ratio.h100e * ratio.usd_per_h100e)
        assert np.isclose(ratio.h100e, ratio.units * ratio.h100e_per_chip)
        assert np.isclose(ratio.tdp_w, ratio.h100e * ratio.tdp_w_per_h100e)
        results.append({
            "first_quarter": start,
            "last_quarter": end,
            "spend_ratio": ratio.spend_usd,
            "h100e_ratio": ratio.h100e,
            "chip_count_ratio": ratio.units,
            "spec_per_chip_ratio": ratio.h100e_per_chip,
            "tdp_ratio": ratio.tdp_w,
            "tdp_per_h100e_ratio": ratio.tdp_w_per_h100e,
            "unit_cost_ratio_8_bit": ratio.usd_per_h100e,
            "unit_cost_ratio_16_bit": ratio.usd_per_h100e16,
            "log_spend_change": np.log(ratio.spend_usd),
            "log_capacity_change": np.log(ratio.h100e),
            "log_unit_cost_change": np.log(ratio.usd_per_h100e),
        })
    return pd.DataFrame(results)


def compare_quarterly_supply():
    chips = pd.read_csv(OUT / "nvidia_clean.csv")
    totals = pd.read_csv(OUT / "quarterly.csv").set_index("quarter")
    periods = ["2026Q2", "2026Q3"]
    measures = ["units", "h100e", "h100e16", "tdp_w"]
    observed = chips.loc[chips.quarter.isin(periods)]
    before, after = (
        observed.loc[observed.quarter.eq(quarter)].groupby("sku")[measures].sum()
        for quarter in periods
    )
    # Only the observed quarterly sample is zero-filled for SKUs absent in one quarter.
    # This assignment does not establish zero shipments outside the sample.
    before, after = before.align(after, join="outer", axis=0, fill_value=0)
    delta = after - before
    expected = totals.loc[periods[1], measures] - totals.loc[periods[0], measures]
    assert np.allclose(delta.sum(), expected)
    comparison = before.add_suffix("_q2").join(after.add_suffix("_q3")).join(delta.add_prefix("delta_"))
    comparison["share_q3_chip_units"] = after.units / after.units.sum()
    comparison["share_net_8bit_growth"] = delta.h100e / delta.h100e.sum()
    comparison["share_net_16bit_growth"] = delta.h100e16 / delta.h100e16.sum()
    return comparison.reset_index()


def compare_workloads():
    paired = pd.read_csv(OUT / "mlperf_b200_b300_pairs.csv")
    assumptions = pd.read_csv(OUT / "chip_specs_and_thresholds.csv").set_index("sku")
    quote = pd.read_csv(DATA / "hgx_on_demand_price_snapshot.csv")
    rate = quote.loc[quote.provider.eq("Nebius")].set_index("gpu").usd_per_gpu_hour
    result = paired[["Organization", "Model", "Scenario", "Result_base", "Result_upgrade",
                     "csv_row_base", "csv_row_upgrade", "throughput_ratio"]].copy()
    result["purchase_premium"] = assumptions.at["B300", "price"] / assumptions.at["B200", "price"]
    result["rated_power_premium"] = assumptions.at["B300", "spec_tdp"] / assumptions.at["B200", "spec_tdp"]
    result["passes_both_purchase_and_rated_power"] = (
        result.throughput_ratio >= result[["purchase_premium", "rated_power_premium"]].max(axis=1)
    )
    nebius = result.Organization.eq("Nebius")
    result.loc[nebius, "same_provider_rental_premium"] = rate["B300"] / rate["B200"]
    gains = result.pivot(index=["Organization", "Model"], columns="Scenario", values="throughput_ratio")
    gains["server_gain_pct_points_less_than_offline"] = 100 * (gains.Offline - gains.Server)
    return result, gains.reset_index()


def api_utilization_break_even(results):
    bench = results.loc[
        results.Organization.eq("Nebius") & results.Model.eq("gpt-oss-120b")
        & results.Scenario.eq("Server")
    ].iloc[0]
    prices = pd.read_csv(DATA / "gpt_oss_api_price_snapshot.csv")
    rates = pd.read_csv(DATA / "hgx_on_demand_price_snapshot.csv")
    rates = rates.loc[rates.provider.eq("Nebius")].set_index("gpu").usd_per_gpu_hour
    assert prices.model.eq("gpt-oss-120b").all() and not prices.provider.duplicated().any()
    assert prices.filter(like="_usd_per_m").gt(0).all().all() and rates.gt(0).all()
    scenario = pd.MultiIndex.from_product(
        [("B200", "B300"), (0, 1, 4, 8), (0.0, 0.5, 1.0), (0.0, 0.25)],
        names=["gpu", "input_per_output", "input_cache_share", "self_host_overhead_uplift"],
    ).to_frame(index=False)
    scenario["system_tokens_s"] = scenario.gpu.map({"B200": bench.Result_base,
                                                   "B300": bench.Result_upgrade})
    scenario["system_usd_h"] = 8 * scenario.gpu.map(rates)
    scenario["fixed_cost_usd_h"] = scenario.system_usd_h * (1 + scenario.self_host_overhead_uplift)
    scenario["self_host_full_load_usd_m_output"] = (
        scenario.fixed_cost_usd_h * 1_000_000 / (3600 * scenario.system_tokens_s)
    )
    comparison = prices.merge(scenario, how="cross")
    blend = (
        comparison.uncached_input_usd_per_m * (1 - comparison.input_cache_share)
        + comparison.cached_input_usd_per_m * comparison.input_cache_share
    )
    comparison["api_usd_m_output_including_input"] = (
        comparison.output_usd_per_m + comparison.input_per_output * blend
    )
    comparison["break_even_utilization"] = (
        comparison.self_host_full_load_usd_m_output / comparison.api_usd_m_output_including_input
    )
    comparison["effective_billed_output_tokens_s_to_break_even"] = (
        comparison.break_even_utilization * comparison.system_tokens_s
    )
    comparison["break_even_possible_within_benchmark_capacity"] = comparison.break_even_utilization <= 1
    return comparison


if __name__ == "__main__":
    paths = revenue_and_spec_paths()
    supply = compare_quarterly_supply()
    benchmark, comparisons = compare_workloads()
    costs = api_utilization_break_even(benchmark)
    for name, data in {
        "effective_compute_quarterly_windows": paths,
        "effective_compute_q3_chip_growth": supply,
        "effective_compute_benchmark_thresholds": benchmark,
        "effective_compute_scenario_gains": comparisons,
        "effective_compute_api_break_even": costs,
    }.items():
        data.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(paths.round(4).to_string(index=False))
    print(comparisons.round(4).to_string(index=False))
    print(costs.query("gpu == 'B300' and input_per_output in [0, 1, 4] and input_cache_share == 0")[
        ["provider", "input_per_output", "self_host_overhead_uplift", "break_even_utilization",
         "effective_billed_output_tokens_s_to_break_even"]
    ].round(4).to_string(index=False))
