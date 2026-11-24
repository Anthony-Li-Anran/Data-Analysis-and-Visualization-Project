"""Compare paired MLPerf system submissions and apply public hourly price scenarios."""
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
BENCHMARK = DATA / "mlperf_inference_v6_1_summary.csv"
PRICES = DATA / "hgx_on_demand_price_snapshot.csv"
KEY = ["Organization", "Model", "MlperfModel", "Scenario", "Units", "weight_data_types"]
DETAILS = ["csv_row", "gpu", "Result", "Availability", "SystemName",
           "accelerator_model_name", "Location", "host_processor_model_name",
           "framework", "inferred", "has_power"]


def pair(rows, base, upgrade):
    selected = rows.loc[rows.gpu.isin([base, upgrade]), KEY + DETAILS]
    assert not selected.duplicated(KEY + ["gpu"]).any()
    joined = selected.loc[selected.gpu.eq(base)].merge(
        selected.loc[selected.gpu.eq(upgrade)], on=KEY, how="outer",
        validate="one_to_one", suffixes=("_base", "_upgrade"), indicator=True,
    )
    return joined


def main():
    raw = pd.read_csv(BENCHMARK)
    raw.insert(0, "csv_row", raw.index + 2)
    gpu = raw.accelerator_model_name.str.upper().str.extract(
        r"\b(B200|B300|GB300|VERA RUBIN)\b", expand=False
    )
    raw["gpu"] = gpu
    eligible = (
        raw.version.eq("v6.1") & raw.Division.eq("closed") & raw.SystemType.eq("datacenter")
        & raw.errors.eq(0) & raw.Units.eq("Tokens/s") & raw.weight_data_types.eq("fp4")
        & raw.Scenario.isin(["Offline", "Server"]) & raw.Result.gt(0)
    )
    scope = (
        raw.Organization.isin(["Nebius", "CoreWeave"]) & raw["number_of_nodes"].eq(1)
        & raw["total_accelerators"].eq(8) & raw.Availability.eq("available")
        & raw.gpu.isin(["B200", "B300"])
    )
    candidate = raw.loc[scope]
    rows = candidate.loc[eligible]
    pairs = pair(rows, "B200", "B300")
    paired = pairs.loc[pairs._merge.eq("both")].copy()
    paired["benchmark_id"] = paired.Organization + "/" + paired.Model + "/" + paired.Scenario
    assert not paired.benchmark_id.duplicated().any()
    paired["throughput_ratio"] = paired.Result_upgrade / paired.Result_base
    paired["tokens_s_per_gpu_base"] = paired.Result_base / 8
    paired["tokens_s_per_gpu_upgrade"] = paired.Result_upgrade / 8
    # Apply quoted rates only to the benchmark platform they describe; use Runpod as a separate sensitivity.
    quotes = pd.read_csv(PRICES)
    assert quotes.usd_per_gpu_hour.gt(0).all()
    assert not quotes.duplicated(["provider", "gpu"]).any()
    price_pairs = quotes.pivot(index="provider", columns="gpu", values="usd_per_gpu_hour")
    assert price_pairs.notna().all().all()
    nebius = paired.loc[paired.Organization.eq("Nebius"), [
        "benchmark_id", "Model", "Scenario", "tokens_s_per_gpu_base",
        "tokens_s_per_gpu_upgrade", "throughput_ratio",
    ]]
    costs = nebius.merge(price_pairs.reset_index(), how="cross")
    costs["price_basis"] = costs.provider.map({"Nebius": "same provider public price scenario",
                                             "Runpod": "cross-provider price sensitivity only"})
    costs["required_throughput_ratio"] = costs.B300 / costs.B200
    costs["cost_per_m_output_tokens_b200"] = costs.B200 * 1e6 / (3600 * costs.tokens_s_per_gpu_base)
    costs["cost_per_m_output_tokens_b300"] = costs.B300 * 1e6 / (3600 * costs.tokens_s_per_gpu_upgrade)
    costs["cost_ratio_b300_to_b200"] = (
        costs.cost_per_m_output_tokens_b300 / costs.cost_per_m_output_tokens_b200
    )
    assert (
        costs.cost_ratio_b300_to_b200 - costs.required_throughput_ratio / costs.throughput_ratio
    ).abs().max() < 1e-12

    preview_scope = (
        raw.Organization.eq("NVIDIA") & raw["total_accelerators"].eq(72)
        & raw.Model.eq("deepseek-r1") & raw.Division.eq("closed") & raw.version.eq("v6.1")
        & raw.SystemType.eq("datacenter") & raw.errors.eq(0)
        & raw.Units.eq("Tokens/s") & raw.weight_data_types.eq("fp4") & raw.Result.gt(0)
        & ((raw.gpu.eq("GB300") & raw.Availability.eq("available"))
           | (raw.gpu.eq("VERA RUBIN") & raw.Availability.eq("preview")))
    )
    preview = pair(raw.loc[preview_scope], "GB300", "VERA RUBIN")
    assert preview._merge.eq("both").all() and len(preview) == 3
    preview["throughput_ratio"] = preview.Result_upgrade / preview.Result_base

    audit = {
        "input_sha256": {f.name: sha256(f.read_bytes()).hexdigest() for f in [BENCHMARK, PRICES]},
        "mlperf_source_rows": len(raw),
        "matching_gpu_submitter_topology_availability_rows": len(candidate),
        "passing_comparability_filters_rows": len(rows),
        "paired_comparisons": len(paired),
        "paired_comparisons_by_submitter": paired.Organization.value_counts().to_dict(),
        "unmatched_model_scenario_keys": int(pairs._merge.ne("both").sum()),
        "price_snapshot_rows": len(quotes),
        "paired_system_side_has_power_rows": sum(
            int(paired[col].eq(True).sum()) for col in ["has_power_base", "has_power_upgrade"]
        ),
        "separate_nvidia_rubin_preview_comparisons": len(preview),
    }
    OUT.mkdir(exist_ok=True)
    for filename, frame in {
        "mlperf_b200_b300_pair_audit": pairs,
        "mlperf_b200_b300_pairs": paired,
        "mlperf_rubin_preview_pairs": preview,
        "benchmark_cloud_cost_scenarios": costs,
    }.items():
        frame.to_csv(OUT / f"{filename}.csv", index=False, encoding="utf-8-sig")
    (OUT / "benchmark_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    print(paired[["Organization", "Model", "Scenario", "throughput_ratio"]].round(4).to_string(index=False))
    print(costs[["Model", "Scenario", "provider", "cost_per_m_output_tokens_b200",
                 "cost_per_m_output_tokens_b300", "cost_ratio_b300_to_b200"]].round(5).to_string(index=False))
    print(preview[["Model", "Scenario", "Result_base", "Result_upgrade",
                   "throughput_ratio"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
