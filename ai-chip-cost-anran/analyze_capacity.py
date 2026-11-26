"""Replay Azure traffic shapes to size fixed inference capacity and API overflow."""
import argparse
from hashlib import sha256
import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "results"
TRACE = DATA / "azure_minute_profiles.csv.gz"
SOURCE_NAMES = {"conversation": "conv", "code": "code"}
PREFIX_HOURS = {"conversation": 36, "code": 8}
MODEL = "gpt-oss-120b"
MAIN_MINUTES, MAIN_LOAD = 5, 1.0
REFERENCE_INPUT_RATIO, BASE_INPUT_WEIGHT = 4.0, .05


def snapshot_trace(source, hours):
    counts = dict(raw_rows=0, invalid_timestamp=0, invalid_tokens=0, zero_output=0)
    frames = []
    digest = sha256()
    observed_after_cutoff = False
    first = last = None
    for chunk in pd.read_csv(source, chunksize=200_000):
        stamp = pd.to_datetime(chunk.TIMESTAMP, format="mixed", errors="coerce", utc=True)
        times = stamp.dropna()
        assert times.is_monotonic_increasing and (last is None or times.iloc[0] >= last)
        last = times.iloc[-1]
        if first is None:
            first = times.iloc[0]
            start = first.floor("D")
            stop = start + pd.Timedelta(hours=hours)
        # Use only the timestamp-bounded prefix, which is also reproducible from the full release.
        observed_after_cutoff = last >= stop
        if observed_after_cutoff:
            boundary = np.flatnonzero(stamp.ge(stop))[0]
            chunk, stamp = chunk.iloc[:boundary], stamp.iloc[:boundary]
        counts["raw_rows"] += len(chunk)
        digest.update(chunk.to_csv(index=False, header=False, lineterminator="\n").encode())
        prompt = pd.to_numeric(chunk.ContextTokens, errors="coerce")
        output = pd.to_numeric(chunk.GeneratedTokens, errors="coerce")
        tokens_ok = np.isfinite(prompt) & np.isfinite(output) & prompt.ge(0) & output.ge(0)
        tokens_ok &= prompt.mod(1).eq(0) & output.mod(1).eq(0)
        valid = stamp.notna() & tokens_ok
        counts["invalid_timestamp"] += int(stamp.isna().sum())
        counts["invalid_tokens"] += int((stamp.notna() & ~tokens_ok).sum())
        counts["zero_output"] += int((valid & output.eq(0)).sum())
        retained = valid & output.gt(0)
        minute = stamp.loc[retained].dt.floor("min")
        frame = pd.DataFrame({"minute": minute, "requests": 1,
                              "input_tokens": prompt.loc[retained].astype("int64"),
                              "output_tokens": output.loc[retained].astype("int64")})
        frames.append(frame.groupby("minute").sum())
        if observed_after_cutoff:
            break
    assert observed_after_cutoff and first.floor("min") == start
    complete = pd.date_range(start, stop, freq="min", inclusive="left")
    actual = pd.concat(frames).groupby(level=0).sum()
    result = actual.reindex(complete, fill_value=0).rename_axis("minute")
    counts["source"] = source.name
    counts["canonical_prefix_sha256"] = digest.hexdigest()
    counts["first_timestamp"] = first.isoformat()
    counts["window_start"] = start.isoformat()
    counts["window_end_exclusive"] = stop.isoformat()
    counts["retained_requests"] = int(result.requests.sum())
    counts["edge_requests_excluded"] = int(actual.requests.sum() - result.requests.sum())
    counts["zero_traffic_minutes"] = int(result.output_tokens.eq(0).sum())
    counts["minutes_retained"] = len(result)
    counts["release"] = "2024"
    counts["retained_input_tokens"] = int(result.input_tokens.sum())
    counts["retained_output_tokens"] = int(result.output_tokens.sum())
    assert len(result) >= 30
    return result.reset_index(), counts


def build_snapshot(source_dir):
    audits, profiles = {}, []
    for label, short in SOURCE_NAMES.items():
        file = source_dir / f"AzureLLMInferenceTrace_{short}_1week.csv"
        profile, audits[label] = snapshot_trace(file, PREFIX_HOURS[label])
        profiles.append(profile.assign(profile=label))
    pd.concat(profiles, ignore_index=True).to_csv(
        TRACE, index=False, compression={"method": "gzip", "mtime": 1}
    )
    (DATA / "azure_trace_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def time_blocks(profile, minutes):
    by_time = profile.set_index("minute")[["requests", "input_tokens", "output_tokens"]]
    binned = by_time.resample(f"{minutes}min").sum()
    days = binned.index.normalize().unique()
    if len(days) > 1:
        split = days[len(days) // 2]
    else:
        mid = by_time.index[len(by_time) // 2]
        split = mid.floor("15min")
    return binned.loc[binned.index < split], binned.loc[binned.index >= split]


def policy_costs(daily, per_server, hourly_rental, minutes, tariff, max_servers,
                 cache_share=0.0, input_weight=BASE_INPUT_WEIGHT, billing_input_ratio=None):
    volume = daily.output_tokens.to_numpy(float)
    tokens_in = daily.input_tokens.to_numpy(float)
    price_in = (1 - cache_share) * tariff.uncached_input_usd_per_m
    price_in += cache_share * tariff.cached_input_usd_per_m
    billable_input = tokens_in if billing_input_ratio is None else volume * billing_input_ratio
    api_all = (volume * tariff.output_usd_per_m + billable_input * price_in) / 1e6
    counts = np.arange(max_servers + 1)
    work = volume + input_weight * tokens_in
    spill = np.maximum(work[:, None] - per_server * counts, 0)
    share = np.divide(spill, work[:, None], where=work[:, None] > 0,
                      out=np.zeros_like(spill))
    return pd.DataFrame({
        "servers": counts,
        "rent_cost": counts * hourly_rental * len(daily) * minutes / 60,
        "api_cost": api_all @ share,
        "spill_fraction": volume @ share / volume.sum(),
    }).assign(total_cost=lambda row: row.rent_cost + row.api_cost)


def evaluate(early, late, q, hourly, minutes, tariff, load=MAIN_LOAD, price_factor=1,
             capacity_factor=1, cache_share=0, input_weight=BASE_INPUT_WEIGHT,
             billing_input_ratio=None):
    reference_work = q * minutes * 60 * (1 + BASE_INPUT_WEIGHT * REFERENCE_INPUT_RATIO)
    raw_baseline_work = early.output_tokens + BASE_INPUT_WEIGHT * early.input_tokens
    demand_scale = load * reference_work / raw_baseline_work.mean()
    early, late = early * demand_scale, late * demand_scale
    capacity = q * minutes * 60 * capacity_factor * (1 + input_weight * REFERENCE_INPUT_RATIO)
    demand = lambda frame: frame.output_tokens + input_weight * frame.input_tokens
    ceiling = ceil(max(demand(early).max(), demand(late).max()) / capacity)
    scaled_tariff = tariff * price_factor
    def costs(frame):
        return policy_costs(frame, capacity, hourly, minutes, scaled_tariff, ceiling,
                            cache_share, input_weight, billing_input_ratio)

    train, test = costs(early), costs(late)
    choice = train.loc[train.total_cost.idxmin()]
    selected = test.loc[int(choice.servers)]
    api_only = test.iloc[0]
    rent_only = test.loc[test.spill_fraction.eq(0)].iloc[0]
    # The flat control keeps output and input totals identical within each split.
    flat = early.copy()
    flat["input_tokens"] = early.input_tokens.mean()
    flat["output_tokens"] = early.output_tokens.mean()
    flat_train = costs(flat)
    flat_choice = int(flat_train.loc[flat_train.total_cost.idxmin(), "servers"])
    best_test = test.loc[test.total_cost.idxmin()]
    row = {
        "calibration_days": early.index.normalize().nunique(),
        "holdout_days": late.index.normalize().nunique(),
        "calibration_hours": len(early) * minutes / 60,
        "holdout_hours": len(late) * minutes / 60,
        "holdout_start": late.index.min().isoformat(),
        "holdout_output_tokens": late.output_tokens.sum(),
        "holdout_input_per_output": late.input_tokens.sum() / late.output_tokens.sum(),
        "holdout_peak_to_mean": demand(late).max() / demand(late).mean(),
        "fixed_servers": int(choice.servers),
        "holdout_cost": selected.total_cost,
        "holdout_spill_fraction": selected.spill_fraction,
        "api_only_cost": api_only.total_cost,
        "peak_rent_servers": int(rent_only.servers),
        "peak_rent_cost": rent_only.total_cost,
        "flat_control_servers": flat_choice,
        "flat_control_replay_cost": float(test.loc[flat_choice, "total_cost"]),
        "hindsight_servers": int(best_test.servers),
        "hindsight_cost": best_test.total_cost,
        "holdout_saving_vs_api": 1 - selected.total_cost / api_only.total_cost,
        "holdout_saving_vs_peak_rent": 1 - selected.total_cost / rent_only.total_cost,
    }
    return row, train, test, late.assign(work_tokens=demand(late)), capacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Rebuild the minute snapshot from official CSVs")
    args = parser.parse_args()
    if args.source_dir:
        build_snapshot(args.source_dir)
    minute = pd.read_csv(TRACE, parse_dates=["minute"])
    pairs = pd.read_csv(OUT / "mlperf_b200_b300_pairs.csv")
    reference = pairs.query("Organization == 'Nebius' and Model == @MODEL and Scenario == 'Server'")
    q = reference.Result_upgrade.item()
    prices = pd.read_csv(DATA / "hgx_on_demand_price_snapshot.csv")
    rate = 8 * prices.query("provider == 'Nebius' and gpu == 'B300'").usd_per_gpu_hour.item()
    tariffs = pd.read_csv(DATA / "gpt_oss_api_price_snapshot.csv").set_index("provider").filter(like="_usd_per_m")
    evaluations, curve, example, grid = [], [], [], []
    sensitivities = [
        ("base", MAIN_MINUTES, 1, 0, BASE_INPUT_WEIGHT),
        ("1 minute", 1, 1, 0, BASE_INPUT_WEIGHT),
        ("15 minutes", 15, 1, 0, BASE_INPUT_WEIGHT),
        ("throughput -20%", MAIN_MINUTES, .8, 0, BASE_INPUT_WEIGHT),
        ("API input cache 50%", MAIN_MINUTES, 1, .5, BASE_INPUT_WEIGHT),
        ("ignore input work", MAIN_MINUTES, 1, 0, 0),
        ("input work weight .2", MAIN_MINUTES, 1, 0, .2),
    ]
    for label, group in minute.groupby("profile"):
        for setting, minutes, capacity_factor, cache_share, input_weight in sensitivities:
            early, late = time_blocks(group, minutes)
            for provider, tariff in tariffs.iterrows():
                row, train, test, traffic, capacity = evaluate(
                    early, late, q, rate, minutes, tariff, capacity_factor=capacity_factor,
                    cache_share=cache_share, input_weight=input_weight
                )
                evaluations.append({"profile": label, "provider": provider,
                                    "setting": setting, **row})
                if setting == "base":
                    curve.append(test.assign(profile=label, provider=provider,
                                             selected=lambda values: values.servers.eq(row["fixed_servers"])))
                    # Use the later series' midpoint day by date, independent of the cost outcome.
                    days = traffic.index.normalize().unique()
                    day = traffic.loc[traffic.index.normalize() == days[len(days) // 2]].copy()
                    day["profile"], day["provider"] = label, provider
                    day["capacity"] = capacity * row["fixed_servers"]
                    day["per_server_capacity"] = capacity
                    example.append(day.reset_index())
        early, late = time_blocks(group, MAIN_MINUTES)
        for load in np.round(np.arange(.3, 2.05, .1), 2):
            for factor in np.round(np.arange(.5, 4.05, .1), 2):
                row, _, _, _, _ = evaluate(early, late, q, rate, MAIN_MINUTES, tariffs.loc["DeepInfra"],
                                            load=load, price_factor=factor)
                grid.append({"profile": label, "mean_load": load, "api_price_multiplier": factor, **row})

    control, _ = time_blocks(minute.loc[minute.profile.eq("conversation")], MAIN_MINUTES)
    conversation_ratio = control.input_tokens.sum() / control.output_tokens.sum()
    early, late = time_blocks(minute.loc[minute.profile.eq("code")], MAIN_MINUTES)
    for label, ratio in [("code actual billable ratio", None),
                         ("code at conversation billable ratio", conversation_ratio)]:
        row, _, _, _, _ = evaluate(early, late, q, rate, MAIN_MINUTES, tariffs.loc["DeepInfra"],
                                    load=.3, billing_input_ratio=ratio)
        evaluations.append({"profile": "code", "provider": "DeepInfra", "setting": label, **row})

    for name, frame in {
        "capacity_policy_evaluation": pd.DataFrame(evaluations),
        "capacity_cost_curves": pd.concat(curve, ignore_index=True),
        "capacity_trace_day": pd.concat(example, ignore_index=True),
        "capacity_policy_grid": pd.DataFrame(grid),
    }.items():
        frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(evaluations).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
