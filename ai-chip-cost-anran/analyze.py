"""Rebuild the chip-level audit, quarterly metrics, and procurement scenarios."""
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
TIMELINE = DATA / "ai_chip_sales_timelines_by_chip.csv"
SPECS = DATA / "ai_chip_sales_chip_types.csv"
FIRST, LAST = "2022Q1", "2026Q3"


def read_inputs():
    sales = pd.read_csv(TIMELINE).rename(columns={
        "Name": "source_record", "Chip manufacturer": "designer", "Start date": "start",
        "End date": "end", "Incomplete": "incomplete", "Number of Units": "units",
        "Chip type": "bucket", "Chip type (linked)": "sku", "Chip Cost (USD)": "price",
        "Chip TDP (W)": "source_tdp", "Total TDP (W)": "source_total_tdp",
        "Cost Estimate (USD)": "source_spend", "Compute estimate in H100e (median)": "source_h100e",
        "Notes": "source_notes",
    })
    specs = pd.read_csv(SPECS).rename(columns={
        "Name": "sku", "Designer": "spec_designer", "H100e": "h100e_per_chip",
        "16-bit FLOP/s": "flops16", "8-bit OP/s": "ops8", "Memory": "memory_bytes",
        "Memory bandwidth": "bandwidth_bytes_s",
        "TDP (W) (from ML Hardware (linked))": "spec_tdp",
    })
    sales.insert(0, "csv_row", sales.index + 2)
    for frame, cols in [(sales, ["sku", "bucket", "designer"]), (specs, ["sku", "spec_designer"])]:
        for col in cols:
            frame[col] = frame[col].str.strip()
    for col in ("start", "end"):
        sales[col] = pd.to_datetime(sales[col], format="%Y-%m-%d", errors="raise")
    sales["quarter"] = sales.start.dt.to_period("Q")
    flags = sales.incomplete.fillna("").astype(str).str.strip().str.lower()
    assert set(flags) <= {"", "true", "false"}
    sales["incomplete"] = flags.eq("true")
    assert sales[["designer", "sku", "start", "end"]].notna().all().all()
    assert not sales.duplicated(["designer", "quarter", "sku"]).any()
    assert not specs.sku.duplicated().any()
    calendar_match = sales.start.eq(sales.quarter.dt.start_time) & sales.end.eq(
        sales.quarter.dt.end_time.dt.normalize()
    )
    assert calendar_match.all()
    joined = sales.merge(
        specs[["sku", "spec_designer", "h100e_per_chip", "flops16", "ops8",
               "memory_bytes", "bandwidth_bytes_s", "spec_tdp"]],
        on="sku", how="left", validate="many_to_one", indicator=True,
    )
    assert len(joined) == len(sales)
    joined["status"] = np.select(
        [joined.incomplete, ~joined.quarter.between(FIRST, LAST), joined.designer.ne("Nvidia")],
        ["source_incomplete", "outside_window", "outside_nvidia_sample"],
        default="included",
    )
    main = joined.loc[joined.status.eq("included")].copy()
    assert main._merge.eq("both").all() and main.designer.eq(main.spec_designer).all()
    measures = ["units", "price", "source_tdp", "spec_tdp", "h100e_per_chip", "flops16", "ops8"]
    assert np.isfinite(main[measures]).all().all() and main[measures].gt(0).all().all()
    assert main.source_tdp.eq(main.spec_tdp).all()
    main["spend_usd"] = main.units * main.price
    main["h100e"] = main.units * main.h100e_per_chip
    main["tdp_w"] = main.units * main.spec_tdp
    main["h100e16"] = main.units * main.flops16 / specs.set_index("sku").at["H100", "flops16"]
    assert np.allclose(main.spend_usd, main.source_spend, rtol=0, atol=1)
    assert np.allclose(main.tdp_w, main.source_total_tdp, rtol=0, atol=1)
    compute_delta = (main.h100e - main.source_h100e).abs()
    assert compute_delta.max() <= 1
    audit = {
        "inputs": {f.name: sha256(f.read_bytes()).hexdigest() for f in (TIMELINE, SPECS)},
        "timeline_rows": len(sales), "spec_rows": len(specs), "duplicate_timeline_keys": 0,
        "duplicate_spec_keys_after_strip": 0, "calendar_mismatch_rows": int((~calendar_match).sum()),
        "join_unmatched_rows": int(joined._merge.ne("both").sum()),
        "nvidia_join_unmatched": int(main._merge.ne("both").sum()),
        "exclusion_counts": joined.status.value_counts().to_dict(),
        "nvidia_price_counts_by_sku": main.groupby("sku").price.nunique().to_dict(),
        "max_h100e_rebuild_delta": float(compute_delta.max()),
    }
    audit_cols = ["csv_row", "source_record", "designer", "start", "end", "quarter", "incomplete",
                  "bucket", "sku", "_merge", "status", "units", "price", "source_h100e", "source_notes"]
    joined[audit_cols].to_csv(OUT / "row_audit.csv", index=False, encoding="utf-8-sig")
    return main, audit


def summarize(chips):
    by_q = chips.groupby("quarter").agg(
        rows=("sku", "size"), units=("units", "sum"), spend_usd=("spend_usd", "sum"),
        h100e=("h100e", "sum"), h100e16=("h100e16", "sum"), tdp_w=("tdp_w", "sum"),
    )
    by_q["usd_per_h100e"] = by_q.spend_usd / by_q.h100e
    by_q["usd_per_h100e16"] = by_q.spend_usd / by_q.h100e16
    by_q["tdp_w_per_h100e"] = by_q.tdp_w / by_q.h100e
    by_q["usd_per_chip"] = by_q.spend_usd / by_q.units
    by_q["h100e_per_chip"] = by_q.h100e / by_q.units
    return by_q


def decompose(quarterly):
    # Shapley average over both orders of changing A (USD/chip) and B (H100e/chip).
    # The two contributions add to the dollar change of A/B.
    rows = []
    comparisons = [(str(s), str(t), "adjacent") for s, t in pairwise(quarterly.index)]
    comparisons += [
        (FIRST, LAST, "whole_window"),
        ("2024Q4", "2025Q4", "blackwell_transition"),
    ]
    for start, end, comparison in comparisons:
        old, new = quarterly.loc[start], quarterly.loc[end]
        a0, a1 = old.usd_per_chip, new.usd_per_chip
        b0, b1 = old.h100e_per_chip, new.h100e_per_chip
        price = (a1 - a0) * (1 / b0 + 1 / b1) / 2
        performance = (a0 + a1) * (1 / b1 - 1 / b0) / 2
        delta = new.usd_per_h100e - old.usd_per_h100e
        assert math.isclose(price + performance, delta, abs_tol=1e-8)
        rows.append({
            "comparison": comparison, "start": start, "end": end, "base_usd_per_h100e": a0 / b0,
            "new_usd_per_h100e": a1 / b1, "price_contribution": price,
            "spec_contribution": performance, "net_change": delta,
        })
    return pd.DataFrame(rows)


def procurement(chips):
    sku = chips.sort_values("quarter").groupby("sku").last()
    sku = sku.loc[["A100", "H100", "B200", "B300", "R200"],
                  ["price", "spec_tdp", "h100e_per_chip", "flops16", "ops8",
                   "memory_bytes", "bandwidth_bytes_s"]].copy()
    sku["usd_per_h100e"] = sku.price / sku.h100e_per_chip
    sku["w_per_h100e"] = sku.spec_tdp / sku.h100e_per_chip
    sku["usd_per_dense16_pflops"] = sku.price / (sku.flops16 / 1e15)
    b200 = sku.loc["B200"]
    # Compare usable throughput per dollar and per rated chip watt.
    sku["throughput_ratio_to_match_b200_usd"] = sku.price / b200.price
    sku["throughput_ratio_to_match_b200_w"] = sku.spec_tdp / b200.spec_tdp
    sku["peak_ratio_to_b200_8bit"] = sku.h100e_per_chip / b200.h100e_per_chip
    sku["peak_ratio_to_b200_16bit"] = sku.flops16 / b200.flops16
    return sku


def main():
    chips, audit = read_inputs()
    quarterly = summarize(chips)
    assert quarterly.index.equals(pd.period_range(FIRST, LAST, freq="Q", name="quarter"))
    audit["quarters"] = len(quarterly)
    audit["nvidia_included_hopper_link_counts"] = (
        chips.loc[chips.bucket.eq("H100/H200"), "sku"].value_counts().to_dict()
    )
    changes = decompose(quarterly)
    sku = procurement(chips)
    cohort = chips.loc[chips.quarter.eq(LAST)].copy()
    ex_rubin = summarize(chips.loc[chips.sku.ne("R200")]).loc[LAST]
    rubin = cohort.loc[cohort.sku.eq("R200")].iloc[0]
    h20_quarter = quarterly.loc["2024Q3"]
    h20 = chips.loc[chips.quarter.eq("2024Q3") & chips.sku.eq("H20")].iloc[0]
    # Compare fixed units with a revenue-anchored scenario (units = revenue / price).
    cases = pd.DataFrame([
        {"case": "Q3 published inputs", "usd_per_h100e": quarterly.loc[LAST].usd_per_h100e},
        {"case": "Q3 exclude R200", "usd_per_h100e": ex_rubin.usd_per_h100e},
        {"case": "Q3 R200 price -20%", "usd_per_h100e":
         (cohort.spend_usd.sum() - 0.20 * rubin.spend_usd) / cohort.h100e.sum()},
        {"case": "Q3 R200 price +20%", "usd_per_h100e":
         (cohort.spend_usd.sum() + 0.20 * rubin.spend_usd) / cohort.h100e.sum()},
        {"case": "Q3 R200 price -20% fixed attributed revenue", "usd_per_h100e":
         cohort.spend_usd.sum() / (ex_rubin.h100e + rubin.h100e / 0.8)},
        {"case": "Q3 R200 price +20% fixed attributed revenue", "usd_per_h100e":
         cohort.spend_usd.sum() / (ex_rubin.h100e + rubin.h100e / 1.2)},
        {"case": "Q3 dense 16-bit", "usd_per_h100e": quarterly.loc[LAST].usd_per_h100e16},
        {"case": "Q2 dense 16-bit", "usd_per_h100e": quarterly.loc["2026Q2"].usd_per_h100e16},
        {"case": "2024 Q3 published inputs", "usd_per_h100e": h20_quarter.usd_per_h100e},
        {"case": "2024 Q3 exclude H20", "usd_per_h100e": (
            (h20_quarter.spend_usd - h20.spend_usd) / (h20_quarter.h100e - h20.h100e)
        )},
    ])
    cohort["share_units"] = cohort.units / cohort.units.sum()
    cohort["share_h100e"] = cohort.h100e / cohort.h100e.sum()
    cohort["share_spend"] = cohort.spend_usd / cohort.spend_usd.sum()
    frames = {
        "quarterly": quarterly, "chip_specs_and_thresholds": sku, "decomposition": changes,
        "quality_sensitivity": cases,
        "2026Q3_mix": cohort[["sku", "units", "spend_usd", "h100e", "share_units",
                              "share_h100e", "share_spend"]],
        "nvidia_clean": chips[["csv_row", "quarter", "bucket", "sku", "units", "price",
                               "h100e_per_chip", "spec_tdp", "spend_usd", "h100e",
                               "h100e16", "tdp_w"]],
    }
    for name, frame in frames.items():
        frame.to_csv(OUT / f"{name}.csv", index=name in {"quarterly", "chip_specs_and_thresholds"},
                     encoding="utf-8-sig")
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(quarterly[["rows", "units", "h100e", "usd_per_h100e", "usd_per_h100e16",
                     "tdp_w_per_h100e", "usd_per_chip", "h100e_per_chip"]].round(2).to_string())
    print(changes.tail(4).round(2).to_string(index=False))
    print(sku.round(3).to_string())
    print(cases.round(2).to_string(index=False))
    print(frames["2026Q3_mix"].round(4).to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
