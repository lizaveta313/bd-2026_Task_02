#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

try:
    import polars as pl
except ImportError:
    pl = None


EXPECTED_SCHEMA = {
    "trade_id": pl.UInt64 if pl is not None else None,
    "timestamp": pl.Datetime("ns") if pl is not None else None,
    "price": pl.Float64 if pl is not None else None,
    "quantity": pl.Float64 if pl is not None else None,
    "quote_qty": pl.Float64 if pl is not None else None,
    "is_buyer_maker": pl.Boolean if pl is not None else None,
}
REQUIRED_COLUMNS = ["timestamp", "price", "quantity", "is_buyer_maker"]
NS_PER_SECOND = 1_000_000_000
NS_PER_MINUTE = 60 * NS_PER_SECOND


def validate_schema(schema: pl.Schema) -> None:
    for column_name, expected_type in EXPECTED_SCHEMA.items():
        actual_type = schema.get(column_name)
        if actual_type is None:
            raise ValueError(f"missing required column: {column_name}")
        if actual_type != expected_type:
            raise ValueError(
                f"column {column_name} has type {actual_type}, expected {expected_type}"
            )


def sum_diff_plan(data: pl.LazyFrame, name: str) -> pl.LazyFrame:
    return (
        data.filter((pl.col("buy_den") != 0.0) & (pl.col("sell_den") != 0.0))
        .select(
            (
                (pl.col("buy_num") / pl.col("buy_den"))
                - (pl.col("sell_num") / pl.col("sell_den"))
            )
            .abs()
            .sum()
            .alias(name)
        )
    )


def build_plans(data: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    second_agg = (
        data.select(REQUIRED_COLUMNS)
        .filter(pl.col("quantity") != 0.0)
        .with_columns(
            timestamp_ns=pl.col("timestamp").cast(pl.Int64),
            amount=pl.col("price") * pl.col("quantity"),
        )
        .with_columns(
            second=pl.col("timestamp_ns") // NS_PER_SECOND,
            minute=pl.col("timestamp_ns") // NS_PER_MINUTE,
        )
        .group_by("second", "minute")
        .agg(
            buy_num=pl.when(~pl.col("is_buyer_maker"))
            .then(pl.col("amount"))
            .otherwise(0.0)
            .sum(),
            buy_den=pl.when(~pl.col("is_buyer_maker"))
            .then(pl.col("quantity"))
            .otherwise(0.0)
            .sum(),
            sell_num=pl.when(pl.col("is_buyer_maker"))
            .then(pl.col("amount"))
            .otherwise(0.0)
            .sum(),
            sell_den=pl.when(pl.col("is_buyer_maker"))
            .then(pl.col("quantity"))
            .otherwise(0.0)
            .sum(),
        )
    )

    minute_agg = (
        second_agg.group_by("minute")
        .agg(
            buy_num=pl.col("buy_num").sum(),
            buy_den=pl.col("buy_den").sum(),
            sell_num=pl.col("sell_num").sum(),
            sell_den=pl.col("sell_den").sum(),
        )
    )

    return sum_diff_plan(second_agg, "VWAP_s"), sum_diff_plan(minute_agg, "VWAP_m")


def frame_value(frame: pl.DataFrame) -> float:
    value = frame.item()
    return 0.0 if value is None else float(value)


def calculate(path: str) -> tuple[float, float]:
    if pl is None:
        raise RuntimeError("required dependency is not installed: polars")

    data = pl.scan_parquet(path)
    validate_schema(data.collect_schema())

    second_plan, minute_plan = build_plans(data)
    second_result, minute_result = pl.collect_all(
        [second_plan, minute_plan],
        engine="streaming",
    )
    return frame_value(second_result), frame_value(minute_result)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {os.path.basename(sys.argv[0])} <path_to_parquet>", file=sys.stderr)
        return 1

    path = sys.argv[1]
    if not os.path.isfile(path) or not os.access(path, os.R_OK):
        print(f"error: parquet file is not readable: {path}", file=sys.stderr)
        return 1

    try:
        vwap_s, vwap_m = calculate(path)
    except Exception as exc:
        print(f"error: failed to read or process parquet file: {exc}", file=sys.stderr)
        return 1

    print(f"VWAP_s={vwap_s:.6f}")
    print(f"VWAP_m={vwap_m:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
