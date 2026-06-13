#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from typing import Iterable

try:
    import polars as pl
except ImportError:  # pragma: no cover - handled at runtime in main()
    pl = None  # type: ignore[assignment]


NS_PER_SECOND = 1_000_000_000
REQUIRED_COLUMNS = ("timestamp", "quantity", "quote_qty", "is_buyer_maker")


def collect_streaming(frame: "pl.LazyFrame") -> "pl.DataFrame":
    """Collect with the streaming engine across Polars API versions."""
    try:
        return frame.collect(engine="streaming")
    except TypeError:
        return frame.collect(streaming=True)


def validate_columns(frame: "pl.LazyFrame") -> None:
    schema = frame.collect_schema()
    missing = [name for name in REQUIRED_COLUMNS if name not in schema]
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(missing)}")


def aggregate_by_second(path: str) -> "pl.DataFrame":
    data = pl.scan_parquet(path)
    validate_columns(data)

    prepared = data.select(REQUIRED_COLUMNS).with_columns(
        second=pl.col("timestamp").cast(pl.Int64) // NS_PER_SECOND,
    )

    second_agg = (
        prepared.group_by("second")
        .agg(
            buy_num=pl.when(~pl.col("is_buyer_maker"))
            .then(pl.col("quote_qty"))
            .otherwise(0.0)
            .sum(),
            buy_den=pl.when(~pl.col("is_buyer_maker"))
            .then(pl.col("quantity"))
            .otherwise(0.0)
            .sum(),
            sell_num=pl.when(pl.col("is_buyer_maker"))
            .then(pl.col("quote_qty"))
            .otherwise(0.0)
            .sum(),
            sell_den=pl.when(pl.col("is_buyer_maker"))
            .then(pl.col("quantity"))
            .otherwise(0.0)
            .sum(),
        )
        .with_columns(minute=pl.col("second") // 60)
        .sort("second")
    )

    return collect_streaming(second_agg)


def with_vwaps(frame: "pl.DataFrame") -> "pl.DataFrame":
    return frame.with_columns(
        buy_vwap=(
            pl.when(pl.col("buy_den") > 0.0)
            .then(pl.col("buy_num") / pl.col("buy_den"))
            .otherwise(None)
        ),
        sell_vwap=(
            pl.when(pl.col("sell_den") > 0.0)
            .then(pl.col("sell_num") / pl.col("sell_den"))
            .otherwise(None)
        ),
    ).with_columns(
        buy_vwap=pl.col("buy_vwap").fill_null(strategy="forward"),
        sell_vwap=pl.col("sell_vwap").fill_null(strategy="forward"),
    )


def compensated_sum(values: Iterable[float | None]) -> float:
    return math.fsum(float(value) for value in values if value is not None)


def diff_sum(frame: "pl.DataFrame") -> float:
    if frame.is_empty():
        return 0.0

    diffs = with_vwaps(frame).select(
        (pl.col("buy_vwap") - pl.col("sell_vwap")).abs().alias("diff")
    )["diff"]
    return compensated_sum(diffs)


def calculate(path: str) -> tuple[float, float]:
    if pl is None:
        raise RuntimeError("Polars is not installed")

    second_agg = aggregate_by_second(path)
    vwap_s = diff_sum(second_agg)

    minute_agg = (
        second_agg.group_by("minute")
        .agg(
            buy_num=pl.col("buy_num").sum(),
            buy_den=pl.col("buy_den").sum(),
            sell_num=pl.col("sell_num").sum(),
            sell_den=pl.col("sell_den").sum(),
        )
        .sort("minute")
    )
    vwap_m = diff_sum(minute_agg)

    return vwap_s, vwap_m


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {os.path.basename(sys.argv[0])} <input.parquet>", file=sys.stderr)
        return 2

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"error: file does not exist: {path}", file=sys.stderr)
        return 2

    try:
        vwap_s, vwap_m = calculate(path)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"VWAP_s={vwap_s:.6f}")
    print(f"VWAP_m={vwap_m:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
