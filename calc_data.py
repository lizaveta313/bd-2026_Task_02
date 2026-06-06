#!/usr/bin/env python3
import os
import sys
from typing import Dict, Iterable

try:
    import numpy as np
    import pyarrow.parquet as pq
except ImportError:
    np = None
    pq = None


DEFAULT_BATCH_SIZE = 1_000_000
REQUIRED_COLUMNS = ["price", "quantity", "timestamp", "is_buyer_maker"]


def parse_batch_size() -> int:
    raw_value = os.environ.get("BATCH_SIZE")
    if raw_value is None:
        return DEFAULT_BATCH_SIZE
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_BATCH_SIZE
    return value if value > 0 else DEFAULT_BATCH_SIZE


def add_grouped_sums(
    buckets: np.ndarray,
    values_num: np.ndarray,
    values_den: np.ndarray,
    out_num: Dict[int, float],
    out_den: Dict[int, float],
) -> None:
    if buckets.size == 0:
        return

    unique_buckets, inverse = np.unique(buckets, return_inverse=True)
    grouped_num = np.bincount(inverse, weights=values_num)
    grouped_den = np.bincount(inverse, weights=values_den)

    for bucket, num, den in zip(unique_buckets, grouped_num, grouped_den):
        key = int(bucket)
        out_num[key] = out_num.get(key, 0.0) + float(num)
        out_den[key] = out_den.get(key, 0.0) + float(den)


def add_side_aggregates(
    buckets: np.ndarray,
    amount: np.ndarray,
    quantity: np.ndarray,
    side_mask: np.ndarray,
    out_num: Dict[int, float],
    out_den: Dict[int, float],
) -> None:
    # Zero-quantity rows cannot affect a valid VWAP and skipping them reduces work.
    mask = side_mask & (quantity != 0.0)
    add_grouped_sums(buckets[mask], amount[mask], quantity[mask], out_num, out_den)


def sum_abs_vwap_diff(
    buy_num: Dict[int, float],
    buy_den: Dict[int, float],
    sell_num: Dict[int, float],
    sell_den: Dict[int, float],
) -> float:
    total = 0.0
    compensation = 0.0

    if len(buy_num) <= len(sell_num):
        keys: Iterable[int] = buy_num.keys()
        other = sell_num
    else:
        keys = sell_num.keys()
        other = buy_num

    for bucket in keys:
        if bucket not in other:
            continue

        b_den = buy_den.get(bucket, 0.0)
        s_den = sell_den.get(bucket, 0.0)
        if b_den == 0.0 or s_den == 0.0:
            continue

        value = abs((buy_num[bucket] / b_den) - (sell_num[bucket] / s_den))

        # Kahan summation keeps the final bucket total stable for long inputs.
        corrected = value - compensation
        new_total = total + corrected
        compensation = (new_total - total) - corrected
        total = new_total

    return total


def calculate(path: str, batch_size: int) -> tuple[float, float]:
    if np is None or pq is None:
        raise RuntimeError("required dependencies are not installed: numpy and pyarrow")

    parquet_file = pq.ParquetFile(path)

    sec_buy_num: Dict[int, float] = {}
    sec_buy_den: Dict[int, float] = {}
    sec_sell_num: Dict[int, float] = {}
    sec_sell_den: Dict[int, float] = {}
    min_buy_num: Dict[int, float] = {}
    min_buy_den: Dict[int, float] = {}
    min_sell_num: Dict[int, float] = {}
    min_sell_den: Dict[int, float] = {}

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=REQUIRED_COLUMNS):
        price = batch.column("price").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        quantity = batch.column("quantity").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        is_buyer_maker = batch.column("is_buyer_maker").to_numpy(zero_copy_only=False).astype(
            np.bool_, copy=False
        )

        amount = price * quantity
        second_bucket = timestamp // 1000
        minute_bucket = timestamp // 60000

        buy_mask = ~is_buyer_maker
        sell_mask = is_buyer_maker

        add_side_aggregates(second_bucket, amount, quantity, buy_mask, sec_buy_num, sec_buy_den)
        add_side_aggregates(second_bucket, amount, quantity, sell_mask, sec_sell_num, sec_sell_den)
        add_side_aggregates(minute_bucket, amount, quantity, buy_mask, min_buy_num, min_buy_den)
        add_side_aggregates(minute_bucket, amount, quantity, sell_mask, min_sell_num, min_sell_den)

    vwap_s = sum_abs_vwap_diff(sec_buy_num, sec_buy_den, sec_sell_num, sec_sell_den)
    vwap_m = sum_abs_vwap_diff(min_buy_num, min_buy_den, min_sell_num, min_sell_den)
    return vwap_s, vwap_m


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {os.path.basename(sys.argv[0])} <path_to_parquet>", file=sys.stderr)
        return 1

    path = sys.argv[1]
    if not os.path.isfile(path) or not os.access(path, os.R_OK):
        print(f"error: parquet file is not readable: {path}", file=sys.stderr)
        return 1

    try:
        vwap_s, vwap_m = calculate(path, parse_batch_size())
    except Exception as exc:
        print(f"error: failed to read or process parquet file: {exc}", file=sys.stderr)
        return 1

    print(f"VWAP_s={vwap_s:.6f}")
    print(f"VWAP_m={vwap_m:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
