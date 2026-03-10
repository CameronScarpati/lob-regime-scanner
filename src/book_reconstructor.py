"""Order book snapshot reconstructor.

Maintains full bid/ask book as sorted price-level arrays,
resamples to uniform time intervals, and outputs Parquet snapshots.
"""
