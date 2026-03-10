"""Parse Bybit L2 order book data into pandas DataFrames.

Reads compressed CSV files and produces a DataFrame with columns:
timestamp, side, price, qty, level.
"""
