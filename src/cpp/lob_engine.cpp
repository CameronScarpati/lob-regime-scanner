// LOB Reconstruction Engine — Implementation
// High-performance order book reconstructor.

#include "lob_engine.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Mutators
// ---------------------------------------------------------------------------

void LOBEngine::update(const std::string& side, double price, double qty) {
    if (side == "bid") {
        if (qty <= 0.0)
            bids_.erase(price);
        else
            bids_[price] = qty;
    } else {
        if (qty <= 0.0)
            asks_.erase(price);
        else
            asks_[price] = qty;
    }
}

void LOBEngine::apply_snapshot(
    const std::string& side,
    const std::vector<std::pair<double, double>>& levels) {
    if (side == "bid") {
        bids_.clear();
        for (auto& [p, q] : levels) {
            if (q > 0.0) bids_[p] = q;
        }
    } else {
        asks_.clear();
        for (auto& [p, q] : levels) {
            if (q > 0.0) asks_[p] = q;
        }
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

std::pair<double, double> LOBEngine::best_bid() const {
    if (bids_.empty()) return {NAN_VAL, NAN_VAL};
    auto it = bids_.begin();  // greatest key thanks to std::greater
    return {it->first, it->second};
}

std::pair<double, double> LOBEngine::best_ask() const {
    if (asks_.empty()) return {NAN_VAL, NAN_VAL};
    auto it = asks_.begin();  // smallest key (default ordering)
    return {it->first, it->second};
}

double LOBEngine::mid_price() const {
    if (bids_.empty() || asks_.empty()) return NAN_VAL;
    return (bids_.begin()->first + asks_.begin()->first) / 2.0;
}

double LOBEngine::spread() const {
    if (bids_.empty() || asks_.empty()) return NAN_VAL;
    return asks_.begin()->first - bids_.begin()->first;
}

std::vector<std::pair<double, double>>
LOBEngine::top_n(const std::string& side, int n) const {
    std::vector<std::pair<double, double>> result;
    result.reserve(n);

    if (side == "bid") {
        // bids_ already sorted descending by price (std::greater)
        int count = 0;
        for (auto it = bids_.begin(); it != bids_.end() && count < n;
             ++it, ++count) {
            result.emplace_back(it->first, it->second);
        }
    } else {
        // asks_ sorted ascending by price (default)
        int count = 0;
        for (auto it = asks_.begin(); it != asks_.end() && count < n;
             ++it, ++count) {
            result.emplace_back(it->first, it->second);
        }
    }

    // Pad with NaN
    while (static_cast<int>(result.size()) < n) {
        result.emplace_back(NAN_VAL, NAN_VAL);
    }
    return result;
}

void LOBEngine::snapshot_dict(int64_t timestamp_us, int n_levels,
                              std::vector<std::string>& keys,
                              std::vector<double>& values) const {
    // 3 fixed cols + 4 * n_levels
    int n_cols = 3 + 4 * n_levels;
    keys.clear();
    keys.reserve(n_cols);
    values.clear();
    values.reserve(n_cols);

    keys.push_back("timestamp");
    values.push_back(static_cast<double>(timestamp_us));

    keys.push_back("mid_price");
    values.push_back(mid_price());

    keys.push_back("spread");
    values.push_back(spread());

    auto bid_levels = top_n("bid", n_levels);
    auto ask_levels = top_n("ask", n_levels);

    for (int i = 0; i < n_levels; ++i) {
        keys.push_back("bid_price_" + std::to_string(i + 1));
        values.push_back(bid_levels[i].first);
        keys.push_back("bid_qty_" + std::to_string(i + 1));
        values.push_back(bid_levels[i].second);
    }

    for (int i = 0; i < n_levels; ++i) {
        keys.push_back("ask_price_" + std::to_string(i + 1));
        values.push_back(ask_levels[i].first);
        keys.push_back("ask_qty_" + std::to_string(i + 1));
        values.push_back(ask_levels[i].second);
    }
}

// ---------------------------------------------------------------------------
// Batch reconstruct — processes the entire event array in C++ and returns
// column-major data ready for DataFrame construction.
//
//   types:  0 = snapshot, 1 = delta
//   sides:  0 = bid, 1 = ask
// ---------------------------------------------------------------------------

std::pair<std::vector<std::string>, std::vector<double>>
LOBEngine::batch_reconstruct(const int64_t* timestamps, const int* types,
                             const int* sides, const double* prices,
                             const double* qtys, const int64_t* update_ids,
                             int64_t n_events, int n_levels) {
    LOBEngine book;
    int n_cols = 3 + 4 * n_levels;

    // First pass: count unique timestamps for output allocation
    // (We emit one snapshot per unique timestamp.)
    std::vector<int64_t> unique_ts;
    unique_ts.reserve(1024);
    int64_t prev_ts = -1;

    for (int64_t i = 0; i < n_events; ++i) {
        if (timestamps[i] != prev_ts) {
            unique_ts.push_back(timestamps[i]);
            prev_ts = timestamps[i];
        }
    }

    // Pre-allocate output (column-major: col0[row0..N], col1[row0..N], ...)
    int n_snapshots = static_cast<int>(unique_ts.size());
    std::vector<double> data(static_cast<size_t>(n_snapshots) * n_cols, NAN_VAL);

    // Column names
    std::vector<std::string> col_names;
    col_names.reserve(n_cols);
    col_names.push_back("timestamp");
    col_names.push_back("mid_price");
    col_names.push_back("spread");
    for (int i = 1; i <= n_levels; ++i) {
        col_names.push_back("bid_price_" + std::to_string(i));
        col_names.push_back("bid_qty_" + std::to_string(i));
    }
    for (int i = 1; i <= n_levels; ++i) {
        col_names.push_back("ask_price_" + std::to_string(i));
        col_names.push_back("ask_qty_" + std::to_string(i));
    }

    // Helper: write current book state as a snapshot row into column-major data.
    auto write_snapshot = [&](int row, int64_t ts_val) {
        size_t base = static_cast<size_t>(row);
        data[base] = static_cast<double>(ts_val);
        data[base + static_cast<size_t>(n_snapshots)] = book.mid_price();
        data[base + static_cast<size_t>(n_snapshots) * 2] = book.spread();

        auto blevels = book.top_n("bid", n_levels);
        auto alevels = book.top_n("ask", n_levels);

        for (int l = 0; l < n_levels; ++l) {
            size_t col_bid_p = 3 + static_cast<size_t>(l) * 2;
            size_t col_bid_q = col_bid_p + 1;
            data[base + col_bid_p * n_snapshots] = blevels[l].first;
            data[base + col_bid_q * n_snapshots] = blevels[l].second;
        }
        for (int l = 0; l < n_levels; ++l) {
            size_t col_ask_p =
                3 + static_cast<size_t>(n_levels) * 2 +
                static_cast<size_t>(l) * 2;
            size_t col_ask_q = col_ask_p + 1;
            data[base + col_ask_p * n_snapshots] = alevels[l].first;
            data[base + col_ask_q * n_snapshots] = alevels[l].second;
        }
    };

    // Second pass: process events and emit snapshots.
    // Match Python semantics: process each (ts, type, uid) group, then
    // emit a snapshot when we transition to a new timestamp.
    int snap_idx = 0;
    int64_t current_ts = -1;
    int64_t event_idx = 0;

    std::vector<std::pair<double, double>> snap_bid_levels;
    std::vector<std::pair<double, double>> snap_ask_levels;

    while (event_idx < n_events) {
        int64_t ts = timestamps[event_idx];
        int64_t group_uid = update_ids[event_idx];
        int group_type = types[event_idx];

        // Process the event group first (before emitting)
        if (group_type == 0) {
            snap_bid_levels.clear();
            snap_ask_levels.clear();

            while (event_idx < n_events && timestamps[event_idx] == ts &&
                   types[event_idx] == 0 && update_ids[event_idx] == group_uid) {
                if (sides[event_idx] == 0)
                    snap_bid_levels.emplace_back(prices[event_idx],
                                                 qtys[event_idx]);
                else
                    snap_ask_levels.emplace_back(prices[event_idx],
                                                 qtys[event_idx]);
                event_idx++;
            }

            if (!snap_bid_levels.empty())
                book.apply_snapshot("bid", snap_bid_levels);
            if (!snap_ask_levels.empty())
                book.apply_snapshot("ask", snap_ask_levels);
        } else {
            while (event_idx < n_events && timestamps[event_idx] == ts &&
                   types[event_idx] == 1 && update_ids[event_idx] == group_uid) {
                book.update(sides[event_idx] == 0 ? "bid" : "ask",
                            prices[event_idx], qtys[event_idx]);
                event_idx++;
            }
        }

        book.set_last_update_ts(ts);

        // Emit snapshot when we move to a new timestamp (matching Python)
        if (ts != current_ts) {
            if (current_ts >= 0) {
                write_snapshot(snap_idx, current_ts);
                snap_idx++;
            }
            current_ts = ts;
        }
    }

    // Final snapshot
    if (current_ts >= 0 && snap_idx < n_snapshots) {
        write_snapshot(snap_idx, current_ts);
        snap_idx++;
    }

    // Trim to actual snapshot count if we over-allocated
    if (snap_idx < n_snapshots) {
        n_snapshots = snap_idx;
        data.resize(static_cast<size_t>(n_snapshots) * n_cols);
    }

    return {col_names, data};
}
