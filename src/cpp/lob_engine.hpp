// LOB Reconstruction Engine — Header
// High-performance order book reconstructor using sorted price-level maps.
// Target: 1M+ updates/second.

#ifndef LOB_ENGINE_HPP
#define LOB_ENGINE_HPP

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

class LOBEngine {
public:
    LOBEngine() : last_update_ts_(0) {}

    // --- Mutators ---

    /// Apply a single price-level update. qty <= 0 removes the level.
    void update(const std::string& side, double price, double qty);

    /// Replace one entire side of the book with a full snapshot.
    void apply_snapshot(const std::string& side,
                        const std::vector<std::pair<double, double>>& levels);

    // --- Queries ---

    /// Best bid as (price, qty). Returns (NaN, NaN) if empty.
    std::pair<double, double> best_bid() const;

    /// Best ask as (price, qty). Returns (NaN, NaN) if empty.
    std::pair<double, double> best_ask() const;

    /// Mid-price = (best_bid + best_ask) / 2. Returns NaN if empty.
    double mid_price() const;

    /// Spread = best_ask - best_bid. Returns NaN if empty.
    double spread() const;

    /// Top N levels for a side. Bids descending, asks ascending.
    /// Pads with (NaN, NaN) if fewer than n levels exist.
    std::vector<std::pair<double, double>> top_n(const std::string& side,
                                                  int n) const;

    /// Flat snapshot dict as parallel key/value vectors.
    /// Keys: timestamp, mid_price, spread, bid_price_1..N, bid_qty_1..N,
    ///        ask_price_1..N, ask_qty_1..N.
    void snapshot_dict(int64_t timestamp_us, int n_levels,
                       std::vector<std::string>& keys,
                       std::vector<double>& values) const;

    // --- Batch reconstruct ---

    /// Process a full events array and return snapshot rows as a flat
    /// column-major matrix (n_snapshots x n_cols). Also returns column names.
    /// This avoids per-event Python overhead entirely.
    static std::pair<std::vector<std::string>, std::vector<double>>
    batch_reconstruct(const int64_t* timestamps, const int* types,
                      const int* sides, const double* prices,
                      const double* qtys, const int64_t* update_ids,
                      int64_t n_events, int n_levels);

    void set_last_update_ts(int64_t ts) { last_update_ts_ = ts; }
    int64_t last_update_ts() const { return last_update_ts_; }

private:
    // bids: std::map with std::greater -> highest price first (rbegin = best)
    // asks: std::map with default order -> lowest price first (begin = best)
    std::map<double, double, std::greater<double>> bids_;
    std::map<double, double> asks_;
    int64_t last_update_ts_;

    static constexpr double NAN_VAL = std::numeric_limits<double>::quiet_NaN();
};

#endif // LOB_ENGINE_HPP
