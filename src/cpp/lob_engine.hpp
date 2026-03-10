// LOB Reconstruction Engine — Header
// High-performance order book reconstructor using sorted price-level maps.
// Target: 1M+ updates/second.

#ifndef LOB_ENGINE_HPP
#define LOB_ENGINE_HPP

#include <map>
#include <vector>
#include <string>

class LOBEngine {
public:
    void update(const std::string& side, double price, double qty);
    std::vector<std::pair<double, double>> snapshot(const std::string& side, int levels) const;

private:
    std::map<double, double> bids_;  // price -> qty (descending)
    std::map<double, double> asks_;  // price -> qty (ascending)
};

#endif // LOB_ENGINE_HPP
