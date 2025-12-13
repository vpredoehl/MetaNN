#include "VectorAdder.hpp"
#include <algorithm>

namespace demo {

std::vector<float> addVectors(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = std::min(a.size(), b.size());
    std::vector<float> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(a[i] + b[i]);
    }
    return out;
}

} // namespace demo
