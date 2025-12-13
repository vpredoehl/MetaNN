#ifndef VECTOR_ADDER_HPP
#define VECTOR_ADDER_HPP

#include <vector>

namespace demo {

// Adds two vectors element-wise. If sizes differ, up to min size is used.
std::vector<float> addVectors(const std::vector<float>& a, const std::vector<float>& b);

}

#endif // VECTOR_ADDER_HPP
