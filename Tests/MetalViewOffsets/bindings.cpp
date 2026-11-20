#include "metal_view_binding.h"

#include <array>
#include <cassert>
#include <iostream>
#include <utility>

using MetaNN::NSMetalAdd::Detail::BindView;

// Metal-free model of the public view interface. It lets the real binding
// helper feed a host encoder and makes incorrect offsets observable as writes.
struct View
{
    std::size_t start;
    std::size_t size;
    std::size_t Offset() const { return start; }
    std::size_t Size() const { return size; }
    View Shift(std::size_t by) const
    {
        assert(by < size);
        return {start + by, size - by};
    }
};

template <typename Error, typename Run>
void Reject(Run run)
{
    try { run(); }
    catch (const Error&) { return; }
    assert(false && "invalid view reached encoder");
}

int main()
{
    const auto offset = []<typename Element>(View view, std::size_t bytes, std::size_t count,
                                             std::size_t slot)
    {
        std::pair<std::size_t, std::size_t> binding{};
        BindView<Element>(view, bytes, count, slot,
            [&](std::size_t address, std::size_t index) { binding = {address, index}; });
        assert(binding.second == slot);
        return binding.first;
    };
    const auto exercise = [&](View a, View b, View c)
    {
        std::array<float, 12> left{}, right{}, out;
        out.fill(-123);
        for (std::size_t i = 0; i < 4; ++i)
        {
            left[a.start + i] = float(i + 1);
            right[b.start + i] = float(i + 5);
        }
        const auto ia = offset.operator()<float>(a, sizeof(left), 4, 0) / sizeof(float);
        const auto ib = offset.operator()<float>(b, sizeof(right), 4, 1) / sizeof(float);
        const auto ic = offset.operator()<float>(c, sizeof(out), 4, 2) / sizeof(float);
        for (std::size_t i = 0; i < 4; ++i)
            out[ic + i] = left[ia + i] + right[ib + i];
        for (std::size_t i = 0; i < out.size(); ++i)
            assert(out[i] == (i >= c.start && i < c.start + 4 ? float(6 + 2 * (i - c.start)) : -123));
    };

    const View base{0, 12};
    const auto shifted = base.Shift(4);
    exercise(base, base, base);
    exercise(shifted, base, base);
    exercise(base, shifted, base);
    exercise(base, base, shifted);
    exercise(base.Shift(1), base.Shift(2), base.Shift(6));
    exercise(base.Shift(1).Shift(3), base.Shift(3).Shift(1), shifted);
    exercise(shifted, shifted, shifted);

    // Unary dispatch binds input in slot 0 and output in slot 1. An exact
    // same-range alias must retain the same offsets and require no copy.
    std::array<float, 4> aliased{-123, 5, 7, -123};
    const auto input = offset.operator()<float>({1, 2}, sizeof(aliased), 2, 0) / 4;
    const auto output = offset.operator()<float>({1, 2}, sizeof(aliased), 2, 1) / 4;
    for (std::size_t i = 0; i < 2; ++i)
        aliased[output + i] = -aliased[input + i];
    assert((aliased == std::array<float, 4>{-123, -5, -7, -123}));

    assert(offset.operator()<float>({11, 1}, 48, 1, 2) == 44);
    assert(offset.operator()<double>({3, 1}, 32, 1, 1) == 24);
    assert(offset.operator()<float>(base, 48, 0, 2) == 0);

    const auto forbidden = [](std::size_t, std::size_t) { assert(false && "encoder called"); };
    Reject<std::out_of_range>([&] { BindView<float>(View{0, 1}, 48, 2, 0, forbidden); });
    Reject<std::out_of_range>([&] { BindView<float>(View{11, 2}, 48, 2, 0, forbidden); });
    Reject<std::out_of_range>([&] { BindView<float>(View{13, 1}, 48, 1, 0, forbidden); });
    constexpr auto max = std::numeric_limits<std::size_t>::max();
    Reject<std::overflow_error>([&] { BindView<double>(View{max / 8 + 1, 1}, max, 1, 0, forbidden); });
    Reject<std::overflow_error>([&] { BindView<float>(View{0, max}, max, max / 4 + 1, 0, forbidden); });
    if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t))
    {
        constexpr auto count = std::size_t{1} << 32;
        Reject<std::overflow_error>([&] { BindView<float>(View{0, count}, max, count, 0, forbidden); });
    }
    std::cout << "PASS 17 binding and guard cases\n";
}
