// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <cmath>


namespace ov {
namespace genai {
namespace numpy_utils {

// https://gist.github.com/lorenzoriano/5414671
template <typename T, typename U>
std::vector<T> linspace(U start, U end, size_t num, bool endpoint = false) {
    std::vector<T> indices;
    if (num != 0) {
        if (num == 1)
            indices.push_back(static_cast<T>(start));
        else {
            if (endpoint)
                --num;

            U delta = (end - start) / static_cast<U>(num);
            for (size_t i = 0; i < num; i++)
                indices.push_back(static_cast<T>(start + delta * i));

            if (endpoint)
                indices.push_back(static_cast<T>(end));
        }
    }
    return indices;
}

// Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)
void rescale_zero_terminal_snr(std::vector<float>& betas);

// np.interp(...) implementation
std::vector<float> interp(const std::vector<std::int64_t>& x, const std::vector<size_t>& xp, const std::vector<float>& fp);

} // namespace ov
} // namespace genai
} // namespace numpy_utils
