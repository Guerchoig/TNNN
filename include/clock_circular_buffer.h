#pragma once
#include <iostream>
#include <stdexcept>
#include "common.h"

// template <size_t Capacity>
unsigned short constexpr nof_time_points = 4;
struct clock_circular_buffer
{
    // public:
    clock_circular_buffer();

    void push(clock_count_t value);
    clock_count_t pop();
    clock_count_t peek() const;

    bool is_empty() const;
    bool is_full() const;

    size_t size() const;
    size_t capacity() const;

    // private:
    std::array<clock_count_t, nof_time_points> buffer_;
    size_t size_;
    size_t head_;
    size_t tail_;
};