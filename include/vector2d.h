#pragma once

#include <vector>
#include <stdexcept>

template <typename T>
class vector2d_t
{
public:
    // Type aliases for iterators
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;
    using reverse_iterator = typename std::vector<T>::reverse_iterator;
    using const_reverse_iterator = typename std::vector<T>::const_reverse_iterator;

    // Constructor with dimensions and optional initial value
    vector2d_t(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols) {}

    // Copy constructor (defaulted)
    vector2d_t(const vector2d_t &) = default;
    // Copy assignment operator (defaulted)
    vector2d_t &operator=(const vector2d_t &) = default;

    // Move constructor
    vector2d_t(vector2d_t &&other) noexcept
        : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_))
    {
        other.rows_ = 0;
        other.cols_ = 0;
    }

    // Move assignment operator
    vector2d_t &operator=(vector2d_t &&other) noexcept
    {
        if (this != &other)
        {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = std::move(other.data_);
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }

    // === 1D Iterators ===
    iterator begin() noexcept { return data_.begin(); }
    iterator end() noexcept { return data_.end(); }

    const_iterator begin() const noexcept { return data_.begin(); }
    const_iterator end() const noexcept { return data_.end(); }
    const_iterator cbegin() const noexcept { return data_.cbegin(); }
    const_iterator cend() const noexcept { return data_.cend(); }

    reverse_iterator rbegin() noexcept { return data_.rbegin(); }
    reverse_iterator rend() noexcept { return data_.rend(); }

    const_reverse_iterator rbegin() const noexcept { return data_.rbegin(); }
    const_reverse_iterator rend() const noexcept { return data_.rend(); }
    const_reverse_iterator crbegin() const noexcept { return data_.crbegin(); }
    const_reverse_iterator crend() const noexcept { return data_.crend(); }

    // Non-const operator[] for row access (returns pointer to row start)
    // T &operator[](size_t row) noexcept
    // {
    //     return &data_[row * cols_];
    // }

    // // Const operator[] for row access
    // const T &operator[](size_t row) const noexcept
    // {
    //     return &data_[row * cols_];
    // }

    template <typename... Args>
    void emplace_back_row(Args &&...args)
    {
        if (sizeof...(Args) != cols_)
        {
            throw std::invalid_argument("Number of arguments must match column count");
        }
        (data_.emplace_back(std::forward<Args>(args)), ...;
        rows_++;
    }

    // Non-const version of at()
    T &at(size_t row, size_t col)
    {
        if (row >= rows_)
            throw std::out_of_range("Row index out of range");
        if (col >= cols_)
            throw std::out_of_range("Column index out of range");
        return data_[row * cols_ + col];
    }

    // Const at() with bounds checking for both dimensions
    const T &at(size_t row, size_t col) const
    {
        if (row >= rows_)
            throw std::out_of_range("Row index out of range");
        if (col >= cols_)
            throw std::out_of_range("Column index out of range");
        return data_[row * cols_ + col];
    }

    // Accessors for dimensions
    size_t rows() const noexcept
    {
        return rows_;
    }
    size_t cols() const noexcept
    {
        return cols_;
    }

private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    std::vector<T> data_;
};