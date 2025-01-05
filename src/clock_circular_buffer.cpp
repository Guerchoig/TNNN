#include "clock_circular_buffer.h"
#include "common.h"

clock_circular_buffer::clock_circular_buffer() : size_(0), head_(0), tail_(0)
{
}

void clock_circular_buffer::push(clock_count_t value)
{
    if (is_full())
    {
        throw std::overflow_error("Buffer is full");
    }
    buffer_[tail_] = value;
    tail_ = (tail_ + 1) % nof_time_points;
    ++size_;
}

clock_count_t clock_circular_buffer::pop()
{
    if (is_empty())
    {
        throw std::underflow_error("Buffer is empty");
    }
    clock_count_t value = buffer_[head_];
    head_ = (head_ + 1) % nof_time_points;
    --size_;
    return value;
}

clock_count_t clock_circular_buffer::peek() const
{
    if (is_empty())
    {
        throw std::underflow_error("Buffer is empty");
    }
    return buffer_[head_];
}

bool clock_circular_buffer::is_empty() const
{
    return size_ == 0;
}

bool clock_circular_buffer::is_full() const
{
    return size_ == nof_time_points;
}

size_t clock_circular_buffer::size() const
{
    return size_;
}

size_t clock_circular_buffer::capacity() const
{
    return nof_time_points;
}
