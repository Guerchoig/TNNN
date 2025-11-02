#pragma once

#include "logger.h"
#include <unordered_map>
#include <string>
#include <iostream>
#include <stdexcept>
#include <concepts>

/**
 * @file counters.h
 * @brief Small header-only counters utility used across the project
 *
 * This header defines a lightweight templated counters container `counters_t<T>`
 * that provides named counters with a selectable aggregation strategy.
 * The container is intentionally simple and intended for metrics/logging
 * (not high-performance telemetry). It integrates with the project's
 * `text_logger` via `logger` global.
 *
 * Template parameter:
 * - T: arithmetic type used to store counter values (int, double, etc.)
 *
 * Aggregation strategies:
 * - sum  : accumulated sum
 * - avg  : running average (keeps update count)
 * - min  : keep minimum (not implemented separately; treated as noagg by name)
 * - max  : keep maximum (not implemented separately; treated as noagg by name)
 * - noagg: store the last provided value
 *
 * Usage example:
 * @code
 * counters_t<double> stats;
 * stats.add<counters_t<double>::sum>("requests");
 * stats.inc_by<counters_t<double>::sum>("requests", 1.0);
 * stats.print_all_counters();
 * @endcode
 */

template <typename T>
    requires std::is_arithmetic_v<T>
class counters_t
{
public:
    /**
     * Reset all counters' values to 0 (and update count to 0).
     * This does not remove counters or change their aggregation type.
     */
    void reset_all()
    {
        for (auto &[name, data] : counters_)
        {
            data.value = T{};
            data.counter = 0;
        }
    }
    /**
     * Aggregation policy for a named counter.
     * - sum  : increments accumulate as a sum
     * - avg  : maintains a running average (counter tracks samples)
     * - min  : reserved (treated as noagg in current implementation)
     * - max  : reserved (treated as noagg in current implementation)
     * - noagg: no aggregation — each update replaces the stored value
     */
    enum aggregation_type
    {
        sum,
        avg,
        min,
        max,
        noagg // means just save
    };

    /**
     * Per-counter stored data: value, aggregation type and number of updates.
     * `counter` is used by the `avg` aggregation to compute running average.
     */
    struct counter_data_t
    {
        T value;
        aggregation_type type;
        uint64_t counter = 0; // Track updates per counter
    };

    using iterator = typename std::unordered_map<std::string, counter_data_t>::iterator;
    using const_iterator = typename std::unordered_map<std::string, counter_data_t>::const_iterator;

    /**
     * Get the current value of a named counter. Throws std::out_of_range if
     * the counter does not exist.
     */
    T get(const std::string &name) const
    {
        auto it = counters_.find(name);
        if (it == counters_.end())
            throw std::out_of_range("Counter with name '" + name + "' not found.");
        return it->second.value;
    }

    /**
     * Add a new named counter with aggregation policy `TY`.
     * This overload uses a template parameter for the aggregation_type so
     * callers write `add<counters_t<T>::sum>("name")` which is concise and
     * constexpr-friendly.
     */
    template <counters_t<T>::aggregation_type TY>
    void add(const std::string &name)
    {
        if (counters_.contains(name))
        {
            throw std::invalid_argument("Counter with name '" + name + "' already exists.");
        }
        counters_[name] = counter_data_t({T{}, TY, 0});
    }

    /**
     * Check whether a named counter exists.
     */
    bool contains(const std::string &name) const
    {
        return counters_.contains(name);
    }

    /**
     * Increment a named counter by one. If the counter does not exist it is
     * created with the `sum` aggregation policy.
     */
    void inc(const std::string &name)
    {
        if (!counters_.contains(name))
            counters_[name] = counter_data_t({T{}, sum, 0});
        counters_[name].value++;
    }

    /**
     * Reset a named counter according to aggregation policy `TY`.
     * Implemented via tag-dispatch to avoid runtime branching on the enum.
     */
    template <aggregation_type TY>
    void reset(const std::string &name)
    {
        reset_impl(name, std::integral_constant<aggregation_type, TY>());
    }

    /**
     * Increase (or set) the counter by `value` according to aggregation policy
     * `TY`. For `sum` this adds, for `avg` it updates a running mean, for
     * `noagg` it assigns the new value.
     */
    template <aggregation_type TY>
    void inc_by(const std::string &name, T value)
    {
        inc_by_impl(name, value, std::integral_constant<aggregation_type, TY>());
    }

    counter_data_t &operator[](const std::string &name)
    {
        return counters_.at(name);
    }

    const counter_data_t &operator[](const std::string &name) const
    {
        return counters_.at(name);
    }

    /**
     * Print a single counter identified by `name` using the global `logger`.
     * If the container is empty this function silently returns.
     */
    void print_by_name(const std::string &name) const
    {
        auto it = counters_.find(name);
        if (it == counters_.end())
        {
            if (counters_.size() > 0)
                throw std::out_of_range("Counter with name '" + name + "' not found.");
            else
                return;
        }
        print_a_counter(*it);
    }

    /**
     * Print all counters (name, type, value, update count) prefixed with
     * `ident` which can be used to align or label the output.
     */
    void print_all_counters(const std::string &ident = "") const
    {
        for (auto it = counters_.begin(); it != counters_.end(); it++)
        {
            logger << ident;
            print_a_counter(*it);
        }
    }

    /**
     * Print a single-line header suitable for `print_counters_as_a_table`.
     */
    static void print_counters_header()
    {
        logger << "Name\tType\tValue\tUpdates" << std::endl;
    }

    /**
     * Helper to print counters in a tabular format with iteration and layer
     * identifiers. Useful when exporting metrics over time.
     */
    void print_counters_as_a_table(uint iteration, uint layer_num) const
    {
        for (auto it = counters_.begin(); it != counters_.end(); it++)
            logger << iteration << "\t" << layer_num << "\t" << it->first << "\t" << type_name_(it->second.type) << "\t"
                   << it->second.value << "\t" << it->second.counter << std::endl;
    }

    iterator begin() noexcept
    {
        return counters_.begin();
    }
    iterator end() noexcept { return counters_.end(); }
    const_iterator begin() const noexcept { return counters_.begin(); }
    const_iterator end() const noexcept { return counters_.end(); }
    const_iterator cbegin() const noexcept { return counters_.cbegin(); }
    const_iterator cend() const noexcept { return counters_.cend(); }

private:
    std::string type_name_(aggregation_type type) const
    {
        switch (type)
        {
        case sum:
            return "sum";
        case avg:
            return "avg";
        case min:
            return "min";
        case max:
            return "max";
        case noagg:
            return "noagg";
        default:
            break;
        };
        return "noagg";
    }

    void inc_by_impl(const std::string &name, T value, std::integral_constant<aggregation_type, sum>)
    {
        if (!counters_.contains(name))
            counters_[name] = counter_data_t({T{}, sum, 0});
        counters_[name].value += value;
    }

    void inc_by_impl(const std::string &name, T value, std::integral_constant<aggregation_type, avg>)
    {
        if (!counters_.contains(name))
            counters_[name] = counter_data_t({T{}, avg, 0});

        auto &data = counters_[name];
        data.value = data.value * (static_cast<T>(data.counter) / (data.counter + 1.0)) + value / (data.counter + 1.0); // TODO: Check overflowval = counters_[name].value * (static_cast<T>(counter) / (counter + 1.0));
        data.counter++;
    }

    void inc_by_impl(const std::string &name, T value, std::integral_constant<aggregation_type, noagg>)
    {
        if (!counters_.contains(name))
            counters_[name] = counter_data_t({T{}, noagg, 0});
        counters_[name].value = value;
    }

    void reset_impl(const std::string &name, std::integral_constant<aggregation_type, noagg>)
    {
        counters_[name] = counter_data_t({T{}, noagg, 0});
    }
    void reset_impl(const std::string &name, std::integral_constant<aggregation_type, sum>)
    {
        counters_[name] = counter_data_t({T{}, sum, 0});
    }

    void reset_impl(const std::string &name, std::integral_constant<aggregation_type, avg>)
    {
        counters_[name] = counter_data_t({T{}, avg, 0});
    }

    void print_a_counter(const std::pair<std::string, counters_t<T>::counter_data_t> counter) const
    {
        logger << counter.first
               << " (" << type_name_(counter.second.type) << ") : " << counter.second.value
               << " updates:" << counter.second.counter << std::endl;
    }

    std::unordered_map<std::string, counter_data_t> counters_;
};

// struct avg_counter_t
// {
//     std::atomic<double> avg_value{0};
//     std::atomic<uint64_t> counter{0};
//     void add_value(uint64_t value)
//     {
//         uint64_t n = counter.fetch_add(1, std::memory_order_relaxed);
//         double new_avg = (avg_value.load(std::memory_order_relaxed) * n + value) / (n + 1);
//         avg_value.store(new_avg, std::memory_order_relaxed);
//         counter++;
//     }
//     uint64_t get_value() { return avg_value.load(std::memory_order_relaxed); }
// };

// struct ticks_counter_t
// {
//     // std::atomic<double> average_tick{0};
//     // std::atomic<uint64_t> nof_ticks{0};
//     avg_counter_t average_tick;
//     clock_count_t tick_start;

//     void start_tick() { tick_start = clock(); }

//     void stop_tick()
//     {
//         double end = clock();
//         double delta = end - tick_start;
//         average_tick.add_value(delta);
//     }

//     double get_average_tick()
//     {
//         return average_tick.get_value();
//     }
// };
