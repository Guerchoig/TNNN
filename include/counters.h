#include "logger.h"
#include <unordered_map>
#include <string>
#include <iostream>
#include <stdexcept>
#include <concepts>

template <typename T>
    requires std::is_arithmetic_v<T>
class counters_t
{
public:
    enum aggregation_type
    {
        sum,
        avg,
        min,
        max,
        noagg // means just save
    };

    struct counter_data_t
    {
        T value;
        aggregation_type type;
        uint64_t update_count = 0; // Track updates per counter
    };

    using iterator = typename std::unordered_map<std::string, counter_data_t>::iterator;
    using const_iterator = typename std::unordered_map<std::string, counter_data_t>::const_iterator;

    T get(const std::string &name) const
    {
        auto it = counters_.find(name);
        if (it == counters_.end())
            throw std::out_of_range("Counter with name '" + name + "' not found.");
        return it->second.value;
    }

    template <counters_t<T>::aggregation_type TY>
    void add(const std::string &name)
    {
        if (counters_.contains(name))
        {
            throw std::invalid_argument("Counter with name '" + name + "' already exists.");
        }
        counters_[name] = counter_data_t({T{}, TY, 0});
    }

    bool contains(const std::string &name) const
    {
        return counters_.contains(name);
    }

    void inc(const std::string &name)
    {
        if (!counters_.contains(name))
            counters_[name] = counter_data_t({T{}, sum, 0});
        counters_[name].value++;
    }

    template <aggregation_type TY>
    void reset(const std::string &name)
    {
        reset_impl(name, std::integral_constant<aggregation_type, TY>());
    }

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

    void print_all_counters(std::string &ident = "") const
    {
        for (auto it = counters_.begin(); it != counters_.end(); it++)
        {
            logger << ident;
            print_a_counter(*it);
        }
    }

    static void print_counters_header()
    {
        logger << "Name\tType\tValue\tUpdates" << std::endl;
    }

    void print_counters_as_a_table(uint iteration, uint layer_num) const
    {
        for (auto it = counters_.begin(); it != counters_.end(); it++)
            logger << iteration << "\t" << layer_num << "\t" << it->first << "\t" << type_name_(it->second.type) << "\t"
                   << it->second.value << "\t" << it->second.update_count << std::endl;
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
        data.value = data.value * (static_cast<T>(data.update_count) / (data.update_count + 1.0)) + value / (data.update_count + 1.0); // TODO: Check overflowval = counters_[name].value * (static_cast<T>(update_count) / (update_count + 1.0));
        data.update_count++;
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
               << " updates:" << counter.second.update_count << std::endl;
    }

    std::unordered_map<std::string, counter_data_t> counters_;
};

// struct avg_counter_t
// {
//     std::atomic<double> avg_value{0};
//     std::atomic<uint64_t> update_count{0};
//     void add_value(uint64_t value)
//     {
//         uint64_t n = update_count.fetch_add(1, std::memory_order_relaxed);
//         double new_avg = (avg_value.load(std::memory_order_relaxed) * n + value) / (n + 1);
//         avg_value.store(new_avg, std::memory_order_relaxed);
//         update_count++;
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
