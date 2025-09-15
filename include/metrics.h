#pragma once

#include <array>
#include <atomic>
#include <iostream>
struct metrics_t
{
    enum results_t
    {
        NOF_ATTEMPTS = 0,
        TOTAL_SPIKES = 1,
        LABELED_SPIKES = 2
    };
    std::array<std::atomic<uint64_t>, 4> results = {0, 0, 0, 0};

    void store_one_metric(const results_t &res)
    {
        results[res].fetch_add(1, std::memory_order_relaxed);
    }

    void print_metrics()
    {
        long nof_attempts = results[results_t::NOF_ATTEMPTS];
        long total_spikes = results[results_t::TOTAL_SPIKES];
        long labeled_spikes = results[results_t::LABELED_SPIKES];

        auto precision = nof_attempts == 0 ? 0 : labeled_spikes / nof_attempts;

        std::cout << " Precision: " << precision
                  << " Nof attempts: " << nof_attempts
                  << " Labeled spikes: " << labeled_spikes
                  << " Total spikes: " << total_spikes
                  << std::endl;
    }
    void reset()
    {
        for (size_t it = 0; it < results.size(); ++it)
            results[it].store(0);
    }
};