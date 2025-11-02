#pragma once

#include <array>
#include <atomic>
#include <iostream>
struct metrics_t
{
    enum results_t
    {
        UNLABELED_SPIKES = 0,
        LABELED_SPIKES,
        NO_SPIKES
    };
    std::array<std::atomic<uint64_t>, 4> results = {0, 0, 0, 0};

    void store_one_metric(const results_t &res)
    {
        results[res].fetch_add(1, std::memory_order_relaxed);
    }

    void print_metrics()
    {
        long no_spikes = results[results_t::NO_SPIKES];
        long unlabeled_spikes = results[results_t::UNLABELED_SPIKES];
        long labeled_spikes = results[results_t::LABELED_SPIKES];
        auto total_spikes = unlabeled_spikes + labeled_spikes;
        auto nof_attempts = total_spikes + no_spikes;
        auto precision = total_spikes == 0 ? 0 : static_cast<float>(labeled_spikes) / static_cast<float>(total_spikes) * 100.0f;
        auto accuracy = nof_attempts == 0 ? 0 : static_cast<float>(labeled_spikes) / static_cast<float>(nof_attempts) * 100.0f;

        std::cout << " Accuracy: " << accuracy << "%"
                  << " Precision: " << precision << "%"
                  << " No spikes: " << no_spikes
                  << " Labeled spikes: " << labeled_spikes
                  << " Unlabeled spikes: " << unlabeled_spikes
                  << " Total spikes: " << total_spikes
                  << " Attempts: " << nof_attempts
                  << std::endl;
    }
    void reset()
    {
        for (size_t it = 0; it < results.size(); ++it)
            results[it].store(0);
    }
};