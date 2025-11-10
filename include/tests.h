// вставьте рядом с другими includes в src/test.cpp
#include "brain.h"
#include <iostream>
#include <bitset>

using namespace std;

void run_stdp_tests()
{
    cout << "=== STDP unit test ===\n";

    // Create neurons and a synapse
    neuron_t pre, post;
    synapse_t syn(0.5 /*weight*/, TNN::GLUTAMATE, 0, 1);

    // Helper to print bitset history (spiking_history is typedef std::bitset<spiking_history_len>)
    auto print_before = [&syn, &pre, &post](const std::string &history_label, const history_spikes_t &h)
    {
        cout << "Before weight: " << syn.get_weight() << "\n";
        cout << "pre.last_fired=" << pre.get_last_fired() << ", post.last_fired=" << post.get_last_fired() << "\n";
        cout << history_label << h.to_string() << " (bits: index 0 = LSB = most recent)\n";
    };

    // Scenario A: potentiation (pre fired earlier, post fired later)
    cout << "\n-- Scenario A: potentiation (pre before post) --\n";
    pre.set_last_fired(10);
    post.set_last_fired(12);

    history_spikes_t post_hist;
    post_hist = 0b00000011; // post had two recent spikes
    post.set_spiking_history(post_hist);
    print_before(" post history: ", post_hist);
    stdp_weight_update(pre, post, syn, post.get_last_fired());
    cout << "After weight:  " << syn.get_weight() << "\n";

    // Scenario B: depression (pre fired after post)
    cout << "\n-- Scenario B: depression (pre after post) --\n";
    syn.set_weight(0.5); // reset weight
    pre.set_last_fired(15);
    post.set_last_fired(12);

    history_spikes_t pre_hist;
    pre_hist.reset();
    // leave pre_hist all zeros to trigger "no pre spike" checks in negative branch
    pre.set_spiking_history(pre_hist);
    print_before(" pre history: ", pre_hist);
    stdp_weight_update(pre, post, syn, pre.get_last_fired());
    cout << "After weight:  " << syn.get_weight() << "\n";

    // Scenario C: potentiation (pre == post)
    cout << "\n-- Scenario C: potentiation (pre == post) --\n";
    syn.set_weight(0.5); // reset weight
    pre.set_last_fired(15);
    post.set_last_fired(15);
    pre.set_spiking_history(pre_hist);

    cout << "Before weight: " << syn.get_weight() << "\n";
    cout << "pre.last_fired=" << pre.get_last_fired() << ", post.last_fired=" << post.get_last_fired() << "\n";
    print_before(" pre history: ", pre_hist);
    stdp_weight_update(pre, post, syn, pre.get_last_fired());
    cout << "After weight:  " << syn.get_weight() << "\n";

    cout << "=== STDP test finished ===\n";
}