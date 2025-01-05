#pragma once
#include <boost/circular_buffer.hpp>
#include <vector>
#include <ratio>
// #include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <thread>
#include <array>
#include <atomic>

#define DEBUG
#ifdef DEBUG
#define D(x) std::cout << x
#define ND std::cout << std::endl
#define DN(x) std::cout << x << std::endl
#define DF(x) x
#else
#define D(x) ;
#define ND ;
#define DN(x) ;
#define DF(x) ;
#endif

// using view_dim = unsigned short;
using layer_dim_t = short; // 65535
using scene_signal_t = uint8_t;
using nof_neurons = long; // +-2,15E+09
using potential_t = double;
using weight_t = double;
using distance_t = unsigned char;
using clock_count_t = long long; // 1,84E+19,
                                 // negative value means the time have already been processed
constexpr size_t mnist_size = 28;
using scene_t = std::array<std::array<scene_signal_t, mnist_size>, mnist_size>;
using scenes_t = std::vector<scene_t>;

using teach_signal_t = uint16_t;

struct neuro_node_t;

struct neuron_address_t
{
    layer_dim_t layer;
    layer_dim_t row;
    layer_dim_t col;
    neuro_node_t &ref();
};

struct event_t
{
    neuron_address_t source_addr;
    layer_dim_t src_synapse; // synapse index
    neuron_address_t target_addr;
    clock_count_t time_of_arrival;
    int signal; // Only for detector
};

constexpr size_t events_cirular_buffer_size = 50;

namespace TNN
{
    enum Devices
    {
        IMITATION,
        CAMERA,
        EVENT_CAMERA,
        MICROPHONE
    };

    enum layer_type : short
    {
        NO_LAYER = 0,
        RETINA = 1,
        CORTEX = 2,
        COUCHING = 3,
        ACTUATOR = 4
    };

    enum ferment_t : clock_count_t
    {
        DOPHAMINE = 1L,
        GAMK = -1L
    };

}
std::ostream &operator<<(std::ostream &os, TNN::layer_type t);
std::istream &operator>>(std::istream &is, TNN::layer_type &t);

std::ostream &operator<<(std::ostream &os, TNN::ferment_t t);
std::istream &operator>>(std::istream &is, TNN::ferment_t &t);

struct conn_descr_t
{
    layer_dim_t src_layer;
    layer_dim_t trg_layer;
    // layer_dim_t nof_synapses; // Nof synapses
    TNN::ferment_t ferment; // ferment (signed time of dissolution)
    layer_dim_t radius;
    clock_count_t delay;
};

using conn_descr_coll_t = std::vector<conn_descr_t>;

class atomic_mutex
{
    std::atomic_flag m_ = ATOMIC_FLAG_INIT;

public:
    void lock() noexcept
    {
        while (m_.test_and_set(std::memory_order_acquire))
            m_.wait(true, std::memory_order_relaxed);
    }
    bool try_lock() noexcept
    {
        return !m_.test_and_set(std::memory_order_acquire);
    }
    void unlock() noexcept
    {
        m_.clear(std::memory_order_release);
        m_.notify_all();
    }
};

// max nof items in the events buffer
constexpr layer_dim_t time_steps = 10000;

struct net_timer_t
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    net_timer_t() : start(std::chrono::high_resolution_clock::now()) {}
    clock_count_t time()
    {
        return std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    layer_dim_t time_index(clock_count_t cur_time = 0)
    {
        if (!cur_time)
            cur_time = time();
        return static_cast<layer_dim_t>(cur_time % time_steps);
    }
};



void print_couch();
inline unsigned nof_event_threads = 0;
