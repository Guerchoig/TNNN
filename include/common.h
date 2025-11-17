#pragma once
// #include <boost/circular_buffer.hpp>
#include "logger.h"
#include <memory>
#include <vector>
#include <ratio>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <thread>
#include <array>
#include <atomic>
#include <tuple>
#include <stdint.h>

// #define DEBUG_TRACER

#include "tracer_macros.h"

// Mode of operation parameters
// ----------------------------------------------

// #define READ_NET_FROM_FILE
// #define DEBUG
// -----

#ifdef DEBUG
#define D(x) std::cout << x
#define DN(x) std::cout << x << std::endl
#define DF DN(__PRETTY_FUNCTION__)
#else
#define D(x) ;
#define DN(x) ;
#define DF ;
#endif

// using view_dim = unsigned short;
using brain_coord_t = uint32_t;
using scene_signal_t = uint8_t;
using nof_neurons = long; // +-2,15E+09
using potential_t = double;
using weight_t = double;

using clock_count_t = long long; // 1,84E+19,
// negative value means the time have already been processed

constexpr brain_coord_t mnist_size = 28;
constexpr brain_coord_t mnist_len = mnist_size * mnist_size;

// learning params
// constexpr int image_show_delay = 100; // ms
constexpr int nof_images_in_learning_set = 100;
constexpr int nof_images_in_test_set = 5;
constexpr uint32_t learning_epoques = 10;
constexpr int iterations_per_image = 15;

// Tracer params
#define tracer_period 1;

// Logger params
constexpr int weights_output_precision = 2;
inline text_logger logger{"../logfile.txt "};

// Signum function
template <typename T>
int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

class atomic_mutex
{
    std::atomic_flag flag = ATOMIC_FLAG_INIT;

public:
    void lock()
    {
        while (flag.test_and_set(std::memory_order_acquire))
        {
            // spin-wait (busy waiting) until lock is acquired
        }
    }
    void unlock()
    {
        flag.clear(std::memory_order_release);
    }
};

/**
 * @brief Represents a color with red, green, blue, and alpha (transparency) components
 */
struct rgba_t
{
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
    std::uint8_t a;
};

using scene_t = std::array<scene_signal_t, mnist_len>;

using teach_signal_t = uint16_t;

struct neuron_t;

struct abs_address_t
{
    brain_coord_t layer = 0;
    brain_coord_t abs_row = 0;
    brain_coord_t abs_col = 0;
};

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
        OUTPUT = 3,
        REFERENCE = 4,
        ACTUATOR = 5
    };

    enum ferment_t : clock_count_t
    {
        GLUTAMATE = 1L,
        GABA = -1L
    };

    enum connection_type
    {
        FULLY_CONNECTED,
        ONE_TO_ONE
    };

}

// Tracer interface types -----------------------------------------------
using tracer_buf_t = std::vector<std::pair<abs_address_t, std::uint8_t>>;

struct conn_descr_t
{
    brain_coord_t src_layer;
    brain_coord_t trg_layer;
    TNN::ferment_t ferment; // ferment (signed time of dissolution)
    TNN::connection_type connection_type;
    float linear_density;
};

using conn_descr_coll_t = std::vector<conn_descr_t>;

class net_timer_t
{
private:
    std::atomic<clock_count_t> time_counter = 0;

public:
    clock_count_t time_and_inc() { return time_counter.fetch_add(1) + 1; }
    clock_count_t just_time() { return time_counter.load(); }
    net_timer_t() : time_counter(0) {}
};

inline unsigned nof_event_threads = 0;
