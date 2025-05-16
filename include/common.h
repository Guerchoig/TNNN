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

template <typename T>
using vector_2D_t = std::vector<std::vector<T>>;

constexpr size_t mnist_size = 28;

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

using scene_t = std::array<std::array<scene_signal_t, mnist_size>, mnist_size>;


using teach_signal_t = uint16_t;

struct neuron_t;

/**
 * @brief Represents a 3D address with layer, row, and column coordinates
 */
struct address_t
{
    layer_dim_t layer;
    layer_dim_t row;
    layer_dim_t col;

    bool operator<(const address_t &other) const
    {
        return std::tie(layer, row, col) < std::tie(other.layer, other.row, other.col);
    }

    bool operator==(const address_t &other) const
    {
        return std::tie(layer, row, col) == std::tie(other.layer, other.row, other.col);
    }

    address_t() {}
    address_t(layer_dim_t l,
              layer_dim_t r,
              layer_dim_t c) : layer(l), row(r), col(c) {}
};

template <>
struct std::hash<address_t>
{
    /*************  ✨ Codeium Command ⭐  *************/
    /**
     * @brief Hash function for address_t that combines the hashes of the layer, row, and column coordinates.
     *
     * This function computes a hash value for a given address_t object by using a combination of prime number
     * multiplication and the built-in hash function for each coordinate (layer, row, col). The use of different
     * prime numbers ensures a more uniform distribution of hash values, reducing the likelihood of collisions.
     *
     * @param n The address_t object to be hashed.
     * @return size_t The computed hash value for the given address.
     */

    /******  be260060-316d-4383-ac5b-0c49c7e0c604  *******/
    size_t operator()(const address_t &n) const noexcept
    {
        const size_t prime = 31;
        size_t result = 17; // initial value

        // Combine hashes of layer, row, and col
        result = result * prime + hash<layer_dim_t>{}(n.layer);
        result = result * prime + hash<layer_dim_t>{}(n.row);
        result = result * prime + hash<layer_dim_t>{}(n.col);

        return result;
    }
};

struct neuron_address_t : address_t
{
    using address_t::address_t;
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
        COUCHING = 3,
        ACTUATOR = 4
    };

    enum ferment_t : clock_count_t
    {
        DOPHAMINE = 1L,
        GAMK = -1L
    };

}

struct neuron_event_t
{
    neuron_address_t source_addr;
    layer_dim_t src_synapse; // synapse index
    neuron_address_t target_addr;
    clock_count_t time_of_arrival;
    TNN::ferment_t ferment;
    int signal; // Only for detector
    neuron_event_t() {}
    neuron_event_t(neuron_address_t source_addr,
                   layer_dim_t src_synapse,
                   neuron_address_t target_addr,
                   clock_count_t time_of_arrival,
                   TNN::ferment_t ferment,
                   int signal) : source_addr(source_addr),
                                 src_synapse(src_synapse),
                                 target_addr(target_addr),
                                 time_of_arrival(time_of_arrival),
                                 ferment(ferment), signal(signal) {}
};

struct weight_event_t
{
    neuron_address_t addr;
    layer_dim_t synapse_num;
    clock_count_t spike_time;
    weight_event_t() {}
    weight_event_t(neuron_address_t addr,
                   layer_dim_t synapse_num,
                   clock_count_t spike_time) : addr(addr),
                                               synapse_num(synapse_num),
                                               spike_time(spike_time) {}
};

// Workers's types and params -----------------------------------------------

constexpr uint32_t events_q_size = 1024;
constexpr uint32_t weigths_q_size = 1024;

using events_pack_t = std::vector<neuron_event_t>;
using weights_pack_t = std::vector<weight_event_t>;

using events_output_buf_t = std::unordered_map<address_t, std::unique_ptr<events_pack_t>>;
using weights_output_buf_t = std::unordered_map<address_t, std::unique_ptr<weights_pack_t>>;

// Tracer interface types -----------------------------------------------
using tracer_buf_t = std::vector<std::pair<neuron_address_t, std::uint8_t>>;

// std::ostream &operator<<(std::ostream &os, TNN::layer_type t);
// std::istream &operator>>(std::istream &is, TNN::layer_type &t);

// std::ostream &operator<<(std::ostream &os, TNN::ferment_t t);
// std::istream &operator>>(std::istream &is, TNN::ferment_t &t);

struct conn_descr_t
{
    layer_dim_t src_layer;
    layer_dim_t trg_layer;
    TNN::ferment_t ferment; // ferment (signed time of dissolution)
    layer_dim_t radius;
    clock_count_t delay;
};

using conn_descr_coll_t = std::vector<conn_descr_t>;

// max nof items in the events buffer
constexpr layer_dim_t time_steps = 10000;

struct net_timer_t
{
    std::chrono::time_point<std::chrono::high_resolution_clock> work;
    net_timer_t() : work(std::chrono::high_resolution_clock::now()) {}
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

class head_interface_t
{
public:
    virtual void clear_scene_memory() = 0;
};

class ptracer_interface_t
{
public:
    // virtual void set_scene_index(layer_dim_t index) = 0;
    virtual void display_tracer_buf(std::shared_ptr<tracer_buf_t> item) = 0;
};

// void print_couch();
inline unsigned nof_event_threads = 0;
