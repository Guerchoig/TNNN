#pragma once
#include "common.h"
#include "brain.h"
#include <cstdarg>
#include <utility>
#include <tuple>
#include <iostream>
#include <vector>
#include <memory>

constexpr char data_dir[] = "../data/";

struct rc_t
{
    brain_coord_t nof_rows;
    brain_coord_t nof_cols;
};

struct syn_descr_t
{
    clock_count_t last_fired;
    weight_t weight;
    TNN::ferment_t ferment;
    clock_count_t delay;
    neuron_address_t target_addr;
};

struct neuron_descr_t
{
    potential_t u_mem;
    potential_t threshold;
    clock_count_t last_fired;
    brain_coord_t nof_synapses;
    std::vector<syn_descr_t> synapses;
};

struct layer_descr_t
{
    TNN::layer_type type;
    rc_t dimensions;
};

struct network_descr_t
{
    brain_coord_t nof_layers;
    std::vector<layer_descr_t> layers_descriptions;
    conn_descr_coll_t conn_descriptions;
    network_descr_t() {};
    network_descr_t(std::vector<layer_descr_t> layers_descriptions,
                    conn_descr_coll_t conn_descriptions) : nof_layers(layers_descriptions.size()),
                                                           layers_descriptions(layers_descriptions),
                                                           conn_descriptions(conn_descriptions) {}
};

void create_net(head_t *phead, const network_descr_t &dsc);

std::ostream &operator<<(std::ostream &os, const synapse_t &s);
std::istream &operator>>(std::istream &is, synapse_t &s);

std::ostream &operator<<(std::ostream &os, const neuron_t &s);
std::istream &operator>>(std::istream &is, neuron_t &s);

std::ostream &operator<<(std::ostream &os, const layer_t &layer);
std::istream &operator>>(std::istream &is, layer_t &layer);

std::ostream &operator<<(std::ostream &os, const head_t &s);
std::istream &operator>>(std::istream &is, head_t &s);

std::ostream &operator<<(std::ostream &os, const conn_descr_t &connection);
std::istream &operator>>(std::istream &is, conn_descr_t &connection);

std::ostream &operator<<(std::ostream &os, const conn_descr_coll_t &connections);
std::istream &operator>>(std::istream &is, conn_descr_coll_t &connections);

std::ostream &operator<<(std::ostream &os, const layer_descr_t &dsc);
std::istream &operator>>(std::istream &is, layer_descr_t &dsc);

std::ostream &operator<<(std::ostream &os, const std::vector<layer_descr_t> &dsc);
std::istream &operator>>(std::istream &is, std::vector<layer_descr_t> &dsc);

std::ostream &operator<<(std::ostream &os, const network_descr_t &dsc);
std::istream &operator>>(std::istream &is, network_descr_t &dsc);

std::shared_ptr<layer_t> create_layer_neurons(head_t *phead, TNN::layer_type ltype, layer_place_n_size_t place_n_size);
