#include "input_output.h"
#include "brain.h"
#include <utility>
#include <tuple>
#include <vector>
#include <memory>
#include <fstream>
#include <cmath>
#include <unordered_map>
#include <string>
using namespace TNN;

void create_synapses_between_2_layers(neuron_address_t src,
                                      neuron_address_t trg,
                                      TNN::ferment_t ferment,
                                      layer_dim_t radius,
                                      [[maybe_unused]] clock_count_t delay)
{
    auto &synapses = src.ref().synapses;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> w(0, 1);

    auto trg_max_row = static_cast<layer_dim_t>(phead->layers[trg.layer]->neurons.size() - 1);
    auto trg_max_col = static_cast<layer_dim_t>(phead->layers[trg.layer]->neurons[0].size() - 1);

    // radius is in terms of big layer
    for (layer_dim_t drow = -radius; drow <= radius; ++drow)
    {
        layer_dim_t _row = trg.row + drow;
        if (_row < 0 || _row > trg_max_row)
            continue;
        for (layer_dim_t dcol = -radius; dcol <= radius; ++dcol)
        {
            layer_dim_t _col = trg.col + dcol;
            if (_col < 0 || _col > trg_max_col)
                continue;
            if (std::make_tuple(src.layer, src.row, src.col) != std::make_tuple(trg.layer, _row, _col))
            {
                neuron_address_t adr{trg.layer, _row, _col};
                synapses.emplace_back(/*0L, */ w(gen), ferment, delay, std::move(adr));
            }
        }
    }
}

void create_connections(const conn_descr_coll_t &descriptions)
{
    //<nof_cons, ferment, delay>;
    for (auto dsc = descriptions.begin(); dsc != descriptions.end(); ++dsc)
    {
        auto &src_neurons = phead->layers[dsc->src_layer]->neurons;
        auto &trg_neurons = phead->layers[dsc->trg_layer]->neurons;

        double row_ratio = 0;
        double col_ratio = 0;

        if (src_neurons.size() == 1 && trg_neurons.size() == 1)
            row_ratio = 1;
        else
            row_ratio = static_cast<double>(trg_neurons.size() - 1) / (src_neurons.size() - 1);

        if (src_neurons[0].size() == 1 && trg_neurons[0].size() == 1)
            col_ratio = 1;
        else
            col_ratio = static_cast<double>(trg_neurons[0].size() - 1) / (src_neurons[0].size() - 1);

        for (auto it_row = src_neurons.begin(); it_row != src_neurons.end(); ++it_row)
        {
            auto src_row = static_cast<layer_dim_t>(it_row - src_neurons.begin());
            auto trg_row = static_cast<layer_dim_t>(std::round(src_row * row_ratio));

            for (auto it_col = it_row->begin(); it_col != it_row->end(); ++it_col)
            {
                auto src_col = static_cast<layer_dim_t>(it_col - it_row->begin());
                auto trg_col = static_cast<layer_dim_t>(std::round(src_col * col_ratio));
                create_synapses_between_2_layers({dsc->src_layer, src_row, src_col},
                                                 {dsc->trg_layer, trg_row, trg_col},
                                                 dsc->ferment,
                                                 dsc->radius,
                                                 dsc->delay);
            }
        }
    }
}

void create_layers(const network_descr_t &dsc)
{
    for (layer_dim_t i = 0; i < dsc.nof_layers; ++i)
    {
        auto l = create_layer_neurons(dsc.layers_descriptions[i].type,
                                      dsc.layers_descriptions[i].dimensions.nof_rows,
                                      dsc.layers_descriptions[i].dimensions.nof_cols);
    }
}

/**
 * @brief Creates net
 * @param dsc net description
 */
void create_net(const network_descr_t &dsc)
{
    create_layers(dsc);
    phead->pretina = static_cast<retina_layer_t *>(phead->layers[0].get());
    create_connections(dsc.conn_descriptions);
}

std::ostream &operator<<(std::ostream &os, const neuron_address_t &s)
{
    os << s.layer << " " << s.row << " " << s.col << " ";
    return os;
}
std::istream &operator>>(std::istream &is, neuron_address_t &s)
{
    is >> s.layer >> s.row >> s.col;
    return is;
}

std::istream &operator>>(std::istream &is, layer_type &l_type)
{
    const std::unordered_map<std::string, layer_type> m{{"NOLAYER", TNN::NO_LAYER}, {"RETINA", TNN::RETINA}, {"CORTEX", TNN::CORTEX}, {"COUCHING", TNN::COUCHING}};
    std::string layer_type_string;

    while (layer_type_string.empty() || layer_type_string.compare(" ") == 0) // skip newlines
        std::getline(is, layer_type_string);

    l_type = m.at(layer_type_string);
    return is;
}

std::ostream &operator<<(std::ostream &os, TNN::ferment_t f)
{
    return os << (clock_count_t)f;
}
std::istream &operator>>(std::istream &is, TNN::ferment_t &f)
{
    return is >> (clock_count_t &)f;
}

std::ostream &operator<<(std::ostream &os, const synapse_t &s)
{

    os << "w: " << s.weight << " f: " << s.ferment << " dly: "
       << s.delay << " taddr: " << s.target_addr << std::endl;
    return os;
}

std::istream &operator>>(std::istream &is, synapse_t &s)
{
    std::string dumm_str;
    is >> dumm_str >> s.weight   //
        >> dumm_str >> s.ferment //
        >> dumm_str >> s.delay   //
        >> dumm_str >> s.target_addr;
    return is;
}

std::ostream &operator<<(std::ostream &os, const neuro_node_t &s)
{
    os << "umem: " << s.u_mem << " threshold: " << s.threshold << " lastfired: " << s.last_fired;
    os << " synum: " << s.synapses.size() << std::endl;
    for (auto syn : s.synapses)
        os << syn << " ";
    return os;
}

std::istream &operator>>(std::istream &is, neuro_node_t &s)
{
    int dumm;
    std::string dumm_str;
    layer_dim_t nof_synapses;
    // while (true)
    // {
    //     char ch =
    //         is.get(); // Skip newline
    //     (void)ch;
    // }
    is >> dumm >> dumm              //
        >> dumm_str >> s.u_mem      //
        >> dumm_str >> s.threshold  //
        >> dumm_str >> s.last_fired //
        >> dumm_str >> nof_synapses;
    s.synapses.reserve(nof_synapses);
    for (layer_dim_t i = 0; i < nof_synapses; ++i)
    {
        auto syn = s.synapses.emplace_back();
        is >> syn;
    }
    return is;
}

std::ostream &operator<<(std::ostream &os, const layer_t &layer)
{
    const char *types[5]{"NOLAYER\0", "RETINA\0", "CORTEX\0", "COUCHING\0", "ACTUATOR\0"};
    os << types[layer.type] << std::endl
       << "ROWS: " << layer.neurons.size() << " "; // type & rows
    for (unsigned i = 0;
         i < layer.neurons.size();
         ++i)
    {
        os << "COLS: " << layer.neurons[i].size() << std::endl;
        for (unsigned j = 0;
             j < layer.neurons[i].size(); ++j)
        {
            os << " " << i << " " << j << " " << std::remove_cv_t<layer_t &>(layer).node_ref(i, j) << std::endl;
        }
    }
    return os;
}

template <Is_layer T>
std::istream &operator>>(std::istream &is, T &layer)
{
    // is >> layer.type;
    layer_dim_t nof_rows;
    std::string dumm;
    is >> dumm;
    is >> nof_rows;
    layer.neurons.reserve(nof_rows);
    for (layer_dim_t i = 0; i < nof_rows; ++i)
    {
        layer_dim_t nof_cols;
        is >> dumm;
        is >> nof_cols;

        layer.neurons.emplace_back();
        layer.neurons.back().reserve(nof_cols);
        for (layer_dim_t j = 0; j < nof_cols; ++j)
        {
            layer.neurons_storage.emplace_back();
            if (layer.neurons_storage.empty())
                throw std::runtime_error("Couldn't emplace a neuron");
            layer.neurons.back().emplace_back();
            layer.neurons.back().back() = layer.neurons_storage.size() - 1;
            is >> layer.node_ref(i, j);
        }
    }
    return is;
}

std::ostream &operator<<(std::ostream &os, const head_t &h)
{
    os << h.layers.size() << std::endl;
    for (auto it = h.layers.begin(); it != h.layers.end(); ++it)
    {
        auto &layer = *it;
        os << (*layer) << std::endl;
    }
    return os;
}

template <typename T>
std::shared_ptr<layer_t> emplace_layer_create_neurons(layer_dim_t rows, layer_dim_t cols, TNN::layer_type _type)
{
    return phead->layers.emplace_back(std::make_shared<T>(rows, cols, _type));
}

std::shared_ptr<layer_t> create_layer_neurons(TNN::layer_type type, layer_dim_t rows, layer_dim_t cols)
{
    std::shared_ptr<layer_t> l;
    switch ((int)type)
    {
    case TNN::RETINA:
        l = emplace_layer_create_neurons<retina_layer_t>(rows, cols, type);
        break;
    case TNN::CORTEX:
        l = emplace_layer_create_neurons<cortex_layer_t>(rows, cols, type);
        break;
    case TNN::COUCHING:
        l = emplace_layer_create_neurons<mnist_couch_layer_t>(rows, cols, type);
        break;
    case TNN::ACTUATOR:
        l = emplace_layer_create_neurons<actuator_layer_t>(rows, cols, type);
        break;
    default:
        break;
    }
    return l;
}

std::istream &operator>>(std::istream &is, head_t &h)
{
    layer_dim_t nof_layers;
    is >> nof_layers;
    h.layers.reserve(nof_layers);

    for (layer_dim_t i = 0; i < nof_layers; ++i)
    {
        TNN::layer_type type;
        is >> type;

        std::shared_ptr<layer_t> l;
        switch ((int)type)
        {
        case TNN::RETINA:
            l = h.layers.emplace_back(std::make_shared<retina_layer_t>(type)); // emplace_layer_create_neurons<retina_layer_t>(rows, cols, type);
            is >> *(std::dynamic_pointer_cast<retina_layer_t>(l));
            h.pretina = std::dynamic_pointer_cast<retina_layer_t>(l).get();
            break;
        case TNN::CORTEX:
            l = h.layers.emplace_back(std::make_shared<cortex_layer_t>(type));
            is >> *(std::dynamic_pointer_cast<cortex_layer_t>(l));
            break;
        case TNN::COUCHING:
            l = h.layers.emplace_back(std::make_shared<mnist_couch_layer_t>(type));
            is >> *(std::dynamic_pointer_cast<mnist_couch_layer_t>(l));
            break;
        case TNN::ACTUATOR:
            l = h.layers.emplace_back(std::make_shared<actuator_layer_t>(type));
            break;
        default:
            break;
        }
    }
    return is;
}
