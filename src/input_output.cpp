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
#include <cmath>
using namespace TNN;

constexpr auto sqr(auto &&x) noexcept(noexcept(x * x)) -> decltype(x * x)
{
    return x * x;
}
double calc_delay(neuron_address_t src,
                  neuron_address_t trg)
{
    clock_count_t res = round(spike_velocity * std::sqrt(sqr(src.layer - trg.layer) + sqr(src.row - trg.row) + sqr(src.col - trg.col)));
    return res;
}

void create_synapses_between_2_layers(head_t *phead,
                                      neuron_address_t src,
                                      neuron_address_t trg,
                                      TNN::ferment_t ferment,
                                      layer_dim_t radius)
{
    auto &synapses = phead->neuron_ref(src).synapses;

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
                synapses.emplace_back(/*0L, */ w(gen), ferment, std::move(adr));
            }
        }
    }
}

void create_connections(head_t *phead, const conn_descr_coll_t &descriptions)
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

                create_synapses_between_2_layers(phead, {dsc->src_layer, src_row, src_col},
                                                 {dsc->trg_layer, trg_row, trg_col},
                                                 dsc->ferment,
                                                 dsc->radius);
            }
        }
    }
}

void create_layers(head_t *phead, const network_descr_t &dsc)
{
    for (layer_dim_t i = 0; i < dsc.nof_layers; ++i)
    {
        auto l = create_layer_neurons(phead, dsc.layers_descriptions[i].type,
                                      dsc.layers_descriptions[i].dimensions.nof_rows,
                                      dsc.layers_descriptions[i].dimensions.nof_cols);
    }
}

/**
 * @brief Creates net
 * @param dsc net description
 */
void create_net(head_t *phead, const network_descr_t &dsc)
{
    create_layers(phead, dsc);
    phead->pretina = static_cast<retina_layer_t *>(phead->layers[0].get());
    phead->pretina->p_eyes_optics = phead->p_eyes_optics;
    create_connections(phead, dsc.conn_descriptions);
    phead->connections = std::move(dsc.conn_descriptions);
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
        std::getline(is, layer_type_string, ' ');
    auto n = layer_type_string.find_first_of("RC");
    if (n != std::string::npos)
        layer_type_string = layer_type_string.substr(n);
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

    os << "w: " << s.weight << " f: " << s.ferment << " taddr: " << s.target_addr << std::endl;
    return os;
}

std::istream &operator>>(std::istream &is, synapse_t &s)
{
    std::string dumm_str;
    is >> dumm_str >> s.weight   //
        >> dumm_str >> s.ferment //
        >> dumm_str >> s.target_addr;
    return is;
}

std::ostream &operator<<(std::ostream &os, const neuron_t &s)
{
    os << "umem: " << s.u_mem << " threshold: " << s.threshold << " lastfired: " << s.last_fired;
    os << " synum: " << s.synapses.size() << std::endl;
    for (auto syn : s.synapses)
        os << syn << " ";
    return os;
}

std::istream &operator>>(std::istream &is, neuron_t &s)
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
    os << types[layer.ltype] << std::endl
       << "ROWS: " << layer.neurons.size() << " "; // type & rows
    for (unsigned i = 0;
         i < layer.neurons.size();
         ++i)
    {
        os << "COLS: " << layer.neurons[i].size() << std::endl;
        for (unsigned j = 0;
             j < layer.neurons[i].size(); ++j)
        {
            os << " " << i << " " << j << " " << std::remove_cv_t<layer_t &>(layer).neuron_ref(i, j) << std::endl;
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
            layer.neurons.back().emplace_back();
            is >> layer.neuron_ref(i, j);
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
std::shared_ptr<layer_t> emplace_layer_create_neurons(head_t *phead, layer_dim_t rows, layer_dim_t cols)
{
    return phead->layers.emplace_back(std::make_shared<T>(rows, cols));
}

std::shared_ptr<layer_t> create_layer_neurons(head_t *phead, TNN::layer_type ltype, layer_dim_t rows, layer_dim_t cols)
{
    std::shared_ptr<layer_t> l;
    switch ((int)ltype)
    {
    case TNN::RETINA:
        l = emplace_layer_create_neurons<retina_layer_t>(phead, rows, cols);
        break;
    case TNN::CORTEX:
        l = emplace_layer_create_neurons<cortex_layer_t>(phead, rows, cols);
        break;
    case TNN::COUCHING:
        l = emplace_layer_create_neurons<mnist_couch_layer_t>(phead, rows, cols);
        break;
    // case TNN::ACTUATOR:
    //     l = emplace_layer_create_neurons<actuator_layer_t>(rows, cols);
    //     break;
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

        std::shared_ptr<layer_t> p_layer;
        std::shared_ptr<retina_layer_t> rl;
        switch ((int)type)
        {
        case TNN::RETINA:
            h.layers.emplace_back(std::make_shared<retina_layer_t>());
            p_layer = h.layers.back();
            rl = std::static_pointer_cast<retina_layer_t>(p_layer);
            is >> *(rl);
            h.pretina = rl.get();
            break;
        case TNN::CORTEX:
            p_layer = h.layers.emplace_back(std::make_shared<cortex_layer_t>());
            is >> *(std::static_pointer_cast<cortex_layer_t>(p_layer));
            break;
        case TNN::COUCHING:
            p_layer = h.layers.emplace_back(std::make_shared<mnist_couch_layer_t>());
            is >> *(std::static_pointer_cast<mnist_couch_layer_t>(p_layer));
            break;
        // case TNN::ACTUATOR:
        //     l = h.layers.emplace_back(std::make_shared<actuator_layer_t>(type));
        //     break;
        default:
            break;
        }
    }
    return is;
}

std::ostream &operator<<(std::ostream &os, const conn_descr_t &connection)
{
    os << connection.src_layer << " " << connection.trg_layer << " " << connection.ferment << " " << connection.radius << std::endl;
    return os;
}

std::istream &operator>>(std::istream &is, conn_descr_t &connection)
{
    is >> connection.src_layer >> connection.trg_layer >> connection.ferment >> connection.radius;
    return is;
}

std::ostream &operator<<(std::ostream &os, const conn_descr_coll_t &connections)
{
    os << connections.size() << std::endl;
    for (auto it = connections.begin(); it != connections.end(); ++it)
    {
        os << *it;
    }
    return os;
}

std::istream &operator>>(std::istream &is, conn_descr_coll_t &connections)
{
    layer_dim_t nof_connections;
    is >> nof_connections;
    connections.reserve(nof_connections);
    for (layer_dim_t i = 0; i < nof_connections; ++i)
    {
        connections.emplace_back();
        is >> connections.back();
    }
    return is;
}

std::ostream &operator<<(std::ostream &os, const layer_descr_t &dsc)
{
    const char *types[5]{"NOLAYER\0", "RETINA\0", "CORTEX\0", "COUCHING\0", "ACTUATOR\0"};
    os << types[dsc.type] << " " << dsc.dimensions.nof_rows << " " << dsc.dimensions.nof_cols << std::endl;
    return os;
}

std::istream &operator>>(std::istream &is, layer_descr_t &dsc)
{
    is >> dsc.type >> dsc.dimensions.nof_rows >> dsc.dimensions.nof_cols;
    return is;
}

std::ostream &operator<<(std::ostream &os, const std::vector<layer_descr_t> &dsc)
{
    os << dsc.size() << std::endl;
    for (auto it = dsc.begin(); it != dsc.end(); ++it)
    {
        os << *it;
    }
    return os;
}

std::istream &operator>>(std::istream &is, std::vector<layer_descr_t> &dsc)
{
    layer_dim_t nof_layers;
    is >> nof_layers;
    dsc.reserve(nof_layers);
    for (layer_dim_t i = 0; i < nof_layers; ++i)
    {
        dsc.emplace_back();
        is >> dsc.back();
    }
    return is;
}

std::ostream &operator<<(std::ostream &os, const network_descr_t &dsc)
{
    os << dsc.layers_descriptions;
    os << dsc.conn_descriptions << std::endl;
    return os;
}

std::istream &operator>>(std::istream &is, network_descr_t &dsc)
{
    is >> dsc.layers_descriptions;
    dsc.nof_layers = dsc.layers_descriptions.size();
    is >> dsc.conn_descriptions;
    return is;
}

void head_t::save_model_to_file(std::string file_name, [[maybe_unused]] std::shared_ptr<tracer_t> ptracer)
{
    try
    {
#ifdef TRACER_DEBUG
        ptracer->lock_screen();
#endif

        std::ofstream ofs(file_name);

        // Make layers' description
        std::vector<layer_descr_t> layers_descriptions;
        for (auto it = layers.begin(); it != layers.end(); ++it)
        {
            layers_descriptions.emplace_back((*it)->ltype, rc_t((*it)->neurons.size(), (*it)->neurons[0].size()));
        }

        // Make network's description
        network_descr_t network_dsc(layers_descriptions, connections);
        ofs << network_dsc;

        ofs.close();
#ifdef TRACER_DEBUG
        ptracer->unlock_screen();
#endif
    }
    catch (...)
    {
        std::cout << "Error saving weights" << std::endl;
    }
}

void head_t::read_model_from_file(std::string file_name, [[maybe_unused]] std::shared_ptr<tracer_t> ptracer)
{
    try
    {
        std::ifstream ifs(file_name);

        network_descr_t network_dsc;
        ifs >> network_dsc;

        create_net(this, network_dsc);
        ifs.close();
    }
    catch (...)
    {
        std::cout << "Error reading weights" << std::endl;
    }
}