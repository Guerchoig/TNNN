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
using namespace params;

constexpr auto
sqr(auto &&x) noexcept(noexcept(x * x)) -> decltype(x * x)
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
                                      brain_coord_t radius)
{
    auto &forward_synapses = phead->neuron_ref(src).synapses;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> w(0, 1);

    auto trg_max_row = static_cast<brain_coord_t>(phead->layers[trg.layer]->neurons.size() - 1);
    auto trg_max_col = static_cast<brain_coord_t>(phead->layers[trg.layer]->neurons[0].size() - 1);

    // radius is in terms of big layer
    for (brain_coord_t drow = -radius; drow <= radius; ++drow)
    {
        brain_coord_t _row = trg.row + drow;
        if (_row < 0 || _row > trg_max_row)
            continue;
        for (brain_coord_t dcol = -radius; dcol <= radius; ++dcol)
        {
            brain_coord_t _col = trg.col + dcol;
            if (_col < 0 || _col > trg_max_col)
                continue;
            if (std::make_tuple(src.layer, src.row, src.col) != std::make_tuple(trg.layer, _row, _col))
            {
                neuron_address_t adr{trg.layer, _row, _col};
                forward_synapses.emplace_back(w(gen), ferment, std::move(adr));

                // Couching layer synapses mirror input ones as if there were backward connections there
                // so input sources become couching synapses targets
                // the weights of those synapses are not used
                // input synapse number is put into the weight field
                if (phead->layers[trg.layer]->ltype == TNN::layer_type::COUCHING)
                {
                    neuron_address_t src_adr{src.layer, src.row, src.col};

                    phead->neuron_ref(trg).synapses.emplace_back(forward_synapses.size() - 1, ferment, std::move(src_adr));
                }
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
            auto src_row = static_cast<brain_coord_t>(it_row - src_neurons.begin());
            auto trg_row = static_cast<brain_coord_t>(std::round(src_row * row_ratio));

            for (auto it_col = it_row->begin(); it_col != it_row->end(); ++it_col)
            {
                auto src_col = static_cast<brain_coord_t>(it_col - it_row->begin());
                auto trg_col = static_cast<brain_coord_t>(std::round(src_col * col_ratio));

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
    for (brain_coord_t i = 0; i < dsc.nof_layers; ++i)
    {
        auto l = create_layer_neurons(phead, dsc.layers_descriptions[i].type,
                                      layer_place_n_size_t(i,
                                                           dsc.layers_descriptions[i].dimensions.nof_rows,
                                                           dsc.layers_descriptions[i].dimensions.nof_cols));
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

    os << s.weight << " " << s.ferment << std::endl;
    return os;
}

std::istream &operator>>(std::istream &is, synapse_t &s)
{
    // std::string dumm_str;
    is >> s.weight >> s.ferment;
    return is;
}

std::ostream &operator<<(std::ostream &os, const neuron_t &s)
{
    os << s.u_mem << std::endl;
    for (auto syn : s.synapses)
        os << syn << " ";
    return os;
}

std::istream &operator>>(std::istream &is, neuron_t &s)
{
    is >> s.u_mem;

    for (brain_coord_t i = 0; static_cast<size_t>(i) < s.synapses.size(); ++i)
    {
        is >> s.synapses[i];
    }
    return is;
}

std::ostream &operator<<(std::ostream &os, const layer_t &layer)
{
    for (unsigned i = 0; i < layer.neurons.size(); ++i)
    {
        for (unsigned j = 0; j < layer.neurons[i].size(); ++j)
        {
            os << std::remove_cv_t<layer_t &>(layer).neuron_ref(i, j) << std::endl;
        }
    }
    return os;
}

template <Is_layer T>
std::istream &operator>>(std::istream &is, T &layer)
{

    for (auto it = layer.neurons.begin(); it != layer.neurons.end(); ++it)
        for (auto jt = it->begin(); jt != it->end(); ++jt)
            is >> layer.neuron_ref(it - layer.neurons.begin(), jt - it->begin());

    return is;
}

std::ostream &operator<<(std::ostream &os, const head_t &h)
{
    for (auto it = h.layers.begin(); it != h.layers.end(); ++it)
    {

        auto &layer = *it;
        os << (*layer) << std::endl;
    }
    return os;
}

std::istream &operator>>(std::istream &is, head_t &h)
{
    for (auto it = h.layers.begin(); it != h.layers.end(); ++it)
    {
        auto _type = (*it)->ltype;
        switch (_type)
        {
        case TNN::NO_LAYER:
        case TNN::ACTUATOR:
            break;
        case TNN::RETINA:
            is >> static_cast<retina_layer_t &>(*(*it));
            break;
        case TNN::CORTEX:
            is >> static_cast<cortex_layer_t &>(*(*it));
            break;
        case TNN::COUCHING:
            is >> static_cast<couching_layer_t &>(*(*it));
            break;
        }
    }
    return is;
}

template <typename T>
std::shared_ptr<layer_t> emplace_layer_create_neurons(head_t *phead, layer_place_n_size_t place_n_size)
{
    return phead->layers.emplace_back(std::make_shared<T>(place_n_size));
}

std::shared_ptr<layer_t> create_layer_neurons(head_t *phead, TNN::layer_type ltype, layer_place_n_size_t place_n_size)
{
    std::shared_ptr<layer_t> l;
    switch ((int)ltype)
    {
    case TNN::RETINA:
        l = emplace_layer_create_neurons<retina_layer_t>(phead, place_n_size);
        break;
    case TNN::CORTEX:
        l = emplace_layer_create_neurons<cortex_layer_t>(phead, place_n_size);
        break;
    case TNN::COUCHING:
        l = emplace_layer_create_neurons<couching_layer_t>(phead, place_n_size);
        break;
    // case TNN::ACTUATOR:
    //     l = emplace_layer_create_neurons<actuator_layer_t>(rows, cols);
    //     break;
    default:
        break;
    }
    return l;
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
    brain_coord_t nof_connections;
    is >> nof_connections;
    connections.reserve(nof_connections);
    for (brain_coord_t i = 0; i < nof_connections; ++i)
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
    brain_coord_t nof_layers;
    is >> nof_layers;
    dsc.reserve(nof_layers);
    for (brain_coord_t i = 0; i < nof_layers; ++i)
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

        ofs << *this << std::endl;

        ofs.close();

#ifdef TRACER_DEBUG
        ptracer->unlock_screen();
#endif
    }
    catch (...)
    {
        std::cout << "Error saving model" << std::endl;
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

        ifs >> *this;

        ifs.close();
    }
    catch (...)
    {
        std::cout << "Error reading weights" << std::endl;
    }
}