#include "input_output.h"
#include "brain.h"
#include "tracer.h"
#include <atomic_queue.h>
#include <iostream>
#include <vector>
#include <utility>
#include <cstdlib>
#include <random>
#include <atomic>
#include <type_traits>
#include <limits>
#include <csignal>
#include <algorithm>
#include <cmath>
#include <iomanip>

/**
 * @file brain.cpp
 * @brief Implementation of the neural network's core functionality
 *
 * This file implements a spiking neural network with following layer types:
 * - Retina layer: Handles input processing
 * - Cortex layer: Processes intermediate neural computations
 * - Couching layer: Handles output processing
 *
 * The network uses STDP (Spike-Timing-Dependent Plasticity) for learning
 * and implements parallel processing using a piece-based architecture.
 */

using namespace TNN;
using namespace params;

/**
 * Network Construction Functions
 * These functions handle the initialization and setup of the neural network
 */

/**
 * @brief Adds a new layer to the neural network
 *
 * @param ltype Type of layer (RETINA, CORTEX, or COUCHING)
 * @param neurons_rows Number of rows in the layer
 * @param neurons_cols Number of columns in the layer
 * @param first_neuron_index Starting index for neurons in this layer (updated by function)
 * @param nof_pieces Number of parallel processing pieces to split the layer into
 * @param phead Pointer to the head object
 * @param ptracer Debug tracer object (if DEBUG_TRACER is defined)
 *
 * Creates a new layer of specified type and dimensions, initializing all neurons
 * and dividing the layer into pieces for parallel processing.
 */
void head_t::add_layer(TNN::layer_type ltype, brain_coord_t neurons_rows, brain_coord_t neurons_cols,
                       brain_coord_t &first_neuron_index, brain_coord_t nof_pieces, std::shared_ptr<head_t> phead
#ifdef DEBUG_TRACER
                       ,
                       std::shared_ptr<tracer_t> ptracer
#endif
)
{
  std::unique_ptr<layer_t> layer;

  switch (ltype)
  {
  case TNN::RETINA:
    layer = std::make_unique<retina_layer_t>(ltype, neurons_rows, neurons_cols, first_neuron_index);
    break;
  case TNN::CORTEX:
    layer = std::make_unique<cortex_layer_t>(ltype, neurons_rows, neurons_cols, first_neuron_index);
    break;
  case TNN::COUCHING:
    layer = std::make_unique<couching_layer_t>(ltype, neurons_rows, neurons_cols, first_neuron_index);
    break;
  default:
    throw std::invalid_argument("Invalid layer type");
  }
  layers_descr.push_back(std::move(layer)); // layer is null now
  auto last_layer_index = layers_descr.size() - 1;
  layer_num_by_1st_neuron_index[first_neuron_index] = last_layer_index;
  auto layer_size = neurons_rows * neurons_cols;

  // Create neurons
  neurons.resize(first_neuron_index + layer_size);

  // Create pieces
  assert(!(layer_size % nof_pieces));

  auto piece_first_neuron_index = first_neuron_index;
  auto piece_size = layer_size / nof_pieces;
  for (brain_coord_t i = 0; i < nof_pieces; ++i)
  {
    pieces.emplace_back(ltype, last_layer_index,
                        piece_first_neuron_index,
                        piece_size, phead,
                        neurons
#ifdef DEBUG_TRACER
                        ,
                        ptracer
#endif
    );
    piece_first_neuron_index += piece_size;
  }

  // Update layer's first_neuron_index
  first_neuron_index += layer_size;
}

constexpr int out_of_range = -1;

/**
 * @brief Creates synaptic connections between two layers
 *
 * @param src_layer Source layer index
 * @param trg_layer Target layer index
 * @param ferment Neurotransmitter type for the connections
 * @param connection_type Type of connection pattern
 *
 * Establishes synaptic connections between neurons in the source and target layers.
 * Each connection is initialized with a random weight and the appropriate neurotransmitter type.
 * Both source and target neurons maintain lists of their synaptic connections.
 */
void head_t::add_connections(brain_coord_t src_layer, brain_coord_t trg_layer,
                             TNN::ferment_t ferment, TNN::connection_type connection_type)
{
  // Add connection description
  connections_descr.emplace_back(src_layer, trg_layer, ferment, connection_type);

  // Prepare random generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> weight(0.001, 1.0);

  for (brain_coord_t i = 0; i < layers_descr[src_layer]->get_rows(); ++i)
    for (brain_coord_t j = 0; j < layers_descr[src_layer]->get_cols(); ++j)
    {
      auto src_addr = neuron_index(abs_address_t(src_layer, i, j));
      auto &src_neuron = neuron_ref(src_addr);

      for (brain_coord_t k = 0; k < layers_descr[trg_layer]->get_rows(); ++k)
        for (brain_coord_t l = 0; l < layers_descr[trg_layer]->get_cols(); ++l)
        {
          auto trg_addr = neuron_index(abs_address_t(trg_layer, k, l));
          auto &trg_neuron = neuron_ref(trg_addr);

          // Create synapse
          synapses.emplace_back(weight(gen), ferment, src_addr, trg_addr);
          auto synapse_index = synapses.size() - 1;

          // Remember synapse index in neurons
          src_neuron.get_output_synapse_indexes().emplace_back(synapse_index);
          trg_neuron.get_input_synapse_indexes().emplace_back(synapse_index);
        }
    }
}

piece_t::piece_t(TNN::layer_type type,
                 brain_coord_t layer_num,
                 brain_coord_t first_index,
                 brain_coord_t size,
                 std::shared_ptr<head_t> phead,
                 std::vector<neuron_t> &neurons
#ifdef DEBUG_TRACER
                 ,
                 std::shared_ptr<tracer_t> ptracer
#endif
                 ) : type(type),
                     layer_num(layer_num),
                     first_index(first_index),
                     size(size),
                     phead(phead),
                     neurons(neurons)
#ifdef DEBUG_TRACER
                     ,
                     ptracer(ptracer)
#endif

{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> u(0.2, 1);

  // Set last_piece_in_layer
  auto &layer = phead->get_layers_descr()[layer_num];
  auto after_last_index = layer->get_first_neurons_index() + layer->get_cols() * layer->get_rows();
  last_piece_in_layer = (first_index + size == after_last_index);

  // Fill neurons with random values
  for (brain_coord_t i = first_index; i < first_index + size; ++i)
  {
    auto &nr = neurons[i];
    nr.set_u_mem(u(gen));
    nr.set_threshold(initial_neuron_threshold);
  }
}

piece_t &piece_t::operator=(piece_t &&other) noexcept
{
  type = other.type;
  layer_num = other.layer_num;
  other.state.store(state.exchange((other.state.load())));
  other.threshold_base.store(threshold_base.exchange((other.threshold_base.load())));
  first_index = other.first_index;
  size = other.size;
  last_piece_in_layer = other.last_piece_in_layer;
  time_moment = other.time_moment;
  phead = other.phead;
  neurons = std::move(other.neurons); // other.neurons;
#ifdef DEBUG_TRACER
  ptracer = other.ptracer;
#endif

  return *this;
}

// head_t constructors ======================================================
head_t::head_t() : last_piece_in_process(-1)
{
  p_eyes_optics = std::make_shared<eyes_optics_t>(mnist_size, mnist_size);
  // nof_event_threads = std::thread::hardware_concurrency() / 2 ;
  nof_event_threads = 1;
}

// worker constructors ======================================================
worker_t::worker_t(std::shared_ptr<head_t> phead) : phead(phead)
{
  worker_thread = std::thread(&worker_t::execute, this);
}

// Working ****************************************************************************
// ************************************************************************************
piece_t *head_t::get_a_piece_to_process()
{
  auto i = last_piece_in_process.load() + 1;
  if (i == pieces.size()) // Restart from beginning
    i = 0;
  for (; i < pieces.size(); ++i)
  {
    if (pieces[i].exchange_state(state_t::in_process) == state_t::ready_to_be_processed)
    {
      last_piece_in_process.store(i);
      break;
    }
  }

  auto &piece = pieces[i];

  piece.set_time_moment(get_net_time());

  return &(piece);
}

void worker_t::execute()
{
  while (!phead->get_finish())
  {
    piece_t *piece = phead->get_a_piece_to_process();
    auto &layer = phead->get_layers_descr()[piece->get_layer_num()];
    layer->process_neurons(*piece);
    piece->set_state(state_t::ready_to_be_processed);
    DF;
  }
  phead->fetch_add_nof_active_workers(-1);
}

void process_input_weights(neuron_t &cur_neuron, head_t &phead, clock_count_t post_synaptic_spike_time)
{
  for (auto synapse_index : cur_neuron.get_input_synapse_indexes())
  {
    auto &synapse = phead.get_synapses()[synapse_index];
    auto &post_neuron = phead.neuron_ref(synapse.get_src_index());
    stdp_weight_update(cur_neuron, post_neuron, synapse, post_synaptic_spike_time);
  }
}

/**
 * @brief Updates synaptic weights according to STDP learning rule
 *
 * @param pre_neuron Presynaptic neuron
 * @param post_neuron Postsynaptic neuron
 * @param synapse The synapse to update
 * @param post_synaptic_spike_time Time of the postsynaptic spike
 *
 * Implements Spike-Timing-Dependent Plasticity (STDP):
 * - Strengthens connections where presynaptic spikes precede postsynaptic spikes
 * - Weakens connections where presynaptic spikes follow postsynaptic spikes
 * - Uses exponential decay functions for both strengthening and weakening
 */
void stdp_weight_update(neuron_t &pre_neuron,
                        neuron_t &post_neuron,
                        synapse_t &synapse,
                        clock_count_t post_synaptic_spike_time)
{
  static const potential_t dw_plus = std::exp(-dw_alpha_plus);
  static const potential_t dw_minus = std::exp(-dw_alpha_minus);

  // Find the unprocessed time length
  auto post_history = post_neuron.get_spiking_history();
  auto pre_history = pre_neuron.get_spiking_history();

  // Unprocessed time length
  size_t len = 1;
  auto max_len = std::min(static_cast<clock_count_t>(spiking_history_len), post_synaptic_spike_time + 1);
  for (; static_cast<clock_count_t>(len) < max_len; ++len)
    if (post_history.test(len))
      break;

  if (post_synaptic_spike_time > pre_neuron.get_last_fired())
    pre_history <<= (post_synaptic_spike_time - pre_neuron.get_last_fired());
  else if (post_synaptic_spike_time < pre_neuron.get_last_fired())
    pre_history >>= (pre_neuron.get_last_fired() - post_synaptic_spike_time);

  potential_t history_term = 0.0;
  potential_t plus_term = 1.0;
  potential_t minus_term = 1.0;
  for (size_t i = 0; i < len; ++i, plus_term *= dw_plus, minus_term *= dw_minus)
  {
    if (pre_history.test(i))
      history_term += plus_term;
    else
      history_term -= minus_term;
  }

  synapse.set_weight(synapse.get_weight() + dw_max * history_term);
}

void update_postsynaptic_neurons(neuron_t &cur_neuron, head_t &phead,
                                 potential_t threshold_base,
                                 clock_count_t delta_time)
{
  auto output_synapse_indexes = cur_neuron.get_output_synapse_indexes();
  for (auto it = output_synapse_indexes.begin(); it != output_synapse_indexes.end(); ++it)
  {
    auto &synapse = phead.get_synapses()[*it];
    auto &post_neuron = phead.get_neurons()[synapse.get_trg_index()];
    auto weight = synapse.get_weight();
    auto u_mem = cortex_leak_and_input(post_neuron, weight, delta_time); // Save umem
    post_neuron.set_u_mem(u_mem);
    if (u_mem >= (post_neuron.get_threshold() + threshold_base) /*&& delta_time > 0*/)
      post_neuron.set_must_fire(true);
  }
}

potential_t fire_neuron(neuron_t &cur_neuron, piece_t &piece, clock_count_t delta_time)
{
  process_input_weights(cur_neuron, *(piece.get_phead()), piece.get_time_moment());

  update_postsynaptic_neurons(cur_neuron, *(piece.get_phead()),
                              piece.get_threshold_base(), delta_time);

  auto u_res = u_rest;
  cur_neuron.set_spiking_history((cur_neuron.get_spiking_history() << (piece.get_time_moment() - cur_neuron.get_last_fired())) | history_spikes_t(0x01));
  cur_neuron.set_last_fired(piece.get_time_moment());
  cur_neuron.set_must_fire(false);

  return u_res;
}

/**
 * @brief Processes neurons in the retina layer
 *
 * @param piece The piece of the layer to process
 *
 * Handles the first stage of neural processing:
 * - Reads input signals from the scene
 * - Applies retinal-specific leak and input functions
 * - Generates spikes based on membrane potential threshold
 * - Updates weights of outgoing synapses when spikes occur
 * - Optionally traces neural activity for debugging
 */
void retina_layer_t::process_neurons(piece_t &piece)
{

#ifdef DEBUG_TRACER
  auto tracer_buf = piece.get_ptracer()->get_tracer_buf();
#endif

  // calc_threshold_base();
  // std::cout << "Layer: " << layer_num << " threshold_base: " << retina.threshold_base << std::endl;
  // retina.double_counters.reset<counters_t<double>::sum>("spikes_counter");
  auto &neurons = piece.get_neurons();
  for (brain_coord_t i = piece.get_first_index(); i < piece.get_first_index() + piece.get_size(); ++i)
  {
    // Updating potential
    std::uint8_t scene_val;

    auto &cur_neuron = neurons[i];

    auto delta_time = piece.get_time_moment() - cur_neuron.get_last_fired();

    scene_val = piece.get_phead()->get_signal(i);

    auto u_res = retina_leak_and_input(cur_neuron, scene_val, delta_time);

#ifdef DEBUG_TRACER
    // Draw scene
    auto pixel_addr = piece.get_phead()->abs_address(i);
    piece.get_ptracer()->push_to_buf(tracer_buf, pixel_addr,
                                     scene_val, piece.get_time_moment());
#endif
    // Fire retina neuron ===============================================================
    if (u_res > cur_neuron.get_threshold())
    {
      u_res = fire_neuron(cur_neuron, piece, delta_time);
#ifdef DEBUG_TRACER
      auto layer_addr = piece.get_phead()->abs_address(i);
      layer_addr.layer += 2;
      piece.get_ptracer()->push_to_buf(tracer_buf, layer_addr,
                                       std::numeric_limits<std::uint8_t>::max(), piece.get_time_moment());
#endif
    }
    cur_neuron.set_u_mem(u_res);
  }
  // player->ticks_counter.stop_tick();

#ifdef DEBUG_TRACER
  // Trace show scene
  if (piece.get_time_moment() % tr::period == 0)
    piece.get_ptracer()->display_tracer_buf(tracer_buf, piece.get_time_moment());
#endif
}

/**
 * @brief Processes neurons in the cortex layer
 *
 * @param piece The piece of the layer to process
 *
 * Implements the main processing in the neural network:
 * - Checks for neurons that have reached firing threshold
 * - Updates synaptic weights using STDP when spikes occur
 * - Propagates spikes to connected neurons
 * - Maintains neural activity history for learning
 * - Handles debug tracing if enabled
 */
void cortex_layer_t::process_neurons(piece_t &piece)
{
#ifdef DEBUG_TRACER
  auto tracer_buf = piece.get_ptracer()->get_tracer_buf();
#endif

  auto &neurons = piece.get_neurons();
  for (brain_coord_t i = piece.get_first_index(); i < piece.get_first_index() + piece.get_size(); ++i)
  {
    // Updating potential
    auto &cur_neuron = neurons[i];
    auto delta_time = piece.get_time_moment() - cur_neuron.get_last_fired();

    potential_t u_res = 0;
    // Fire cortex neuron ===============================================================
    if (cur_neuron.get_must_fire())
    {
      u_res = fire_neuron(cur_neuron, piece, delta_time);

#ifdef DEBUG_TRACER
      auto addr = piece.get_phead()->abs_address(i);
      addr.layer += 2;
      piece.get_ptracer()->push_to_buf(tracer_buf, addr,
                                       std::numeric_limits<std::uint8_t>::max(), piece.get_time_moment());
#endif
    }
    cur_neuron.set_u_mem(u_res);
  }
  // player->ticks_counter.stop_tick();

#ifdef DEBUG_TRACER
  // Trace show scene
  if (piece.get_time_moment() % tr::period == 0)
    piece.get_ptracer()->display_tracer_buf(tracer_buf, piece.get_time_moment());
#endif
}

void couching_layer_t::process_neurons(piece_t &piece)
{
#ifdef DEBUG_TRACER
  auto tracer_buf = piece.get_ptracer()->get_tracer_buf();
#endif

  auto &neurons = piece.get_neurons();
  auto phead = piece.get_phead();
  for (brain_coord_t i = piece.get_first_index(); i < piece.get_first_index() + piece.get_size(); ++i)
  {
    // Updating potential
    auto &cur_neuron = neurons[i];
    auto delta_time = piece.get_time_moment() - cur_neuron.get_last_fired();
    potential_t u_res = cur_neuron.get_u_mem();

    // Fire couching neuron ===============================================================
    auto fired = false;
    if (cur_neuron.get_must_fire() && (i - piece.get_first_index() == phead->get_label()))
    {
      u_res = fire_neuron(cur_neuron, piece, delta_time);
#ifdef DEBUG_TRACER
      auto addr = piece.get_phead()->abs_address(i);
      addr.layer += 2;
      piece.get_ptracer()->push_to_buf(tracer_buf, addr,
                                       std::numeric_limits<std::uint8_t>::max(), piece.get_time_moment());
#endif
      fired = true;
    }

    cur_neuron.set_u_mem(u_res);
    phead->store_one_metric(fired, i - piece.get_first_index());
  }
  if (piece.is_last_piece_in_layer())
  {
    phead->set_finished_processing_an_image(true);
    DF;
    phead->cv_processing_image.notify_one();
  }
#ifdef DEBUG_TRACER
  // Trace show scene
  if (piece.get_time_moment() % tr::period == 0)
    piece.get_ptracer()->display_tracer_buf(tracer_buf, piece.get_time_moment());
#endif
}

// // Potentials & Weights updater functions------------------------------------------

/**
 * Neural Potential Calculation Functions
 * These functions handle the computation of membrane potentials and synaptic inputs
 */

/**
 * @brief Calculates leaked membrane potential for cortex neurons
 *
 * @param neuron The neuron to calculate potential for
 * @param delta_time Time since last update
 * @return The leaked membrane potential
 *
 * Implements exponential decay of membrane potential over time
 */
potential_t cortex_leaked_u(neuron_t &neuron, clock_count_t delta_time)
{
  static const potential_t cortex_leak_alpha = std::exp(-params::cortex_leak_freq);

  return neuron.get_u_mem() * exp_term(delta_time, cortex_leak_alpha);
}

potential_t cortex_leak_and_input(neuron_t &neuron,
                                  potential_t &weight,
                                  clock_count_t delta_time)
{
  auto u = cortex_leaked_u(neuron, delta_time) + weight;
  return u;
}

potential_t retina_leak_and_input([[maybe_unused]] neuron_t &neuron,
                                  scene_signal_t signal,
                                  clock_count_t delta_time)
{
  auto leak_term = cortex_leaked_u(neuron, delta_time);
  auto u = leak_term + signal * detector_alpha;
  return u;
}

// // head_t functions -------------------------------------------------------------

/**
 * Thread Management Functions
 * Handle the parallel processing infrastructure of the neural network
 */

/**
 * @brief Initializes and starts worker threads
 *
 * Creates worker threads for parallel processing of neural network pieces.
 * The number of workers is determined by the nof_event_threads parameter,
 * typically set to optimize performance based on available CPU cores.
 */
void head_t::wake_up()
{
  // Init threads
  finish.store(false);
  // Create workers
  for (unsigned i = 0; i < nof_event_threads; ++i)
    workers.push_back(std::make_shared<worker_t>(std::shared_ptr<head_t>(this)));
}

void head_t::go_to_sleep()
{
  // Set finish flag
  finish.store(true);

  // wait for workers to stop
  while (nof_active_workers.load())
    ;
  // Pick up stopped workers
  for (auto worker : workers)
    worker->get_worker_thread().join();
}

abs_address_t head_t::abs_address(brain_coord_t neuron_index)
{
  brain_coord_t layer_num = 0;
  auto it = layer_num_by_1st_neuron_index.upper_bound(neuron_index);
  if (it == layer_num_by_1st_neuron_index.end())
    layer_num = layers_descr.size() - 1;
  else
    layer_num = it->second - 1;
  layer_t &layer = *(layers_descr[layer_num]);
  auto nofcols = layer.get_cols();
  auto first_index = layer.get_first_neurons_index();
  auto [row, col] = twod_from1(neuron_index - first_index, nofcols);

  return abs_address_t(layer_num, row, col);
}

brain_coord_t head_t::neuron_index(abs_address_t &&addr)
{
  // Calculate first piece index in layer
  auto first_neurons_index = layers_descr[addr.layer]->get_first_neurons_index();
  auto neuron_index = oned_from2(addr.abs_row, addr.abs_col, layers_descr[addr.layer]->get_cols());
  return first_neurons_index + neuron_index;
}

void head_t::store_one_metric(bool fired, brain_coord_t col)
{
  metrics.store_one_metric(metrics_t::results_t::NOF_ATTEMPTS);

  auto label = get_label();

  if (fired)
  {
    metrics.store_one_metric(metrics_t::results_t::TOTAL_SPIKES);
    if (col == label)
      metrics.store_one_metric(metrics_t::results_t::LABELED_SPIKES);
  }
}

scene_t *head_t::next_image(std::shared_ptr<mnist_set> pmnist)
{
  moving_gaze.lock();
  auto [scene, label_] = pmnist->next_image();
  scene_index = get_p_eyes_optics()->scene_index;
  if (!scene)
  {
    moving_gaze.unlock();
    return scene;
  }
  p_eyes_optics->set_scene(scene);
  set_label(label_);
  set_finished_processing_an_image(false);
  moving_gaze.unlock();
  return scene;
}

void head_t::print_nof_synapses_per_neuron()
{
  try
  {
    std::ofstream ofs;
    ofs.open("../synapses_per_neuron.txt", std::ios::out | std::ios::trunc);
    for (size_t i = 0; i < neurons.size(); i++)
    {
      ofs << i << "  inp: " << neurons[i].get_input_synapse_indexes().size()
          << " outp: " << neurons[i].get_output_synapse_indexes().size()
          << std::endl;
    }
    ofs.close();
  }
  catch (...)
  {
    std::cout << "Error saving synapses" << std::endl;
  }
}

potential_t exp_term(clock_count_t delta_time, potential_t alpha)
{
  potential_t val = 1.0;
  for (clock_count_t i = 0; i < delta_time; i++)
    val *= alpha;
  return val;
}

//     {
// #ifdef DEBUG_TRACER
//         ptracer->lock_screen();
// #endif

//         std::ofstream ofs(file_name);

//         // Make layers' description
//         std::vector<layer_descr_t> layers_descriptions;
//         for (auto it = layers.begin(); it != layers.end(); ++it)
//         {
//             layers_descriptions.emplace_back((*it)->ltype, rc_t((*it)->neurons.size(), (*it)->neurons[0].size()));
//         }

//         // Make network's description
//         network_descr_t network_dsc(layers_descriptions, connections);

//         ofs << network_dsc;

//         ofs << *this << std::endl;

//         ofs.close();

// #ifdef DEBUG_TRACER
//         ptracer->unlock_screen();
// #endif
//     }
//     catch (...)
//     {
//         std::cout << "Error saving model" << std::endl;
//     }
// }