#include "input_output.h"
#include "brain.h"
#include "tracer.h"
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

/**
 * Network Construction Functions
 * These functions handle the initialization and setup of the neural network
 */

/**
 * @brief Adds a new layer to the neural network
 *
 * @param ltype Type of layer (RETINA, CORTEX, or OUTPUT)
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
                       brain_coord_t &first_neuron_index, brain_coord_t nof_pieces, std::shared_ptr<head_t> phead TRACE_PARAM)
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
  case TNN::OUTPUT:
    layer = std::make_unique<output_layer>(ltype, neurons_rows, neurons_cols, first_neuron_index);
    break;
  case TNN::REFERENCE:
    layer = std::make_unique<reference_layer_t>(ltype, neurons_rows, neurons_cols, first_neuron_index);
    break;
  default:
    throw std::invalid_argument("Invalid layer type");
  }
  layers_descr.push_back(std::move(layer)); // layer is null now
  auto last_layer_index = layers_descr.size() - 1;
  layer_num_by_1st_neuron_index[first_neuron_index] = last_layer_index;
  auto layer_size = neurons_rows * neurons_cols;

  // Create neurons
  neurons.resize(first_neuron_index + layer_size); // Thresholds are set to default

  // Create pieces
  assert(!(layer_size % nof_pieces));

  auto piece_first_neuron_index = first_neuron_index;
  auto piece_size = layer_size / nof_pieces;
  for (brain_coord_t i = 0; i < nof_pieces; ++i)
  {
    pieces.emplace_back(ltype, last_layer_index,
                        piece_first_neuron_index,
                        piece_size, phead, neurons TRACE_ARG);
    piece_first_neuron_index += piece_size;
  }

  // Update layer's first_neuron_index
  first_neuron_index += layer_size;
}

// Explicit instantiation for the types used in this project to ensure the
// template is emitted into the object file (we call it from other TUs).
// template void head_t::print_field<neuron_t, potential_t>(const std::string &filename, potential_t (neuron_t::*getter)() const, const std::vector<neuron_t> &collection TRACE_PARAM) const;

constexpr int out_of_range = -1;
// Add connections between two layers. Supports multiple connection types.
void head_t::add_connections(brain_coord_t src_layer, brain_coord_t trg_layer,
                             TNN::ferment_t ferment, TNN::connection_type connection_type)
{
  // Record connection description
  connections_descr.emplace_back(src_layer, trg_layer, ferment, connection_type);

  // Prepare random generator for initial weights
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> weight(0.001, 1.0);

  switch (connection_type)
  {
  case TNN::FULLY_CONNECTED:
  {
    // Connect every neuron in source layer to every neuron in target layer
    for (brain_coord_t i = 0; i < layers_descr[src_layer]->get_rows(); ++i)
      for (brain_coord_t j = 0; j < layers_descr[src_layer]->get_cols(); ++j)
      {
        auto src_addr = neuron_index(abs_address_t(src_layer, i, j));
        auto &src_neuron = neuron_ref(src_addr);

        for (brain_coord_t k = 0; k < layers_descr[trg_layer]->get_rows(); ++k)
          for (brain_coord_t l = 0; l < layers_descr[trg_layer]->get_cols(); ++l)
          {
            auto trg_addr = neuron_index(abs_address_t(trg_layer, k, l));
            synapses.emplace_back(weight(gen), ferment, src_addr, trg_addr);
            auto synapse_index = static_cast<brain_coord_t>(synapses.size() - 1);
            src_neuron.emplace_output_synapse_index(synapse_index);

            neuron_ref(trg_addr).add_input();
          }

        // Adjust thresholds according to out-degree
        normalize_neurons_thresholds();
      }
  }
  break;

  case TNN::ONE_TO_ONE:
  {
    // Connect neurons by their flattened index up to the smaller layer size
    auto src_first = layers_descr[src_layer]->get_first_neuron_index();
    auto trg_first = layers_descr[trg_layer]->get_first_neuron_index();
    auto src_size = layers_descr[src_layer]->get_rows() * layers_descr[src_layer]->get_cols();
    auto trg_size = layers_descr[trg_layer]->get_rows() * layers_descr[trg_layer]->get_cols();
    auto min_size = std::min(src_size, trg_size);

    for (brain_coord_t idx = 0; idx < min_size; ++idx)
    {
      auto src_addr = src_first + idx;
      auto trg_addr = trg_first + idx;
      neuron_ref(trg_addr).add_input(); // Update input count for threshold normalization
      synapses.emplace_back(weight(gen), ferment, src_addr, trg_addr);
      auto synapse_index = static_cast<brain_coord_t>(synapses.size() - 1);
      neuron_ref(src_addr).emplace_output_synapse_index(synapse_index);
    }

    normalize_neurons_thresholds();
  }
  break;

  default:
    throw std::invalid_argument("Unsupported connection type");
  }
}

void head_t::normalize_neurons_thresholds()
{
  for (auto &neuron : neurons)
  {
    neuron.set_threshold(params.initial_neuron_threshold);
  }
}

piece_t::piece_t(TNN::layer_type type,
                 brain_coord_t layer_num,
                 brain_coord_t first_index,
                 brain_coord_t size,
                 std::shared_ptr<head_t> phead,
                 std::vector<neuron_t> &neurons TRACE_PARAM) : type(type),
                                                               layer_num(layer_num),
                                                               first_index(first_index),
                                                               size(size),
                                                               phead(phead),
                                                               neurons(neurons) TRACE_MEMBER_INIT

{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> u(0.2, 1);

  threshold_base.store(params.normal_threshold_base.load());

  // Fill neurons with random values
  for (brain_coord_t i = first_index; i < first_index + size; ++i)
  {
    auto &nr = neurons[i];
    nr.set_u_mem(u(gen));
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
  time_moment = other.time_moment;
  phead = other.phead;
  neurons = std::move(other.neurons); // other.neurons;
  TRACE_STMT(ptracer = other.ptracer;);

  return *this;
}

// head_t constructors ======================================================
head_t::head_t() : last_piece_in_process(-1)
{
  p_eyes_optics = std::make_shared<eyes_optics_t>(mnist_size, mnist_size);
  nof_event_threads = std::thread::hardware_concurrency() - 1;
  // nof_event_threads = 1;
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

  if (i == 0) // Retina layer, first piece
    piece.set_time_moment(inc_net_time());
  else
    piece.set_time_moment(get_net_time());

  return &(piece);
}

void worker_t::execute()
{
  piece_t *last_piece = &(phead->get_pieces()[phead->get_pieces().size() - 1]);

  while (!phead->get_finish())
  {
    piece_t *piece = phead->get_a_piece_to_process();
    auto &layer = phead->get_layers_descr()[piece->get_layer_num()];
    layer->process_neurons(*piece);
    piece->set_state(state_t::ready_to_be_processed);
    // Notify main thread if this was end of current image processing
    if (piece == last_piece)
    {
      phead->set_finished_processing_an_image(true);
      phead->cv_processing_image.notify_one();
    }
  }
  phead->fetch_add_nof_active_workers(-1);
}

// stdp_weight_update ****************************************************************************
void stdp_weight_update(neuron_t &cur_neuron,
                        neuron_t &post_neuron,
                        synapse_t &synapse,
                        clock_count_t time_moment)
{
  static const potential_t dw_plus = std::exp(-1.0 / params.dw_plus_time.load());
  static const potential_t dw_minus = std::exp(-1.0 / params.dw_minus_time.load());

  auto cur_history = cur_neuron.get_spiking_history();
  auto post_history = post_neuron.get_spiking_history();

  // Time span between cur and post
  auto delta_time = std::clamp(post_neuron.get_last_fired() - cur_neuron.get_last_fired(),
                               -clock_count_t{spiking_history_len} + 1, clock_count_t{spiking_history_len} - 1);
  potential_t history_term = 0.0;

  // STDP: cur before post -> increase weight, cur after post -> decrease weight
  if (delta_time >= 0)
  {
    // Process every spike of post_neuron after cur_neuron last spike
    potential_t plus_term = 1.0;
    for (int i = delta_time; i >= 0; --i, plus_term *= dw_plus)
      if (post_history.test(i))
        history_term += plus_term;
  }
  else
  {
    // Process every tick of cur_neuron didn't fire after post_neuron last spike
    potential_t minus_term = 1.0;
    for (int i = 1; i < -delta_time + 1; ++i, minus_term *= dw_minus)
      if (!cur_history.test(i))
        history_term -= minus_term;
      else
        break; // No need to continue if cur fired
  };
  synapse.set_weight(std::clamp(synapse.get_weight() + history_term, 0.0, 10.0));
}

void neuron_t::after_spike_update(clock_count_t time_moment)
{
  u_mem.store(params.u_rest);
  threshold_adaptation += params.th_adapt_inc_per_spike.load();
  last_fired = time_moment;
  must_fire.store(false);
}

void update_postsynaptic_neurons(neuron_t &cur_neuron, head_t &phead,
                                 potential_t threshold_base,
                                 clock_count_t time_moment)
{
  auto output_synapse_indexes = cur_neuron.get_output_synapse_indexes();

  // Update post neurons------------------------
  for (auto it = output_synapse_indexes.begin(); it != output_synapse_indexes.end(); ++it)
  {
    auto &synapse = phead.get_synapse(*it);
    auto &post_neuron = phead.get_neurons()[synapse.get_trg_index()];
    // Weight update
    stdp_weight_update(cur_neuron, post_neuron, synapse, time_moment);

    auto weight = synapse.get_weight();

    if (post_neuron.get_must_fire())
      continue; // will fire anyway, no need to update

    // Use time since post_neuron last fired for leak/input computation
    clock_count_t delta_for_post = time_moment - post_neuron.get_last_fired();

    post_neuron.leaks_values_by_time(time_moment, delta_for_post);
    post_neuron.set_u_mem(post_neuron.get_u_mem() + weight);

    if ((post_neuron.get_u_mem() > (post_neuron.get_threshold() + post_neuron.get_threshold_adaptation())) && delta_for_post > 0)
      post_neuron.set_must_fire(true);
  }
}

// update threshold_adaptation ****************************************************************************

void neuron_t::leaks_values_by_time(clock_count_t time_moment, clock_count_t delta_time)
{
  if (time_moment == leaks_update_time)
    return;
  static auto th_leak = std::exp(-1.0 / params.threshold_decrease_time.load());
  // Membrane potential leak
  static auto membrane_leak_alpha = std::exp(-1 / params.umem_decrease_time.load());
  u_mem.store(u_mem.load() * ipow(membrane_leak_alpha, delta_time));

  // Threshold adaptation leak
  threshold_adaptation *= ipow(th_leak, delta_time);

  leaks_update_time = time_moment;
}

// Layer processing functions ****************************************************************************
void retina_layer_t::process_neurons(piece_t &piece)
{

  TRACE_DECL(auto tracer_buf = piece.get_ptracer()->get_tracer_buf();)

  auto &neurons = piece.get_neurons();
  auto time_moment = piece.get_time_moment();

  for (brain_coord_t i = piece.get_first_index(); i < piece.get_first_index() + piece.get_size(); ++i)
  {
    // Updating potential
    std::uint8_t scene_val;

    auto &cur_neuron = neurons[i];
    auto delta_time = time_moment - cur_neuron.get_last_fired();

    scene_val = piece.get_phead()->get_signal(i);

    cur_neuron.leaks_values_by_time(time_moment, delta_time);

    if (time_moment != cur_neuron.get_leaks_update_time())
    {
      // Membrane potential leak
      static auto membrane_leak_alpha = std::exp(-1 / params.umem_decrease_time.load());
      cur_neuron.set_u_mem(cur_neuron.get_u_mem() * ipow(membrane_leak_alpha, delta_time));
      cur_neuron.set_leak_update_time(time_moment);
    }

    cur_neuron.set_u_mem(cur_neuron.get_u_mem() + scene_val / params.amplitude_quant_size.load() * params.detector_alpha.load());

    TRACE_STMT( // Draw scene
        auto pixel_addr = piece.get_phead()->abs_address(i);
        piece.get_ptracer()->push_to_buf(tracer_buf, pixel_addr, scene_val, time_moment););

    // Fire retina neuron ===============================================================
    if ((cur_neuron.get_u_mem() > (cur_neuron.get_threshold() + cur_neuron.get_threshold_adaptation())) && delta_time > 0)
    {
      update_postsynaptic_neurons(cur_neuron, *(piece.get_phead()),
                                  piece.get_threshold_base(), time_moment); // also reduces u_mem

      cur_neuron.set_u_mem(params.u_rest);
      cur_neuron.set_last_fired(time_moment);

      TRACE_STMT(auto layer_addr = piece.get_phead()->abs_address(i);
                 layer_addr.layer += 1;
                 piece.get_ptracer()->push_to_buf(tracer_buf, layer_addr,
                                                  std::numeric_limits<std::uint8_t>::max(), time_moment););
    }
  }
  // player->ticks_counter.stop_tick();

  TRACE_STMT(if (time_moment % tr::period == 0) piece.get_ptracer()->display_tracer_buf(tracer_buf, time_moment););
}

void reference_layer_t::process_neurons(piece_t &piece)
{
  std::bitset<nof_cathegories_const> reference_values{0};

  auto &phead = *(piece.get_phead());
  if (!phead.get_couching_mode())
    return;

  TRACE_DECL(auto tracer_buf = piece.get_ptracer()->get_tracer_buf();)

  auto &neurons = piece.get_neurons();
  reference_values.set(piece.get_phead()->get_label());

  std::uint8_t scene_val;
  brain_coord_t k = 0;
  for (brain_coord_t i = piece.get_first_index();
       i < piece.get_first_index() + piece.get_size(); ++i, ++k)
  {
    auto &cur_neuron = neurons[i];
    auto output_synapse_indexes = cur_neuron.get_output_synapse_indexes();

    for (auto it = output_synapse_indexes.begin(); it != output_synapse_indexes.end(); ++it)
    {
      auto &synapse = phead.get_synapse(*it);
      auto &post_neuron = phead.get_neurons()[synapse.get_trg_index()];

      // Fire reference neuron ===============================================================
      if (reference_values.test(k))
      {
        post_neuron.set_must_fire(true);
        auto layer_addr = piece.get_phead()->abs_address(i);
        TRACE_STMT(layer_addr = piece.get_phead()->abs_address(i);
                   layer_addr.layer += 1;
                   piece.get_ptracer()->push_to_buf(tracer_buf, layer_addr,
                                                    std::numeric_limits<std::uint8_t>::max(),
                                                    piece.get_time_moment()););
      }
    }
  }
  TRACE_STMT(if (piece.get_time_moment() % tr::period == 0)
                 piece.get_ptracer()
                     ->display_tracer_buf(tracer_buf, piece.get_time_moment()););
}

void cortex_layer_t::process_neurons(piece_t &piece)
{
  TRACE_DECL(auto tracer_buf = piece.get_ptracer()->get_tracer_buf();)

  auto &neurons = piece.get_neurons();

  // Debug
  // potential_t avg = 0.0;
  // Debug

  for (brain_coord_t i = piece.get_first_index();
       i < piece.get_first_index() + piece.get_size(); ++i)
  {
    // Updating potential
    auto &cur_neuron = neurons[i];
    auto time_moment = piece.get_time_moment();

    // Fire cortex neuron ===============================================================
    if (cur_neuron.get_must_fire())
    {
      update_postsynaptic_neurons(cur_neuron, *(piece.get_phead()),
                                  piece.get_threshold_base(), time_moment); // also reduces u_mem;

      cur_neuron.after_spike_update(time_moment);

      TRACE_STMT(auto addr = piece.get_phead()->abs_address(i);
                 addr.layer += 1;
                 piece.get_ptracer()->push_to_buf(tracer_buf, addr, std::numeric_limits<std::uint8_t>::max(), piece.get_time_moment()););
    }
    // Debug
    // if (piece.get_phead()->get_scene_index() == 2)
    //   avg += cur_neuron.get_u_mem() / (piece.get_first_index() + piece.get_size());
    // Debug
  }
  // Debug
  // if (piece.get_phead()->get_scene_index() == 2)
  // *plogger << avg << std::endl;
  // Debug

  TRACE_STMT(if (piece.get_time_moment() % tr::period == 0) piece.get_ptracer()->display_tracer_buf(tracer_buf, piece.get_time_moment()););
}

void output_layer::process_neurons(piece_t &piece)
{
  TRACE_DECL(auto tracer_buf = piece.get_ptracer()->get_tracer_buf();)

  auto &neurons = piece.get_neurons();
  auto phead = piece.get_phead();
  std::bitset<nof_cathegories_const> fired_neurons{0};
  brain_coord_t k = 0;
  for (brain_coord_t i = piece.get_first_index();
       i < piece.get_first_index() + piece.get_size(); ++i, ++k)
  {
    // Updating potential
    auto &cur_neuron = neurons[i];
    auto delta_time = piece.get_time_moment() - cur_neuron.get_last_fired();

    // Fire couching neuron ===============================================================
    if (cur_neuron.get_must_fire())
    {
      cur_neuron.after_spike_update(piece.get_time_moment());
      fired_neurons.set(k);

      TRACE_STMT(auto addr = piece.get_phead()->abs_address(i);
                 addr.layer += 1;
                 piece.get_ptracer()->push_to_buf(tracer_buf, addr, std::numeric_limits<std::uint8_t>::max(), piece.get_time_moment()););
    }
    // cur_neuron.set_u_mem(u_res);
  }

  // Store learning metrics
  if (!phead->get_couching_mode())
    phead->store_one_metric(fired_neurons);

  TRACE_STMT(if (piece.get_time_moment() % tr::period == 0) piece.get_ptracer()->display_tracer_buf(tracer_buf, piece.get_time_moment()););
}

// Potentials & Weights updater functions------------------------------------------

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

abs_address_t head_t::abs_address(brain_coord_t neuron_index) const
{
  brain_coord_t layer_num = 0;
  auto it = layer_num_by_1st_neuron_index.upper_bound(neuron_index);
  if (it == layer_num_by_1st_neuron_index.end())
    layer_num = layers_descr.size() - 1;
  else
    layer_num = it->second - 1;
  layer_t &layer = *(layers_descr[layer_num]);
  auto nofcols = layer.get_cols();
  auto first_index = layer.get_first_neuron_index();
  auto [row, col] = twod_from1(neuron_index - first_index, nofcols);

  return abs_address_t(layer_num, row, col);
}

brain_coord_t head_t::neuron_index(abs_address_t &&addr)
{
  // Calculate first piece index in layer
  auto first_neuron_index = layers_descr[addr.layer]->get_first_neuron_index();
  auto neuron_index = oned_from2(addr.abs_row, addr.abs_col, layers_descr[addr.layer]->get_cols());
  return first_neuron_index + neuron_index;
}

void head_t::store_one_metric(std::bitset<nof_cathegories_const> fired_neurons)
{
  if (fired_neurons == 0)
    metrics.store_one_metric(metrics_t::results_t::NO_SPIKES);
  else if (fired_neurons.count() > 1 || !fired_neurons.test(label))
    metrics.store_one_metric(metrics_t::results_t::UNLABELED_SPIKES);
  else // fired_neurons.test(label)
    metrics.store_one_metric(metrics_t::results_t::LABELED_SPIKES);
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

potential_t ipow(potential_t alpha, clock_count_t pow)
{
  potential_t val = 1.0;
  for (clock_count_t i = 0; i < pow; i++)
    val *= alpha;
  return val;
}
