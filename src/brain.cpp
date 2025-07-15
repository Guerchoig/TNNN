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

using namespace TNN;
using namespace params;

// Constructors ************************************************************
//              ************************************************************

// Layer constructor's common ==============================================
template <Is_layer T>
void create_neurons(T *layer, const layer_place_n_size_t &place_n_size)
{
  if (!layer)
  {
    throw std::invalid_argument("Layer pointer is null.");
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> u(0.2, 1);

  auto &neurons = layer->neurons;
  neurons.clear(); // Clear existing neurons to ensure fresh initialization
  neurons.reserve(place_n_size.rows);

  for (brain_coord_t i = 0; i < place_n_size.rows; ++i)
  {
    neurons.emplace_back();
    neurons.back().reserve(place_n_size.cols);

    for (brain_coord_t j = 0; j < place_n_size.cols; ++j)
    {
      auto uu = u(gen);

      try
      {
        neurons.back().emplace_back(uu, initial_neuron_threshold, 0L);
      }
      catch (const std::exception &e)
      {
        std::cerr << "Failed to create neuron: " << e.what() << std::endl;
        throw;
      }
    }
  }
}

// Retina layer constructors ------------------------------------------------
retina_layer_t::retina_layer_t()
{
  ltype = TNN::layer_type::RETINA;
}

retina_layer_t::retina_layer_t(const layer_place_n_size_t &place_n_size)
{
  ltype = TNN::layer_type::RETINA;
  create_neurons(this, place_n_size);
}

// Cortex layer constructors ------------------------------------------------
cortex_layer_t::cortex_layer_t()
{
  ltype = TNN::layer_type::CORTEX;
}

cortex_layer_t::cortex_layer_t(const layer_place_n_size_t &place_n_size)
{
  ltype = TNN::layer_type::CORTEX;
  create_neurons(this, place_n_size);
}

// MNIST Couch layer constructors ------------------------------------------------
couching_layer_t::couching_layer_t()
{
  ltype = TNN::layer_type::COUCHING;
};
couching_layer_t::couching_layer_t(const layer_place_n_size_t &place_n_size)
{
  ltype = TNN::layer_type::COUCHING;
  create_neurons(this, place_n_size);
}

// head_t constructors ======================================================
head_t::head_t()
{
  p_eyes_optics = std::make_shared<eyes_optics_t>(mnist_size, mnist_size);

  nof_event_threads = std::thread::hardware_concurrency() / 2 - 1;
}

// tworker_t constructor ==================================================
template <typename Derived>
#ifdef DEBUG_TRACER
tworker_t<Derived>::tworker_t(head_t *phead, brain_coord_t layer_num, ptracer_t ptracer) : ptracer(ptracer)
#else
tworker_t<Derived>::tworker_t(head_t *phead, brain_coord_t layer_num)
#endif
{
  this->layer_num = layer_num;
  this->phead = phead;

  worker_thread = std::thread(&tworker_t::execute, this);
  phead->active_workers++;
}

// Working ****************************************************************************
// ************************************************************************************

// Upper level workers  ================================================================

template <typename Derived>
void tworker_t<Derived>::execute()
{
  while (!phead->finish.load())
  {
    static_cast<Derived *>(this)->worker();

    // if (layer_num == 1)
    //   phead->save_choosen_weights({1, 13, 0}, {0, 1}, 9);

    if (output_events_buf.size() > 0)
      move_to_workers<&tworker_t<Derived>::output_events_buf,
                      &tworker_t<Derived>::input_events>();
    if (output_weights_buf.size() > 0)
      move_to_workers<&tworker_t<Derived>::output_weights_buf,
                      &tworker_t<Derived>::input_weights>();
  }

  phead->active_workers--;
}

void retina_worker_t::worker()
{
  retina_process();
}

void cortex_worker_t::worker()
{
  cortex_process(); // time_moment = empty_time
}

void couch_worker_t::worker()
{
  cortex_process(); // time_moment = empty_time
}

// Workers process methods ================================================

// template <typename Derived>
// void tworker_t<Derived>::calc_delta_threshold(neuron_t &neuron, clock_count_t delta_time)
// {
//   auto th = neuron.threshold + inc_threshold_after_fired - threshold_rally_rate * delta_time;
//   neuron.threshold = std::max(th, initial_neuron_threshold);
// }

template <typename Derived>
void tworker_t<Derived>::process_input_weights()
{
  std::unique_ptr<std::vector<weight_event_t>> p_wpack;

  while (input_weights.try_pop(p_wpack))
  {
    // Proccess input weights pack
    for (auto &we : *p_wpack)
    {
      auto &neuron = phead->neuron_ref(we.addr);
      auto &synapse = neuron.synapses.at(we.synapse_num);
      auto &post_neuron = phead->neuron_ref(synapse.target_addr);
      stdp_weight_update(neuron,
                         post_neuron,
                         synapse,
                         we.postsynaptic_spike_time,
                         we.synapse_num);
    }
  }
}

template <typename Derived>
void tworker_t<Derived>::stdp_weight_update(neuron_t &neuron,
                                            [[maybe_unused]] neuron_t &post_neuron,
                                            synapse_t &synapse,
                                            clock_count_t post_synaptic_spike_time,
                                            [[maybe_unused]] brain_coord_t synapse_num)
{

  auto delta_time = calc_delta_time(neuron, post_synaptic_spike_time);

  // Update counters
  // std::stringstream ss;
  // ss << delta_time;

  auto dw = calc_dw(delta_time);

  // if (layer_num == 0 && synapse_num < 80)
  //   logger << "dw:\t" << dw << "\t" << synapse.weight << std::endl;

  synapse.weight += dw;

  // phead->layers[layer_num]->double_counters.inc_by<counters_t<double>::avg>("dw", dw);

  // dw = ltp_delta_max * post_neuron.trace - ltd_delta_max * neuron.trace;
  // auto delta_time = post_synaptic_spike_time - neuron.last_fired; // delta_time
  // // Decay the spike traces
  // if (delta_time != 0)
  // {
  //   neuron.trace *= std::exp(-delta_time / tau_plus);
  //   post_neuron.trace *= std::exp(-delta_time / tau_minus);
  // }

  // // Check for pre-synaptic spike (LTP)
  // potential_t dw;
  // // if (delta_time == 0)
  // //   dw = ltp_delta_max * post_neuron.trace - ltd_delta_max * neuron.trace;
  // // If post neuron spiked recently (pre before post case)
  // if (delta_time >= 0)
  //   dw = ltp_delta_max * post_neuron.trace;
  // else if (delta_time < 0) // Check for post-synaptic spike (LTD)
  //   dw = -ltd_delta_max * neuron.trace;

  // synapse.weight += dw;

  // // Apply weight bounds
  // synapse.weight = std::clamp(synapse.weight, w_min, w_max);
}

template <typename Derived>
int64_t tworker_t<Derived>::calc_delta_time(neuron_t &neuron, clock_count_t post_synaptic_spike_time)
{
  auto _delta_time = post_synaptic_spike_time - neuron.last_fired;

  // Haven't spiked since last update
  if (_delta_time >= 0)
    return _delta_time;

  uint32_t i = 1;
  uint32_t abs_dt = abs(_delta_time); // >= 1
  for (prev_spikes_t mask = 0x01 << abs_dt; mask != 0; mask <<= 1, ++i)
    if ((neuron.prev_spikes & mask) == mask)
      break;

  return i; // max(i) == sizeof(prev_spikes_t)
};

template <typename Derived>
void tworker_t<Derived>::retina_process()
{

  process_input_weights();

  retina_process_input_events();
}

template <typename Derived>
void tworker_t<Derived>::retina_process_input_events()
{
  auto time_moment = phead->net_timer.time();

  // Set shortcuts
  auto &retina = *(phead->pretina);
  auto &neurons = retina.neurons;

  // Process scene inputs

  retina.p_eyes_optics->get_locked_scene();

#ifdef DEBUG_TRACER
  auto tracer_buf = ptracer->get_tracer_buf();
#endif

  calc_threshold_base();
  // std::cout << "Layer: " << layer_num << " threshold_base: " << retina.threshold_base << std::endl;
  retina.double_counters.reset<counters_t<double>::sum>("spikes_counter");
  for (brain_coord_t i = 0; static_cast<size_t>(i) < phead->layers[layer_num]->neurons.size(); ++i)
  {

    for (brain_coord_t j = 0; static_cast<size_t>(j) < phead->layers[layer_num]->neurons[0].size(); ++j)
    {
      // Updating potential
      neuron_t &trg = neurons[i][j];
      std::uint8_t scene_val;

      auto delta_time = calc_delta_time(trg, time_moment);

      scene_val = retina.p_eyes_optics->get_signal(i, j);

      auto u_res = retina_leak_and_input(trg, scene_val, delta_time);

#ifdef DEBUG_TRACER
      // Draw scene
      ptracer->push_to_buf(tracer_buf, static_cast<brain_coord_t>(layer_num), i, j,
                           scene_val, time_moment);
#endif
      // Fire retina neuron ===============================================================
      if (u_res >= (trg.threshold + retina.threshold_base) && delta_time > 0)
      {

        retina.double_counters.inc_by<counters_t<double>::sum>("spikes_counter", 1);

        create_synapses_events(trg,
                               {static_cast<brain_coord_t>(layer_num), i, j},
                               time_moment);

        u_res = u_rest;
        trg.prev_spikes = (trg.prev_spikes << (time_moment - trg.last_fired)) | prev_spikes_t(0x01);
        trg.last_fired = time_moment;

#ifdef DEBUG_TRACER
        ptracer->push_to_buf(tracer_buf, static_cast<brain_coord_t>(layer_num + 2), i, j,
                             std::numeric_limits<std::uint8_t>::max(), time_moment);
#endif
      }
      trg.u_mem = u_res;
    }
    // player->ticks_counter.stop_tick();
  }

  retina.p_eyes_optics->unlock_scene();

#ifdef DEBUG_TRACER
  // Trace show scene
  if (time_moment % tr::period == 0)
    ptracer->display_tracer_buf(tracer_buf, time_moment);
#endif
}

potential_t cortex_signal(const synapse_t &synapse)
{
  auto du = synapse.weight; // * membrana_resistance;
  return du;
}

// potential_t atan_delta_threshold(double next_to_prev)
// {
//   auto km1 = std::atan(1 - next_to_prev);
//   auto res = params::threshold_increment * km1;
//   return res;
// }

template <typename Derived>
void tworker_t<Derived>::calc_threshold_base()
{
  auto &layer = *(phead->layers[layer_num]);
  auto spikes = layer.double_counters.get("spikes_counter");
  auto percentage = spikes / (layer.neurons.size() * layer.neurons[0].size());

  bool high_persentage = percentage > max_firing_percentage;
  bool low_persentage = percentage < min_firing_percentage;

  auto slowdown = layer.slowdown.load();

  layer.threshold_base = params::normal_threshold_base;

  if (slowdown == slowdown_apply || high_persentage)
    layer.threshold_base = params::high_threshold_base;

  if (low_persentage)
    layer.threshold_base = params::low_threshold_base;

  layer.slowdown.store(slowdown_donothing);

  // if (layer_num < static_cast<brain_coord_t>(phead->layers.size() - 1))
  // {
  //   auto next_avg_output_size = phead->layers[layer_num + 1]->avg_out_ev_counter.get_value();
  //   auto curr_avg_output_size = phead->layers[layer_num]->avg_out_ev_counter.get_value();

  //   if (curr_avg_output_size > 0)
  //   {
  //     double next_to_curr = next_avg_output_size / curr_avg_output_size;
  //     delta_threshold = atan_delta_threshold(next_to_curr);
  //   }
  //   else if (next_avg_output_size != 0)
  //     delta_threshold = -params::threshold_increment;
  // }
}

template <typename Derived>
void tworker_t<Derived>::cortex_process_input_events(bool couching_mode)
{

  std::unique_ptr<std::vector<neuron_event_t>> p_input_epack;
#ifdef DEBUG_TRACER
  auto tracer_buf = ptracer->get_tracer_buf();
#endif

  auto &curr_layer = *(phead->layers[layer_num]);
  auto &prev_layer = *(phead->layers[layer_num - 1]);

  // std::cout << "Layer: " << layer_num << " threshold_base: " << curr_layer.threshold_base << std::endl;

  while (input_events.try_pop(p_input_epack))
  {
    if (p_input_epack.get() == nullptr)
    {
      break;
    }
    // Calculate delta threshold to coordinate layers speed
    calc_threshold_base();
    curr_layer.double_counters.reset<counters_t<double>::sum>("spikes_counter");
    curr_layer.double_counters.reset<counters_t<double>::sum>("weights_counter");

    auto threshold_base = prev_layer.threshold_base.load();
    bool oversize = input_events.was_size() > 1;
    int slowdown = slowdown_donothing;
    if ((threshold_base < params::high_threshold_base) && oversize)
    {
      // need to rise the threshold in the previous layer
      slowdown = slowdown_apply;
    }
    if ((threshold_base == params::high_threshold_base) && !oversize)
    {
      // need to fall the threshold in the previous layer
      slowdown = slowdown_cancel;
    }
    prev_layer.slowdown.store(slowdown);

    clock_count_t update_time = 0;
    // Process one input events pack
    for (auto &e : *p_input_epack)
    {
      auto &trg = phead->neuron_ref(e.target_addr);
      auto &src = phead->neuron_ref(e.source_addr);
      auto &synapse = src.synapses[e.src_synapse];

      auto delta_time = calc_delta_time(trg, e.presynaptic_spike_time);

      // Calculate u
      potential_t u_res = 0.0;
      potential_t u = cortex_leak_and_input(trg, synapse, delta_time);

      // Fire cortex neuron ===============================================================

      if (u > (trg.threshold + curr_layer.threshold_base) && delta_time > 0)
      {
        update_time = e.presynaptic_spike_time;
        if constexpr (!std::is_same_v<Derived, couch_worker_t>)
        {
          create_synapses_events(trg, std::move(e.target_addr), update_time);
          curr_layer.double_counters.inc_by<counters_t<double>::aggregation_type::sum>("spikes_counter", 1);
        }
#ifdef DEBUG_TRACER
        ptracer->push_to_buf(tracer_buf, static_cast<brain_coord_t>(e.target_addr.layer + 2),
                             e.target_addr.row, e.target_addr.col,
                             std::numeric_limits<std::uint8_t>::max(), e.presynaptic_spike_time);
#endif
        u_res = u_rest;

        if constexpr (std::is_same_v<Derived, couch_worker_t>)
        {
          constexpr bool fired = true;
          store_one_metric(e, couching_mode, fired);
        }

        if (couching_mode)
        {
          if constexpr (std::is_same_v<Derived, couch_worker_t>)
          {
            auto label = phead->get_label();
            if (e.target_addr.col == label)
              update_time = e.presynaptic_spike_time;
            else
              update_time = e.presynaptic_spike_time + infinite_delay;
          }
        }
        else
          update_time = e.presynaptic_spike_time;

        trg.prev_spikes = (trg.prev_spikes << (delta_time)) | prev_spikes_t(0x01);
        trg.last_fired = e.presynaptic_spike_time;

        // Put weight to output buffer
        auto ew = weight_event_t{
            e.source_addr,
            e.src_synapse,
            update_time};

        put_to_output_buf<weight_event_t,
                          &tworker_t<Derived>::output_weights_buf,
                          &weight_event_t::addr>(std::move(ew));
        curr_layer.double_counters.inc_by<counters_t<double>::aggregation_type::sum>("weights_counter", 1);
      }
      else
      {
        u_res = u;
        if constexpr (std::is_same_v<Derived, couch_worker_t>)
        {
          constexpr bool not_fired = false;
          store_one_metric(e, couching_mode, not_fired);
        }
      }
      trg.u_mem = u_res;
    }
    p_input_epack->clear();
#ifdef DEBUG_TRACER
    ptracer->display_tracer_buf(tracer_buf, update_time);
#endif
  }
}

template <typename Derived>
void tworker_t<Derived>::cortex_process()
{

  auto couching_mode = phead->couching_mode.load();

  process_input_weights();
  // auto player = phead->layers[static_cast<Derived *>(this)->layer_num];
  // player->ticks_counter.start_tick();
  cortex_process_input_events(couching_mode);
  // player->ticks_counter.stop_tick();
}

// tworker_t output methods ----------------------------------------------------

template <typename Derived>
void tworker_t<Derived>::create_synapses_events(neuron_t &firing_neuron, neuron_address_t &&addr,
                                                clock_count_t time_moment)
{
  for (auto afferr_synapse = firing_neuron.synapses.begin();
       afferr_synapse != firing_neuron.synapses.end();
       ++afferr_synapse)
  {
    auto &tg = afferr_synapse->target_addr;

    neuron_event_t ev{
        addr,
        static_cast<brain_coord_t>(afferr_synapse - firing_neuron.synapses.begin()),
        tg,
        time_moment,
        afferr_synapse->ferment,
        0};

    // Put event to the output buffer
    put_to_output_buf<neuron_event_t,
                      &tworker_t<Derived>::output_events_buf,
                      &neuron_event_t::target_addr>(std::move(ev));
    // float_counters.inc_by<counters_t<double>::avg>("put_to_output_buf", clock() - cdt);
  }
}

template <typename Derived>
template <typename T, auto BufPtr, auto AddrPtr>
void tworker_t<Derived>::put_to_output_buf(T &&ev)
{
  auto target_layer_num = (ev.*AddrPtr).layer;
  auto &buf = (*this).*BufPtr;

  size_t res = std::numeric_limits<size_t>::max();
  for (auto it = buf.begin(); it != buf.end(); ++it)
  {
    if (it->first == target_layer_num)
    {
      res = it - buf.begin();
      break;
    }
  }

  if (res == std::numeric_limits<size_t>::max())
  {
    buf.emplace_back(target_layer_num, std::make_unique<std::vector<T>>());
    res = buf.size() - 1;
  }
  buf[res].second->push_back(std::forward<T>(ev));
}

/**
 * @brief Moves output events/weights from output buffer to the input buffers of the respective workers.
 *
 * @param output_buf the output buffer to move from
 *
 * Throws std::runtime_error if any of the workers' input buffers are full, or if a null pointer is encountered.
 */
template <typename Derived>
template <auto OutputBuf, auto InputBuf>
void tworker_t<Derived>::move_to_workers()
{
  for (auto it = (this->*OutputBuf).begin(); it != (this->*OutputBuf).end(); ++it)
  {
    if (it->second.get())
    {
      auto p = phead->workers[it->first];
      // if constexpr (std::is_same_v<
      //                   typename std::remove_reference_t<decltype(*(it->second))>::value_type,
      //                   neuron_event_t>)
      // {
      //   auto size = it->second->size();
      //   phead->layers[layer_num]->avg_out_ev_counter.add_value(size);
      // }
      // else
      // {
      //   auto size = it->second->size();
      //   phead->layers[layer_num]->double_counters.inc_by<counters_t<double>::avg>("avg_out_wh_counter", size);
      // }

      // Push events
      if (!((*p).*InputBuf).try_push(std::move(it->second)))
        throw std::runtime_error(__PRETTY_FUNCTION__ + std::string(": worker's input buffer is full"));

      // std::this_thread::yield();
    }
  }
  (this->*OutputBuf).clear();
}

// Potentials & Weights updater functions------------------------------------------

potential_t cortex_leaked_u(neuron_t &neuron, clock_count_t delta_time)
{
  if (delta_time <= 0)
    return neuron.u_mem;

  // exponent decay
  potential_t leak_term = std::exp(-cortex_leak_freq * delta_time);
  auto u = neuron.u_mem * leak_term;

  return u;
}

potential_t cortex_leak_and_input(neuron_t &neuron,
                                  synapse_t &synapse,
                                  clock_count_t delta_time)
{
  auto u = cortex_leaked_u(neuron, delta_time) + cortex_signal(synapse);
  return u;
}

potential_t retina_signal(scene_signal_t signal)
{
  return signal * detector_alpha;
}

potential_t retina_leak_and_input([[maybe_unused]] neuron_t &neuron,
                                  scene_signal_t signal,
                                  // std::pair<scene_signal_t,
                                  //           clock_count_t> &
                                  //     timed_memory_signal,
                                  clock_count_t delta_time)
{
  auto signal_ = retina_signal(signal);
  auto leak_term = cortex_leaked_u(neuron, delta_time);
  auto u = leak_term + signal_;
  return u;
}

potential_t calc_dw(clock_count_t dt)
{
  assert(dt >= 0);
  if (dt < zero_dt)
    return pos_dw_rate * (zero_dt - dt);
  else
    return neg_dw_rate * (dt - zero_dt);
}

// head_t functions -------------------------------------------------------------

#ifdef DEBUG_TRACER
void head_t::wake_up(ptracer_t ptracer)
{
  // Init threads
  finish.store(false);
  // Create workers
  auto pp = std::make_shared<retina_worker_t>(this, 0, ptracer);
  workers.push_back(std::move(pp));
  for (unsigned i = 1; i < layers.size() - 1; ++i)
    workers.push_back(std::make_shared<cortex_worker_t>(this, i, ptracer));
  workers.push_back(std::make_shared<couch_worker_t>(this, layers.size() - 1, ptracer));
}
#else
void head_t::wake_up()
{
  // Init threads
  finish.store(false);
  // Create workers
  auto pp = std::make_shared<retina_worker_t>(this, 0);
  workers.push_back(std::move(pp));
  for (unsigned i = 1; i < layers.size() - 1; ++i) // for (unsigned i = 1; i < areas_descr.size() - 1; ++i)
    workers.push_back(std::make_shared<cortex_worker_t>(this, i));
  workers.push_back(std::make_shared<couch_worker_t>(this, layers.size() - 1));
}

#endif
void head_t::go_to_sleep()
{
  // Set finish flag
  finish.store(true);

  // wait for workers to stop
  while (active_workers.load())
    ;
  for (brain_coord_t layer = 0; static_cast<size_t>(layer) < (layers.size()); ++layer)
    workers[layer]->worker_thread.join();
}

// Printing---------------------------------------------------------------
//
void print_image(scene_t *pscene)
{
  for (unsigned i = 0; i < mnist_size; ++i)
  {
    for (unsigned j = 0; j < mnist_size; ++j)
      std::cout << (*pscene)[i][j] << " ";
    std::cout << std::endl;
  }
}

void head_t::print_counters(uint iteration) const
{
  if (iteration == 1)
  {
    logger << "Iteration\tLayer\t";
    counters_t<double>::print_counters_header();
  }
  for (size_t i = 0; i < workers.size(); ++i)
  {
    auto &pworker = workers[i];
    layers[pworker->layer_num]->double_counters.print_counters_as_a_table(iteration, pworker->layer_num);
  }
}

template <typename Derived>
void tworker_t<Derived>::store_one_metric(neuron_event_t &e,
                                          [[maybe_unused]] bool couching_mode,
                                          bool fired)
{
  metrics_t::results_t res;

  auto label = phead->get_label();
  if (fired)
    if (e.target_addr.col == label)
      res = metrics_t::results_t::PT;
    else
      res = metrics_t::results_t::PF;
  else if (e.target_addr.col == label)
    res = metrics_t::results_t::NF;
  else
    res = metrics_t::results_t::NT;
  phead->metrics.store_one_metric(res);
}

double retardation(uint64_t times)
{
  double res = 1;
  while (times--)
    res = exp(-times);
  return res;
}

void head_t::save_choosen_weights(const neuron_address_t &addr, const std::pair<brain_coord_t, brain_coord_t> &direction, const brain_coord_t distance) const
{
  assert(static_cast<size_t>(addr.layer + 1) < layers.size());

  auto &layer = *layers[addr.layer];

  // Make weights buffer
  std::vector<std::vector<double>> one_neuron_weights;
  auto buf_size_x = direction.first * distance;
  auto buf_size_y = direction.second * distance;
  for (auto p = 0; p <= buf_size_x; ++p)
    one_neuron_weights.push_back(std::vector<double>(buf_size_y + 1));

  // Fill buffer for all choosen neurons
  auto max_i = addr.row + buf_size_x;
  auto max_j = addr.col + buf_size_y;
  auto j = addr.col;
  for (auto i = addr.row; i <= max_i && j <= max_j; i += direction.first, j += direction.second)
  {
    auto &neuron = layer.neuron_ref(i, j);

    // Fill buffer for one neuron
    for (auto synapse : neuron.synapses)
      one_neuron_weights[synapse.target_addr.row][synapse.target_addr.col] = synapse.weight;

    // Print buffer

    for (size_t k = 0; k < one_neuron_weights.size(); ++k, logger << std::endl)
      for (size_t l = 0; l < one_neuron_weights[k].size(); ++l, logger << std::endl)
      {
        logger << "Neuron:\t" << i << "\t" << j << "\t" << "Synapse:\t" << k << "\t" << l << "\t";
        logger << std::setprecision(weights_output_precision) << one_neuron_weights[k][l] << "\t";
      }

    // Clear buffer
    for (auto &row : one_neuron_weights)
      std::fill(row.begin(), row.end(), 0.0);
  }
}
