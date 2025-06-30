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

using namespace TNN;
using namespace params;

double retardation(uint64_t times)
{
  double res = 1;
  while (times--)
    res = exp(-times);
  return res;
}

// Layer constructor's common ------------------------------------------
template <Is_layer T>
void create_neurons(T *layer, layer_place_n_size_t place_n_size)
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

// Retina constructors ------------------------------------------------
retina_layer_t::retina_layer_t()
{
  ltype = TNN::layer_type::RETINA;
}

retina_layer_t::retina_layer_t(layer_place_n_size_t place_n_size)
{
  ltype = TNN::layer_type::RETINA;
  create_neurons(this, place_n_size);
}

// Cortex constructors ------------------------------------------------
cortex_layer_t::cortex_layer_t()
{
  ltype = TNN::layer_type::CORTEX;
}

cortex_layer_t::cortex_layer_t(layer_place_n_size_t place_n_size)
{
  ltype = TNN::layer_type::CORTEX;
  create_neurons(this, place_n_size);
}

// MNIST couch constructors ------------------------------------------------
couching_layer_t::couching_layer_t()
{
  ltype = TNN::layer_type::COUCHING;
};
couching_layer_t::couching_layer_t(layer_place_n_size_t place_n_size)
{
  ltype = TNN::layer_type::COUCHING;
  create_neurons(this, place_n_size);
}

// Head constructors ------------------------------------------------
head_t::head_t()
{
  p_eyes_optics = std::make_shared<eyes_optics_t>(mnist_size, mnist_size);

  nof_event_threads = std::thread::hardware_concurrency() / 2 - 1;
}

// tworker_t constructor ----------------------------------------------------
template <typename Derived>
#ifdef TRACER_DEBUG
tworker_t<Derived>::tworker_t(head_t *phead, brain_coord_t layer_num ptracer_t &ptracer) : phead(phead), layer_num(layer_num), ptracer(ptracer)
#else
tworker_t<Derived>::tworker_t(head_t *phead, brain_coord_t layer_num) : phead(phead), layer_num(layer_num)
#endif
{
  worker_thread = std::thread(&tworker_t::execute, this);
  phead->active_workers++;
}

template <typename Derived>
void tworker_t<Derived>::execute()
{
  while (!phead->finish.load())
  {
    // DN(__PRETTY_FUNCTION__);
    static_cast<Derived *>(this)->worker();
    if (output_events_buf.size() > 0)
      move_to_workers<&tworker_t<Derived>::output_events_buf,
                      &tworker_t<Derived>::input_events,
                      &tworker_t<Derived>::events_counter>();
    if (output_weights_buf.size() > 0)
      move_to_workers<&tworker_t<Derived>::output_weights_buf,
                      &tworker_t<Derived>::input_weights,
                      &tworker_t<Derived>::weight_events_counter>();
  }
  phead->active_workers--;
}

// Layers's workers -------------------------------------------------------------

void retina_worker_t::worker()
{
  // D(std::this_thread::get_id());
  // DN(" retina_process_input_events ");
  auto retina_time = phead->net_timer.time();
  retina_process(retina_time);
}

void cortex_worker_t::worker()
{
  // D(std::this_thread::get_id());
  // DN(" cortex_process_input_events ");
  cortex_process(); // time_moment = empty_time
}

void couch_worker_t::worker()
{
  // D(std::this_thread::get_id());
  // DN(" couch_process_input_events ");
  cortex_process(); // time_moment = empty_time
}

// tworker_t process methods ----------------------------------------------------

void update_threshold(neuron_t &neuron, clock_count_t presynaptic_spike_time)
{
  auto delta_time = presynaptic_spike_time - neuron.last_fired;
  if (delta_time <= 0)
    return;
  neuron.threshold = std::min(neuron.threshold + inc_threshold_after_fired - threshold_rally_rate * delta_time, initial_neuron_threshold);
}

template <typename Derived>
void tworker_t<Derived>::process_input_weights([[maybe_unused]] clock_count_t time_moment)
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
      stdp_weight_update(neuron, post_neuron, synapse, we.postsynaptic_spike_time);
    }
  }
}

potential_t cortex_signal(const synapse_t &synapse)
{
  auto du = synapse.weight; // * membrana_resistance;
  return du;
}

template <typename Derived>
// template <bool JustInput >
void tworker_t<Derived>::cortex_process_input_events([[maybe_unused]] clock_count_t time_moment,
                                                     bool couching_mode)
{

  std::unique_ptr<std::vector<neuron_event_t>> p_epack;
#ifdef TRACER_DEBUG
  auto tracer_buf = std::make_shared<tracer_buf_t>();
#endif

  while (input_events.try_pop(p_epack))
  {
    if (p_epack.get() == nullptr)
      break;
    // Process input events pack
    for (auto &e : *p_epack)
    {
      auto &trg = phead->neuron_ref(e.target_addr);
      auto &src = phead->neuron_ref(e.source_addr);
      auto &synapse = src.synapses[e.src_synapse];

      // Store metrics lambda -------------------------------------------------
      auto store_metric = [this, e, couching_mode](bool fired)
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
        phead->metrics.store_metric(res);
      };
      // ---------------------------------------------------------------------

      // Calculate u
      potential_t u = 0.0;
      potential_t u_res = 0.0;

      // if constexpr (JustInput)
      //   u = trg.u_mem + cortex_signal(synapse);
      // else
      u = cortex_leak_and_input(trg, synapse, e.presynaptic_spike_time);

      // Fire cortex neuron ===============================================================

      if (u < trg.threshold)
      {
        u_res = u;
        if constexpr (std::is_same_v<Derived, couch_worker_t>)
        {
          constexpr bool not_fired = false;
          store_metric(not_fired);
        }
      }
      else
      {
        trg.trace += delta_trace;
        clock_count_t update_time = e.presynaptic_spike_time;
        if constexpr (!std::is_same_v<Derived, couch_worker_t>)
          pass_event_to_synapses(trg, std::move(e.target_addr), update_time);
#ifdef TRACER_DEBUG
        // Add to tracer buffer
        if (update_time % tr::period == 0)
          tracer_buf->push_back(std::make_pair<neuron_address_t,
                                               brain_coord_t>({static_cast<brain_coord_t>(e.target_addr.layer + 2),
                                                               e.target_addr.row,
                                                               e.target_addr.col},
                                                              std::numeric_limits<std::uint8_t>::max()));
#endif
        u_res = u_rest;
        update_threshold(trg, e.presynaptic_spike_time);

        if constexpr (std::is_same_v<Derived, couch_worker_t>)
        {
          constexpr bool fired = true;
          store_metric(fired);
        }
        if (couching_mode)
        {
          if constexpr (std::is_same_v<Derived, couch_worker_t>)
          {
            auto label = phead->get_label();
            if (e.target_addr.col == label)
              update_time = e.presynaptic_spike_time;
            else
              update_time = trg.last_fired;
          }
        }
        else
          update_time = e.presynaptic_spike_time;

        trg.last_fired = e.presynaptic_spike_time;

        // Put weight to output buffer
        auto ew = weight_event_t{
            e.source_addr,
            e.src_synapse,
            update_time};

        put_to_output_buf<weight_event_t,
                          &tworker_t<Derived>::output_weights_buf,
                          &weight_event_t::addr>(std::move(ew));
      }
      trg.u_mem = u_res;
    }
    p_epack->clear();
  }
#ifdef TRACER_DEBUG
  // Trace show layers
  ptracer->display_tracer_buf(tracer_buf);
#endif
}

template <typename Derived>
void tworker_t<Derived>::cortex_process([[maybe_unused]] clock_count_t time_moment)
{

  auto couching_mode = phead->couching_mode.load();

  process_input_weights(time_moment);
  cortex_process_input_events(time_moment, couching_mode);
}

template <typename Derived>
clock_count_t tworker_t<Derived>::empty_input_buf_get_time()
{
  clock_count_t time_moment = empty_time;
  std::unique_ptr<std::vector<neuron_event_t>> p_epack;
  while (input_events.try_pop(p_epack))
    if (time_moment == empty_time)
      time_moment = p_epack->at(0).presynaptic_spike_time;

  return time_moment;
}

template <typename Derived>
void tworker_t<Derived>::retina_process(clock_count_t time_moment)
{
  process_input_weights(time_moment);
  retina_process_input_events(time_moment);
}

template <typename Derived>
void tworker_t<Derived>::retina_process_input_events(clock_count_t time_moment)
{
  // Set shortcuts
  retina_layer_t *p_retina = phead->pretina;
  auto neurons = p_retina->neurons;

  // Process scene inputs
  p_retina->p_eyes_optics->get_locked_scene();
  // auto &scene_memories = p_retina->scene_memories;

#ifdef TRACER_DEBUG
  auto tracer_buf = std::make_shared<tracer_buf_t>();
#endif

  for (brain_coord_t i = 0; static_cast<size_t>(i) < phead->layers[layer_num]->neurons.size(); ++i)
  {
    for (brain_coord_t j = 0; static_cast<size_t>(j) < phead->layers[layer_num]->neurons[0].size(); ++j)
    {
      // Updating potential
      neuron_t &trg = p_retina->neurons[i][j];
      std::uint8_t scene_val;

      scene_val = p_retina->p_eyes_optics->get_signal(i, j);
      auto u_res = retina_leak_and_input(trg, scene_val, time_moment);

#ifdef TRACER_DEBUG
      tracer_buf->push_back(std::make_pair<neuron_address_t, brain_coord_t>({static_cast<brain_coord_t>(layer_num), i, j},
                                                                            scene_val));
#endif
      // Fire retina neuron ===============================================================
      if (u_res >= trg.threshold)
      {

        trg.trace += delta_trace;
        pass_event_to_synapses(trg,
                               {static_cast<brain_coord_t>(layer_num), i, j},
                               time_moment);
        u_res = u_rest;
        update_threshold(trg, time_moment);
        trg.last_fired = time_moment;

#ifdef TRACER_DEBUG
        if (!time_moment % tr::period == 0)
          tracer_buf->push_back(std::make_pair<neuron_address_t,
                                               brain_coord_t>({static_cast<brain_coord_t>(layer_num + 2), i, j},
                                                              std::numeric_limits<std::uint8_t>::max()));
#endif
      }
      trg.u_mem = u_res;
    }
  }
  p_retina->p_eyes_optics->unlock_scene();

#ifdef TRACER_DEBUG
  // Trace show scene
  ptracer->display_tracer_buf(tracer_buf);
#endif
}

// tworker_t output methods ----------------------------------------------------

template <typename Derived>
void tworker_t<Derived>::pass_event_to_synapses(neuron_t &firing_neuron, neuron_address_t &&addr,
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
  buf[res].second->push_back(std::move(ev));
}

/**
 * @brief Moves output events/weights from output buffer to the input buffers of the respective workers.
 *
 * @param output_buf the output buffer to move from
 *
 * Throws std::runtime_error if any of the workers' input buffers are full, or if a null pointer is encountered.
 */
template <typename Derived>
template <auto OutputBuf, auto InputBuf, auto Counter>
void tworker_t<Derived>::move_to_workers()
{
  for (auto it = (this->*OutputBuf).begin(); it != (this->*OutputBuf).end(); ++it)
  {
    auto p = std::static_pointer_cast<tworker_t<Derived>>(phead->workers[it->first]);
    if (p == nullptr)
      throw std::runtime_error(__PRETTY_FUNCTION__ + std::string(": null pointer to worker"));
#ifdef DEBUG
    std::stringstream ss;
    ss << __PRETTY_FUNCTION__;
    auto str = ss.str();
    auto rstr = str.substr(str.size() - 90);
#endif
    if (it->second.get())
    {
      ((*p).*Counter).inc_by(it->second->size());
      // Push events
      if (!((*p).*InputBuf).try_push(std::move(it->second)))
        throw std::runtime_error(__PRETTY_FUNCTION__ + std::string(": worker input buffer full"));
      // std::this_thread::yield();
    }
  }
  (this->*OutputBuf).clear();
}

// Potentials & Weights updater functions------------------------------------------

potential_t cortex_leaked_u(neuron_t &neuron, clock_count_t time_moment)
{

  auto delta_time = time_moment - neuron.last_fired;
  if (delta_time <= 0)
    return neuron.u_mem;

  // exponent decay
  potential_t leak_term = std::exp(-cortex_leak_freq * delta_time);
  auto u = neuron.u_mem * leak_term;

  return u;
}

potential_t cortex_leak_and_input(neuron_t &neuron,
                                  synapse_t &synapse,
                                  clock_count_t time_moment)
{
  auto u = cortex_leaked_u(neuron, time_moment) + cortex_signal(synapse);
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
                                  clock_count_t time_moment)
{

  auto u = cortex_leaked_u(neuron, time_moment) + retina_signal(signal);
  // timed_memory_signal.first = signal;
  // timed_memory_signal.second = time_moment;

  return u;
}

void stdp_weight_update(neuron_t &neuron, [[maybe_unused]] neuron_t &post_neuron, synapse_t &synapse, clock_count_t afferent_spike_time)
{
  auto delta_time = afferent_spike_time - neuron.last_fired; // delta_time
  // DN(sgn(delta_time));
  potential_t dw = 0;
  if (delta_time > 0)
    dw = ltp_delta_max * std::exp(-tau_plus * delta_time);
  else if (delta_time == 0)
    dw = ltp_delta_max;
  else if (delta_time < 0)
    dw = -ltd_delta_max * std::exp(-tau_minus * -delta_time);
  synapse.weight += dw;

  // dw = ltp_delta_max * post_neuron.trace - ltd_delta_max * neuron.trace;
  // auto delta_time = afferent_spike_time - neuron.last_fired; // delta_time
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

// head_t functions -------------------------------------------------------------

#ifdef TRACER_DEBUG
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

  // Join threads from first to last
  auto w0 = std::static_pointer_cast<retina_worker_t>(workers[0]);
  w0->worker_thread.join();

  for (brain_coord_t layer = 1; static_cast<size_t>(layer) < (layers.size() - 1); ++layer)
  {
    auto w = std::static_pointer_cast<retina_worker_t>(workers[layer]);
    w->worker_thread.join();
  }
  auto w1 = std::static_pointer_cast<couch_worker_t>(workers[layers.size() - 1]);
  w1->worker_thread.join();
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

template <typename T>
void print_out_one_worker_counters(std::shared_ptr<tworker_t<T>> pworker)
{
  std::stringstream buffer;
  buffer << pworker->layer_num << " events counter: ";
  auto &events_counter = pworker->events_counter;
  events_counter.print(buffer.str());
  auto &weight_events_counter = pworker->weight_events_counter;
  weight_events_counter.print(" weight events counter: ");

  // events_counter.zero();
  // weight_events_counter.zero();
}

void head_t::print_workers_counters()
{
  for (brain_coord_t i = 0; static_cast<size_t>(i) < layers.size(); ++i)
  {
    auto p = std::static_pointer_cast<tworker_t<cortex_worker_t>>(workers[i]);
    auto layer = p->layer_num;
    switch (layers[layer]->ltype)
    {
    case TNN::layer_type::RETINA:
      std::cout << "retina ";
      print_out_one_worker_counters<retina_worker_t>(std::static_pointer_cast<tworker_t<retina_worker_t>>(workers[i]));
      break;
    case TNN::layer_type::CORTEX:
      std::cout << "cortex ";
      print_out_one_worker_counters<cortex_worker_t>(std::static_pointer_cast<tworker_t<cortex_worker_t>>(workers[i]));
      break;
    case TNN::layer_type::COUCHING:
      std::cout << "couching ";
      print_out_one_worker_counters<couch_worker_t>(std::static_pointer_cast<tworker_t<couch_worker_t>>(workers[i]));
      break;
    default:
      break;
    }
  }
}
