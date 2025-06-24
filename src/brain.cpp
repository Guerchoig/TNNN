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
  std::uniform_real_distribution<> u(0, 1);

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
        neurons.back().emplace_back(uu, max_neuron_threshold, 0L);
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
  p_eyes_optics = std::make_shared<eyes_optics_t>(mnist_size, mnist_size, this);

  nof_event_threads = std::thread::hardware_concurrency() / 2 - 1;
}

// tworker_t constructor ----------------------------------------------------
template <typename Derived>
tworker_t<Derived>::tworker_t(head_t *phead, const one_worker_areas_t &_areas,
                              ptracer_t &ptracer) : phead(phead), ptracer(ptracer)
{
  for (size_t i = 0; i < _areas.size(); ++i)
    areas.push_back(_areas[i]);
  worker_thread = std::thread(&tworker_t::execute, this);
  phead->active_workers++;
}

template <typename Derived>
void tworker_t<Derived>::execute()
{
  while (!phead->finish.load())
  {

    for (brain_coord_t area_num = 0;
         static_cast<size_t>(area_num) < areas.size();
         area_num++)
    {
      static_cast<Derived *>(this)->worker(area_num);

      move_to_workers<events_output_buf_t, &tworker_t<Derived>::input_events>(output_events_buf);
      move_to_workers<weights_output_buf_t, &tworker_t<Derived>::input_weights>(output_weights_buf);
    }
  }
  phead->active_workers--;
}

// Layers's workers -------------------------------------------------------------

void retina_worker_t::worker(brain_coord_t area_num)
{
  auto retina_time = phead->net_timer.time();
  visual_scene_proc(area_num, retina_time);
  cortex_proccess_input<true>(area_num, retina_time);
}

void cortex_worker_t::worker(brain_coord_t area_num)
{
  cortex_proccess_input<false>(area_num); // time_moment = empty_time
}

void couch_worker_t::worker(brain_coord_t area_num)
{
  cortex_proccess_input<false>(area_num); // time_moment = empty_time
}

// tworker_t process methods ----------------------------------------------------

void update_threshold(neuron_t &neuron, clock_count_t presynaptic_spike_time)
{
  auto delta_time = presynaptic_spike_time - neuron.last_fired;
  if (delta_time <= 0)
    return;
  neuron.threshold = max_neuron_threshold * exp(-neuron_threshold_alpha * (delta_time));
}

template <typename Derived>
void tworker_t<Derived>::cortex_process_weights([[maybe_unused]] brain_coord_t area_num,
                                                [[maybe_unused]] clock_count_t time_moment)
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
      // auto source_type = phead->layers[areas[area_num].layer]->ltype;
      stdp_weight_update(neuron, post_neuron, synapse, we.postsynaptic_spike_time);
    }
  }
}

potential_t cortex_signal(const synapse_t &synapse)
{
  auto du = synapse.weight * membrana_resistance;
  return du;
}

template <typename Derived>
template <bool JustInput>
void tworker_t<Derived>::cortex_process_events([[maybe_unused]] brain_coord_t area_num,
                                               [[maybe_unused]] clock_count_t time_moment,
                                               bool couching_mode)
{

  std::unique_ptr<std::vector<neuron_event_t>> p_epack;
  auto tracer_buf = std::make_shared<tracer_buf_t>();

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
        if (!couching_mode)
        {
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
        }
      };
      // ---------------------------------------------------------------------

      // Calculate u
      potential_t u = 0.0;
      potential_t u_res = 0.0;

      if constexpr (JustInput)
        u = trg.u_mem + cortex_signal(synapse);
      else
        u = cortex_leak_and_input(trg, synapse, e.presynaptic_spike_time);

      // Fire ==============================================================
      clock_count_t update_time = 0;
      if (u < trg.threshold)
      {
        u_res = u;
        update_time = trg.last_fired;
        if constexpr (std::is_same_v<Derived, couch_worker_t>)
          store_metric(false);
      }
      else
      {
        trg.last_fired = e.presynaptic_spike_time;
        trg.trace += delta_trace;
        update_time = e.presynaptic_spike_time;
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
        if constexpr (std::is_same_v<Derived, couch_worker_t>)
          store_metric(true);
      }
      trg.u_mem = u_res;
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
        // Put weight to output buffer
        auto ew = weight_event_t{
            e.source_addr,
            e.src_synapse,
            update_time};

        put_to_output_buf<weight_event_t,
                          &tworker_t<Derived>::output_weights_buf,
                          &weight_event_t::addr>(std::move(ew));
      }
    }
  }
#ifdef TRACER_DEBUG
  // Trace show layers
  ptracer->display_tracer_buf(tracer_buf);
#endif
}

template <typename Derived>
template <bool JustInput>
void tworker_t<Derived>::cortex_proccess_input([[maybe_unused]] brain_coord_t area_num,
                                               [[maybe_unused]] clock_count_t time_moment)
{

  auto couching_mode = phead->couching_mode.load();

  // In couching mode
  if (couching_mode)
    cortex_process_weights(area_num, time_moment);

  cortex_process_events<JustInput>(area_num, time_moment, couching_mode);
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
void tworker_t<Derived>::visual_scene_proc([[maybe_unused]] brain_coord_t area_num,
                                           clock_count_t time_moment)
{
  // Set shortcuts
  retina_layer_t *p_retina = phead->pretina;
  auto neurons = p_retina->neurons;
  auto area = areas[area_num];
  auto layer_num = area.layer;

  // Process scene inputs
  p_retina->p_eyes_optics->get_locked_scene();
  auto &scene_memories = p_retina->scene_memories;

#ifdef TRACER_DEBUG
  auto tracer_buf = std::make_shared<tracer_buf_t>();
#endif
  for (brain_coord_t i = area.top; i <= area.bottom; ++i)
  {
    for (brain_coord_t j = area.left; j <= area.right; ++j)
    {
      // Updating potential
      neuron_t &neuron = p_retina->neuron_ref(i, j);
      std::uint8_t scene_val;

      scene_val = p_retina->p_eyes_optics->get_signal(i, j);
      scene_memories.at(i).at(j) = {scene_val, 0};

      auto &mem_point = scene_memories.at(i).at(j);

      // leak scene memory
      // if (time_moment != mem_point.second)
      //   mem_point.first *= exp(-scene_memory_leak_alpha * (time_moment - mem_point.second));

      auto u = retina_leak_and_input(neuron, scene_val, mem_point, time_moment);
      retardation(20000);

#ifdef TRACER_DEBUG
      // Trace scenes
      if (time_moment % tr::period == 0)
      {
        tracer_buf->push_back(std::make_pair<neuron_address_t, brain_coord_t>({static_cast<brain_coord_t>(layer_num), i, j},
                                                                              scene_val));

        tracer_buf->push_back(std::make_pair<neuron_address_t, brain_coord_t>({static_cast<brain_coord_t>(layer_num + 1), i, j},
                                                                              mem_point.first));
      }
#endif
      // Firing
      if (u >= neuron.threshold)
      {
        neuron.last_fired = time_moment;
        neuron.trace += delta_trace;
        pass_event_to_synapses(neuron,
                               {static_cast<brain_coord_t>(layer_num), i, j},
                               time_moment);
        u = u_rest;
#ifdef TRACER_DEBUG
        if (!time_moment % tr::period == 0)
          tracer_buf->push_back(std::make_pair<neuron_address_t, brain_coord_t>({static_cast<brain_coord_t>(layer_num + 2), i, j},
                                                                                std::numeric_limits<std::uint8_t>::max()));
#endif
      }
      neuron.u_mem = u;
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
  auto area_address = phead->area_address(ev.*AddrPtr);
  auto &buf = (*this).*BufPtr;
  auto it = buf.find(area_address);
  if (it == buf.end())
    buf[area_address] = std::make_unique<std::vector<T>>();
  buf[area_address]->push_back(std::move(ev));
}

/**
 * @brief Moves output events/weights from output buffer to the input buffers of the respective workers.
 *
 * @param output_buf the output buffer to move from
 *
 * Throws std::runtime_error if any of the workers' input buffers are full, or if a null pointer is encountered.
 */
template <typename Derived>
template <typename Tout, auto InputBuf>
void tworker_t<Derived>::move_to_workers(Tout &output_buf)
{
  for (auto it = output_buf.begin(); it != output_buf.end(); ++it)
  {
    auto p = std::static_pointer_cast<tworker_t<Derived>>(phead->workers[it->first]);
    if (p == nullptr)
      throw std::runtime_error(__PRETTY_FUNCTION__ + std::string(": null pointer to worker"));

    if (it->second.get())
    {
#ifdef DEBUG
      if (it->first.layer == 2)
      {
        D("Buf to layer 2 size: ");
        DN(it->second->size());
      }
#endif
      // Increment counters
      if constexpr (std::is_same_v<Tout, events_output_buf_t>)
        events_counter.inc_by(it->second->size());
      else
        weight_events_counter.inc_by(it->second->size());

      // Push events
      if (!((*p).*InputBuf).try_push(std::move(it->second)))
        throw std::runtime_error(__PRETTY_FUNCTION__ + std::string(": worker input buffer full"));
    }
  }
  output_buf.clear();
}

// Potentials & Weights updater functions------------------------------------------

potential_t cortex_leaked_u(neuron_t &neuron, clock_count_t time_moment)
{

  auto delta_time = time_moment - neuron.last_processed;
  if (delta_time <= 0)
    return neuron.u_mem;

  // Approx exponent decay
  potential_t leak_term = 1;
  for (int i = 0; i < delta_time; ++i)
    leak_term *= (1 - cortex_leak_freq);

  neuron.last_processed = time_moment;
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
                                  std::pair<scene_signal_t,
                                            clock_count_t> &timed_memory_signal,
                                  clock_count_t time_moment)
{

  auto u = cortex_leaked_u(neuron, time_moment) + retina_signal(signal);
  timed_memory_signal.first = signal;
  timed_memory_signal.second = time_moment;

  return u;
}

void stdp_weight_update(neuron_t &neuron, neuron_t &post_neuron, synapse_t &synapse, clock_count_t afferent_spike_time)
{
  auto delta_time = afferent_spike_time - neuron.last_fired; // delta_time
  // Decay the spike traces
  if (delta_time != 0)
  {
    neuron.trace *= std::exp(-delta_time / tau_plus);
    post_neuron.trace *= std::exp(-delta_time / tau_minus);
  }

  // Check for pre-synaptic spike (LTP)
  potential_t dw;
  // if (delta_time == 0)
  //   dw = ltp_delta_max * post_neuron.trace - ltd_delta_max * neuron.trace;
  // If post neuron spiked recently (pre before post case)
  if (delta_time >= 0)
    dw = ltp_delta_max * post_neuron.trace;
  else if (delta_time < 0) // Check for post-synaptic spike (LTD)
    dw = -ltd_delta_max * neuron.trace;

  synapse.weight += dw;

  // Apply weight bounds
  synapse.weight = std::clamp(synapse.weight, w_min, w_max);
}

// head_t functions -------------------------------------------------------------

void head_t::make_worker_areas(worker_areas_coll_t &areas_coll, const areas_descr_coll_t &descr_areas_coll)
{
  if (descr_areas_coll.empty())
    throw std::runtime_error(__PRETTY_FUNCTION__ + std::string(": empty areas description"));

  for (auto descr_areas = descr_areas_coll.begin(); descr_areas != descr_areas_coll.end(); ++descr_areas)
  {
    if (descr_areas->empty())
      throw std::runtime_error(__PRETTY_FUNCTION__ + std::string(": empty areas description"));

    auto &_areas = areas_coll.emplace_back();
    for (auto desrcr_area = descr_areas->begin(); desrcr_area != descr_areas->end(); ++desrcr_area)
    {
      if (static_cast<size_t>(desrcr_area->layer_num) >= layers.size())
        throw std::runtime_error(__PRETTY_FUNCTION__ + std::string(": out of bounds layer number"));

      auto layer_num = desrcr_area->layer_num;
      uint32_t rows = layers[layer_num]->neurons.size();
      uint32_t cols = layers[layer_num]->neurons[0].size();
      uint32_t row_side = rows / desrcr_area->n_by_rows;
      uint32_t col_side = cols / desrcr_area->n_by_cols;

      uint32_t top = desrcr_area->row * row_side;
      uint32_t bottom = (desrcr_area->row + 1) * row_side - 1;

      uint32_t left = desrcr_area->col * col_side;
      uint32_t right = (desrcr_area->col + 1) * col_side - 1;

      _areas.emplace_back(layer_num, top, bottom, left, right);

      if (area_bounds_of_layers.size() < static_cast<size_t>(layer_num + 1))
        area_bounds_of_layers.resize(layer_num + 1);
      area_bounds_of_layers[layer_num] = area_bounds_t(row_side, col_side);
    }
  }
}

void head_t::wake_up(ptracer_t ptracer)
{
  // Init threads
  finish.store(false);

  // Create workers
  worker_areas_coll_t areas_coll;
  areas_descr_coll_t areas_descr{
      {{0, 0, 0, 1, 1}}, // retina
      {{1, 0, 0, 1, 1}}, // cortex 1
      {{2, 0, 0, 1, 1}}, // cortex 2
      {{3, 0, 0, 1, 1}}  // couching
                         // {{4, 0, 0, 1, 1}}  // couching
  };

  make_worker_areas(areas_coll, areas_descr);

  // Add workers lambda
  auto add_worker = [this, areas_descr](size_t area_ind, std::shared_ptr<void> p) -> void
  {
    for (auto &area : areas_descr[area_ind])
      workers[address_t(area.layer_num, area.row, area.col)] = p;
  };

  // Add workers
  auto pp = std::make_shared<retina_worker_t>(this, areas_coll[0], ptracer);

  add_worker(0, std::move(pp));

  for (unsigned i = 1; i < areas_descr.size() - 1; ++i)
  {
    add_worker(i, std::make_shared<cortex_worker_t>(this, areas_coll[i], ptracer));
  }

  add_worker(areas_descr.size() - 1,
             std::make_shared<couch_worker_t>(this,
                                              areas_coll[areas_coll.size() - 1], ptracer));
}

void head_t::go_to_sleep()
{
  finish.store(true);
  while (active_workers.load())
    ;
  auto w0 = std::static_pointer_cast<retina_worker_t>(workers[{0, 0, 0}]);
  w0->worker_thread.join();

  for (brain_coord_t layer = 1; static_cast<size_t>(layer) < (layers.size() - 1); ++layer)
  {
    auto w = std::static_pointer_cast<retina_worker_t>(workers[{layer, 0, 0}]);
    w->worker_thread.join();
  }
  auto w1 = std::static_pointer_cast<couch_worker_t>(workers[address_t(static_cast<brain_coord_t>(layers.size() - 1), 0, 0)]);
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

void head_t::print_worker_counters()
{
  for (auto &worker : workers)
  {
    std::stringstream buffer;
    buffer << "layer " << worker.first.layer << " events counter";
    std::static_pointer_cast<tworker_t<cortex_worker_t>>(worker.second)->events_counter.print(buffer.str());
    buffer.str("");
    buffer << "layer " << worker.first.layer << " weight events counter";
    std::static_pointer_cast<tworker_t<cortex_worker_t>>(worker.second)->weight_events_counter.print(buffer.str());
  }
}
