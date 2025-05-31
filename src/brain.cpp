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

using namespace TNN;

// Layer constructor's common ------------------------------------------
template <Is_layer T>
void create_neurons(T *layer, layer_dim_t rows,
                    layer_dim_t cols)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> u(0, 1);
  auto &neurons = layer->neurons;
  neurons.reserve(rows);
  for (layer_dim_t i = 0; i < rows; ++i)
  {
    neurons.emplace_back();
    neurons.back().reserve(cols);
    for (layer_dim_t j = 0; j < cols; ++j)
    {
      auto uu = u(gen);

      neurons.back().emplace_back(uu, initial_neuron_threshold, 0L);
    }
  }
}

// Retina constructors ------------------------------------------------
retina_layer_t::retina_layer_t()
{
  ltype = TNN::layer_type::RETINA;
}

retina_layer_t::retina_layer_t(layer_dim_t rows, layer_dim_t cols)
{
  ltype = TNN::layer_type::RETINA;
  create_neurons(this, rows, cols);
}

// Cortex constructors ------------------------------------------------
cortex_layer_t::cortex_layer_t()
{
  ltype = TNN::layer_type::CORTEX;
}

cortex_layer_t::cortex_layer_t(layer_dim_t rows, layer_dim_t cols)
{
  ltype = TNN::layer_type::CORTEX;
  create_neurons(this, rows, cols);
}

// MNIST couch constructors ------------------------------------------------
mnist_couch_layer_t::mnist_couch_layer_t()
{
  ltype = TNN::layer_type::COUCHING;
};
mnist_couch_layer_t::mnist_couch_layer_t(layer_dim_t rows, layer_dim_t cols)
{
  ltype = TNN::layer_type::COUCHING;
  create_neurons(this, rows, cols);
}

// Head constructors ------------------------------------------------
head_t::head_t()
{
  p_eyes_optics = std::make_shared<eyes_optics_t>(mnist_size, mnist_size, this);

  nof_event_threads = std::thread::hardware_concurrency() / 2 - 1;
}

// tworker_t constructor & destructor ----------------------------------------------------
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
tworker_t<Derived>::~tworker_t()
{
  phead->active_workers--; // Show this worker is not active anymore

  std::chrono::milliseconds timespan(10); // 10 ms
  while (phead->active_workers.load())    // No worker is active
    std::this_thread::sleep_for(timespan);
  worker_thread.join();
}

// Layers's workers -------------------------------------------------------------

void retina_worker_t::worker()
{

  while (!phead->finish.load())
  {

    for (layer_dim_t area_num = 0;
         static_cast<size_t>(area_num) < areas.size();
         area_num++)
    {
      auto time_moment = phead->net_timer.time();
      // Process cortex inputs
      visual_scene_proc(area_num, time_moment);

      // time_moment = phead->net_timer.time();

      cortex_proc(area_num, time_moment);
      move_signals_n_weights_packs_to_workers();
    }
  }
}

void cortex_worker_t::worker()
{
  while (!phead->finish.load())
  {

    for (layer_dim_t area_num = 0;
         static_cast<size_t>(area_num) < areas.size();
         area_num++)
    {
      auto time_moment = phead->net_timer.time();
      // Process cortex inputs
      cortex_proc(area_num, time_moment);

      move_signals_n_weights_packs_to_workers();
    }
  }
}

void mnist_couch_worker_t::worker()
{
  while (!phead->finish.load())
  {

    for (layer_dim_t area_num = 0;
         static_cast<size_t>(area_num) < areas.size();
         area_num++)
    {
      clock_count_t time_moment = 0;
      // Set to zero non-label couch u_mems; highlight label u_mem
      prepare_mnist_couch_umems(area_num, time_moment);

      time_moment = phead->net_timer.time();
      // Process as cortex, consuming prepared u_mems
      cortex_proc(area_num, time_moment);

      move_signals_n_weights_packs_to_workers();
    }
  }
}

// tworker_t process methods ----------------------------------------------------

template <typename Derived>
void tworker_t<Derived>::cortex_proc([[maybe_unused]] layer_dim_t area_num, [[maybe_unused]] clock_count_t time_moment)
{

  std::unique_ptr<std::vector<weight_event_t>> p_wpack;

  // Process weights
  while (input_weights.try_pop(p_wpack))
  {
    if (p_wpack.get() == nullptr)
      break;
    // Proccess input weights pack
    for (auto &we : *p_wpack)
    {
      auto &neuron = phead->neuron_ref(we.addr);
      auto &synapse = neuron.synapses.at(we.synapse_num);
      hebbian_weight_update(neuron, synapse, we.spike_time);
    }
  }

  // Process events
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
      // if (e.target_addr.col == 25 && e.target_addr.row == 25)
      //   std::cout << e.target_addr.layer << " " << time_moment << " " << trg.last_processed << " " << (time_moment - trg.last_processed) << std::endl;
      auto u = cortex_leak_and_input(trg, synapse, e.time_of_arrival);

      // Fire if needed
      potential_t u_res = 0;
      if (u < trg.threshold)
      {
        u_res = u;
        if constexpr (std::is_same_v<Derived, mnist_couch_worker_t>)
        {
          auto label = std::static_pointer_cast<mnist_couch_layer_t>(phead->layers[e.target_addr.layer])->label.load();

          metrics_t::results_t metr;
          if (e.target_addr.col != label)
            metr = metrics_t::results_t::NT;
          else
            metr = metrics_t::results_t::NF;

          static_cast<Derived *>(this)->accuracy.get_accuracy(metr);
        }
      }

      else
      {
        trg.last_fired = e.time_of_arrival;
        pass_event_to_synapses(trg, std::move(e.target_addr), e.time_of_arrival);
#ifdef TRACER_DEBUG
        // Add to tracer buffer
        tracer_buf->push_back(std::make_pair<neuron_address_t, layer_dim_t>({static_cast<layer_dim_t>(e.target_addr.layer + 2),
                                                                             e.target_addr.row,
                                                                             e.target_addr.col},
                                                                            std::numeric_limits<std::uint8_t>::max()));
#endif
        // Put weight to output buffer
        auto ew = weight_event_t{
            e.source_addr,
            e.src_synapse,
            e.time_of_arrival};
        put_weight_to_output_buf(std::move(ew));

        u_res = u_rest;
        if constexpr (std::is_same_v<Derived, mnist_couch_worker_t>)
        {
          auto label = std::static_pointer_cast<mnist_couch_layer_t>(phead->layers[e.target_addr.layer])->label.load();
          metrics_t::results_t metr;
          if (e.target_addr.col != label)
            metr = metrics_t::results_t::PF;
          else
            metr = metrics_t::results_t::PT;

          static_cast<Derived *>(this)->accuracy.get_accuracy(metr);
        }
      }
      trg.u_mem = u_res;
    }
  }
#ifdef TRACER_DEBUG
  // Trace show layers
  ptracer->display_tracer_buf(tracer_buf);
#endif
}

template <typename Derived>
void tworker_t<Derived>::visual_scene_proc([[maybe_unused]] layer_dim_t area_num, clock_count_t time_moment)
{

  // Set shortcuts
  retina_layer_t *p_retina = phead->pretina;
  auto neurons = p_retina->neurons;
  auto area = areas[area_num];
  auto layer_num = area.layer;

  // Process scene inputs
  p_retina->p_eyes_optics->get_locked_scene();
  auto &scene_memories = p_retina->scene_memories;

  auto tracer_buf = std::make_shared<tracer_buf_t>();

  for (layer_dim_t i = area.top; i <= area.bottom; ++i)
  {
    for (layer_dim_t j = area.left; j <= area.right; ++j)
    {
      // Updating potential
      neuron_t &neuron = p_retina->neuron_ref(i, j);
      std::uint8_t scene_val;

      scene_val = p_retina->p_eyes_optics->get_signal(i, j);

      auto &mem_point = scene_memories.at(i).at(j);
      // leak scene memory
      mem_point.first *= exp(-scene_memory_leak_alpha * (time_moment - mem_point.second));

      // std::cout << time_moment << " " << mem_point.second << " " << time_moment - mem_point.second << std::endl;

      auto u = retina_leak_and_input(neuron, scene_val, mem_point, time_moment);
#ifdef TRACER_DEBUG
      // Trace scenes
      tracer_buf->push_back(std::make_pair<neuron_address_t, layer_dim_t>({static_cast<layer_dim_t>(layer_num), i, j},
                                                                          scene_val));

      tracer_buf->push_back(std::make_pair<neuron_address_t, layer_dim_t>({static_cast<layer_dim_t>(layer_num + 1), i, j},
                                                                          mem_point.first));
#endif
      // Firing
      if (u >= neuron.threshold)
      {
        neuron.last_fired = time_moment;
        pass_event_to_synapses(neuron,
                               {static_cast<layer_dim_t>(layer_num), i, j},
                               time_moment);
        u = u_rest;
        tracer_buf->push_back(std::make_pair<neuron_address_t, layer_dim_t>({static_cast<layer_dim_t>(layer_num + 2), i, j},
                                                                            std::numeric_limits<std::uint8_t>::max()));
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

constexpr potential_t couch_u_label = 5.0;

template <typename Derived>
void tworker_t<Derived>::prepare_mnist_couch_umems([[maybe_unused]] layer_dim_t area_num, [[maybe_unused]] clock_count_t time_moment)
{
  auto &neurons = phead->layers.back()->neurons;
  auto label = std::static_pointer_cast<mnist_couch_layer_t>(phead->layers.back())->label.load();

  for (size_t j = 0; j < neurons[0].size(); ++j)
  {
    neurons[0][j].u_mem = 0;
  }

  neurons[0][label].u_mem = couch_u_label;
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
        static_cast<layer_dim_t>(afferr_synapse - firing_neuron.synapses.begin()),
        tg,
        time_moment,
        afferr_synapse->ferment,
        0};

    put_event_to_output_buf(std::move(ev));
  }
}

#define _put_to_output_buf(addr, buff, _type)                               \
  auto area_address = phead->area_address(addr);                            \
  if (!buff.contains(area_address))                                         \
    buff[area_address] = std::move(std::make_unique<std::vector<_type>>()); \
  else if (buff[area_address].get() == nullptr)                             \
    buff[area_address] = std::move(std::make_unique<std::vector<_type>>()); \
  buff[area_address]->push_back(std::move(ev));

template <typename Derived>
void tworker_t<Derived>::put_event_to_output_buf(neuron_event_t &&ev)
{
  _put_to_output_buf(ev.target_addr, output_events_buf, neuron_event_t);
}

template <typename Derived>
void tworker_t<Derived>::put_weight_to_output_buf(weight_event_t &&ev)
{
  _put_to_output_buf(ev.addr, output_weights_buf, weight_event_t);
}

#define _move_to_workers(output_buf, input_buf)                    \
  for (auto it = output_buf.begin(); it != output_buf.end(); ++it) \
  {                                                                \
    auto p = _cast_to_pretina_worker(phead->workers[it->first]);   \
    if (it->second.get() != nullptr)                               \
    {                                                              \
      if (!p->input_buf.try_push(std::move(it->second)))           \
        throw std::runtime_error("Worker is full");                \
    }                                                              \
  }

template <typename Derived>
void tworker_t<Derived>::move_output_events_to_workers()
{
  _move_to_workers(output_events_buf, input_events);
}
template <typename Derived>
void tworker_t<Derived>::move_output_weights_to_workers()
{
  _move_to_workers(output_weights_buf, input_weights);
}

// Potentials & Weights updater functions------------------------------------------

potential_t cortex_leaked_u(neuron_t &neuron, clock_count_t time_moment)
{
  if (!(time_moment - neuron.last_processed)) // Leakage is accounted only once
    return neuron.u_mem;
  if (time_moment - neuron.last_processed < hebb_correlation_time)
    DN(time_moment - neuron.last_processed);
  auto leaked_u = neuron.u_mem * exp(-cortex_leak_alpha * (time_moment - neuron.last_processed));
  neuron.last_processed = time_moment;

  return leaked_u;
}
potential_t cortex_leak_and_input(neuron_t &neuron,
                                  synapse_t &synapse,
                                  clock_count_t time_moment)
{
  auto u = cortex_leaked_u(neuron, time_moment) + synapse.weight * delta_u_mem;
  return u;
}

potential_t retina_leak_and_input(neuron_t &neuron, scene_signal_t signal,
                                  std::pair<scene_signal_t, clock_count_t> &timed_memory_signal,
                                  clock_count_t time_moment)
{
  auto u = cortex_leaked_u(neuron, time_moment);

  auto delta_curr = signal - timed_memory_signal.first;
  if (abs(delta_curr > visual_detector_threshold))
  {
    u += delta_curr * detector_alpha;
    timed_memory_signal.first = signal;
    timed_memory_signal.second = time_moment;
  }
  return u;
}

void hebbian_weight_update(neuron_t &neuron, synapse_t &synapse, clock_count_t afferent_spike_time)
{
  synapse.weight *= exp(-weight_leak_alpha * (afferent_spike_time - neuron.last_fired));
  if (afferent_spike_time - neuron.last_fired <= hebb_correlation_time)
    synapse.weight += delta_weight * (afferent_spike_time - neuron.last_fired) / hebb_correlation_time;
}

// ------------------------------------------------------------------------------
// head_t functions -------------------------------------------------------------

void head_t::make_worker_areas(worker_areas_coll_t &areas_coll, const areas_descr_coll_t &descr_areas_coll)
{
  for (auto descr_areas = descr_areas_coll.begin(); descr_areas != descr_areas_coll.end(); ++descr_areas)
  {
    auto &_areas = areas_coll.emplace_back();
    for (auto desrcr_area = descr_areas->begin(); desrcr_area != descr_areas->end(); ++desrcr_area)
    {
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
                         // {{3, 0, 0, 1, 1}}, // cortex 3
      {{3, 0, 0, 1, 1}}  // couching
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
             std::make_shared<mnist_couch_worker_t>(this,
                                                    areas_coll[areas_coll.size() - 1], ptracer));
}

void head_t::go_to_sleep()
{
  finish.store(true);
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

// Openers----------------------------------------------------------------

// void print_weights_t::operator()(layer_dim_t layer_num, layer_dim_t row_num)
// {
//   std::lock_guard<atomic_mutex> guard(wmutex);
//   auto &layer = phead->layers[layer_num];
//   auto &neurons = layer->neurons;
//   auto &row = neurons[row_num];
//   for (auto col = row.begin(); col != row.end(); ++col)
//   {
//     auto &neuron = layer->neuron_ref(row_num, col - row.begin());
//     for (auto &synapse : neuron.synapses)
//       weights_file << synapse.weight << " ";
//     weights_file << std::endl;
//   }
//   weights_file << std::endl;
// }

// void head_t::print_output(layer_dim_t layer_num)
// {
//   for (size_t i = 0; i < layers.at(layer_num)->neurons.size(); ++i)
//   {
//     for (size_t j = 0; j < layers.at(layer_num)->neurons[i].size(); ++j)
//     {
//       std::cout << std::dynamic_pointer_cast<actuator_layer_t> //
//                    (layers.at(layer_num))->neuron_ref(i, j).value()
//                 << " ";
//     }
//     std::cout << std::endl;
//   }
// }

// Actuator constructors ------------------------------------------------
// actuator_layer_t::actuator_layer_t()
// {
//   ltype = TNN::layer_type::ACTUATOR;
// }

// actuator_layer_t::actuator_layer_t(layer_dim_t rows,
//                                    layer_dim_t cols)
// {
//   ltype = TNN::layer_type::ACTUATOR;
//   create_neurons(this, rows, cols);
// }