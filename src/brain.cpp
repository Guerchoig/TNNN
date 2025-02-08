#include "input_output.h"
#include "brain.h"
#include "clock_circular_buffer.h"
#include "tracer.h"
#include <iostream>
#include <vector>
#include <utility>
#include <cstdlib>
#include <random>
#include <atomic>
#include <type_traits>

using namespace TNN;
// Nodes ------------------------------------------------------------------
unsigned actuator_node_t::value()
{
  return clocks.size();
}

int ajust_boundaries(int coord, const int left_b, const int right_b)
{
  int res;
  res = coord;
  if (res < left_b)
    res = left_b;
  if (res > right_b)
    res = right_b;
  return res;
}

// Optics ------------------------------------------------------------------
void eyes_optics_t::shift(int dx, int dy, float dist)
{
  assert(dist > 0);
  int delta_x = dx * dist;
  int delta_y = dy * dist;
  left += delta_x;
  right += delta_x;
  top += delta_y;
  bottom += delta_y;
}

void eyes_optics_t::set_focus(int _left, int _top, layer_dim_t _width, layer_dim_t _heigth)
{
  left = _left;
  top = _top;
  right = left + _width - 1;
  bottom = top + _heigth - 1;
}

void eyes_optics_t::look_at(scene_t *_pscene, layer_dim_t _left, layer_dim_t _top)
{
  pscene = _pscene;
  set_focus(_left, _top, (*pscene)[0].size(), (*pscene).size());
  // DF(print_image(pscene));
}

void eyes_optics_t::saccade(float dist)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> w(1, dist);
  const std::pair<int, int> dir[] = {{0, 1}, {0, -1}, {1, 0}, {1, 1}, //
                                     {1, -1},
                                     {-1, 1},
                                     {-1, -1},
                                     {-1, 0}};
  dist = w(gen);
  auto pdir = dir[rand() % (sizeof(dir) / sizeof(dir[0]))];
  shift(pdir.first, pdir.second, dist);
}

// Layer's constructors ------------------------------------------------
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
      T::neurons_storage.emplace_back(uu, initial_neuron_threshold, 0L);
      neurons[i].push_back(T::neurons_storage.size() - 1);
    }
  }
}

retina_layer_t::retina_layer_t()
{
  p_eyes_optics = phead->p_eyes_optics;
  moving_look = false;
}

retina_layer_t::retina_layer_t(TNN::layer_type _type) : layer_t(_type)
{
  p_eyes_optics = phead->p_eyes_optics;
  moving_look = false;
}

retina_layer_t::retina_layer_t(retina_layer_t &&other)
    : p_eyes_optics(std::move(other.p_eyes_optics))
{
  moving_look.store(other.moving_look.load());
}

retina_layer_t::retina_layer_t(layer_dim_t rows,
                               layer_dim_t cols,
                               TNN::layer_type _type)
{
  type = _type;
  create_neurons(this, rows, cols);
}

cortex_layer_t::cortex_layer_t(layer_dim_t rows,
                               layer_dim_t cols,
                               TNN::layer_type _type)
{
  type = _type;
  create_neurons(this, rows, cols);
}

mnist_couch_layer_t::mnist_couch_layer_t(layer_dim_t rows,
                                         layer_dim_t cols,
                                         TNN::layer_type _type)
{
  type = _type;
  create_neurons(this, rows, cols);
}

actuator_layer_t::actuator_layer_t(layer_dim_t rows,
                                   layer_dim_t cols,
                                   TNN::layer_type _type)
{
  type = _type;
  create_neurons(this, rows, cols);
}

void mnist_couch_layer_t::set_label(layer_dim_t i_label)
{
  label = i_label;
  for (auto &n : neurons_storage)
    n.u_mem = 0.0; // neurons_storage.;
  neurons_storage[label].u_mem = 1.0;
}

// Head constructors ------------------------------------------------
head_t::head_t()
{
  p_eyes_optics = std::make_shared<eyes_optics_t>();

  nof_event_threads = std::thread::hardware_concurrency() / 2 - 1;
  // nof_event_threads = 2;
}

// Updates------------------------------------------------

potential_t retina_node_t::input(event_t &&e)
{
  return detector_alpha * e.signal;
}

potential_t cortex_node_t::input(event_t &&e)
{
  auto &inferr_synapse = e.source_addr.ref().synapses[e.src_synapse];
  return inferr_synapse.weight * delta_u_mem;
}

potential_t actuator_node_t::input(event_t &&e)
{
  auto &inferr_synapse = e.source_addr.ref().synapses[e.src_synapse];
  return inferr_synapse.weight * delta_u_mem;
}

potential_t couch_node_t::input(event_t &&e)
{
  return e.target_addr.ref().u_mem * leak_alpha;
}


void actuator_node_t::output([[maybe_unused]] base_worker_t &worker, [[maybe_unused]] event_t &&ev)
{
  auto t = phead->net_timer.time();
  clocks.push(t);
  while (clocks.peek() < t - actuator_tau)
    clocks.pop();
}

void update_final_potential(base_worker_t &worker, event_t &&e)
{
  auto time_moment = phead->net_timer.time();
  auto &trg = e.target_addr.ref();
  std::lock_guard<atomic_mutex> guard(trg.busy);
  // Add input
  auto u = trg.u_mem * (1 - leak_alpha) + trg.input(std::forward<event_t>(e));

  potential_t u_res = 0;
  if (u < trg.threshold)
  {
    u_res = u;
  }
  else
  {
    // distance_t cur_distance = 0;
    trg.last_fired = time_moment;

    // afferr_synapse->last_fired = time_moment;
    trg.output(worker, std::forward<event_t>(e));
    u_res = u_rest;
    // cur_distance = afferr_synapse->delay;
  }

  trg.u_mem = u_res;

  if (e.target_addr.layer != 0)
    trg.update_params(std::forward<event_t>(e), time_moment);
}

void update_potential(base_worker_t &worker, event_t &&e)
{
  auto time_moment = phead->net_timer.time();
  auto &trg = e.target_addr.ref();

  std::lock_guard<atomic_mutex> guard(trg.busy);
  // Add input
  auto u = trg.u_mem * (1 - leak_alpha) + trg.input(std::forward<event_t>(e)) * static_cast<clock_count_t>(e.ferment);

  potential_t u_res = 0;
  if (u < trg.threshold)
  {
    u_res = u;
  }
  else
  {
    // distance_t cur_distance = 0;
    trg.last_fired = time_moment;
    for (auto afferr_synapse = trg.synapses.begin();
         afferr_synapse != trg.synapses.end();
         ++afferr_synapse)
    {
      auto &tg = afferr_synapse->target_addr;

      event_t ev{
          e.target_addr,
          static_cast<layer_dim_t>(afferr_synapse - trg.synapses.begin()),
          tg,
          time_moment + afferr_synapse->delay, afferr_synapse->ferment, 0};

      // afferr_synapse->last_fired = time_moment;
      trg.output(worker, std::move(ev));

      // cur_distance = afferr_synapse->delay;
    }
    u_res = u_rest;
  }
  trg.u_mem = u_res;

  // if (e.target_addr.layer == 3 && e.source_addr.layer == 4 /*&& e.src_synapse == 5*/)
  //   std::cout << "Umem: " << trg.u_mem << '\n';
  trg.update_params(std::forward<event_t>(e), time_moment);
}

void cortex_node_t::update_params(event_t &&e, clock_count_t time_moment)
{
  auto &psrc = e.source_addr.ref();
  psrc.busy.lock();
  synapse_t &src_synapse = psrc.synapses[e.src_synapse];
  src_synapse.weight = src_synapse.weight * (1 - weigth_alpha);
  if (abs(time_moment - psrc.last_fired) < phead->stdp_delay)
    src_synapse.weight += delta_weight;
  psrc.busy.unlock();
}

void head_t::wake_up(scene_t *pscene, unsigned width, unsigned heigth)
{
  // Init eyes
  look_at(pscene);
  set_focus(0, 0, width, heigth);

  // Init threads
  finish.store(false);
  threads.resize(nof_event_threads);

  // Start threads
  threads[0] = std::thread(&input_layers_worker);
  for (unsigned i = 1; i < nof_event_threads; ++i)
  {
    threads.at(i) = std::thread(&internal_layer_worker, i);
  }
}

void head_t::do_sleep()
{
  finish.store(true);
  while (!threads.empty())
    if (threads.at(0).joinable())
    {
      try
      {
        threads.at(0).join();
      }
      catch (...)
      {
        exit(5);
      }
      threads.erase(threads.begin());
    }
}

// Input Layers Workers---------------------------------------------------
//
void worker

void io_worker_t::worker(const retina_layer_t &layer)
{

  if (layer.moving_look.load())
    return;
  auto &neurons = phead->layers[0]->neurons;
  auto optics = phead->p_eyes_optics;

  auto left_b = std::max(static_cast<int>(optics->left), 0);
  auto right_b = std::min(static_cast<int>(optics->right), static_cast<int>(neurons[0].size() - 1));
  auto top_b = std::max(static_cast<int>(optics->top), 0);
  auto bottom_b = std::min(static_cast<int>(optics->bottom), static_cast<int>(neurons.size() - 1));

  // float x_factor = neurons[0].size() / (right_b - left_b + 1);
  // float y_factor = neurons.size() / (bottom_b - top_b + 1);
  auto &prev = optics->prev_view;
  auto &scene = *optics->pscene;

  // Prepare cicle
  for (layer_dim_t eye_row = left_b;
       eye_row <= right_b;
       eye_row++)
  {
    layer_dim_t neuro_row = (eye_row - left_b); //* y_factor;

    for (layer_dim_t eye_col = top_b;
         eye_col <= bottom_b;
         eye_col++)
    {
      layer_dim_t neuro_col = (eye_col - top_b); //* x_factor;
      auto delta_curr = abs(scene[eye_row][eye_col] - prev[eye_row][eye_col]);

      if (delta_curr >= visual_detector_threshold)
      {
        event_t e{{-1, eye_row, eye_col}, 0, {0, neuro_row, neuro_col}, 0L, TNN::DOPHAMINE, delta_curr};
        auto &trg = e.target_addr.ref();
        trg.output(this, std::move(e));
        prev[eye_row][eye_col] = scene[eye_row][eye_col];
      }
    }
  }
}

void mnist_couch_layer_t::worker()
{
  auto layer_num = static_cast<layer_dim_t>(phead->layers.size() - 1);
  auto &neurons = phead->layers.back()->neurons;

  for (layer_dim_t i = 0; i < static_cast<layer_dim_t>(neurons.size()); ++i)
  {
    for (layer_dim_t j = 0; j < static_cast<layer_dim_t>(neurons[i].size()); ++j)
    {
      auto &src_ref = phead->layers.back()->node_ref(i, j);
      layer_dim_t synapse_index = 0;
      update_final_potential({{layer_num, i, j},
                              synapse_index, // one synapse per every neuron
                              src_ref.synapses[synapse_index].target_addr,
                              phead->net_timer.time(),
                              src_ref.synapses[synapse_index].ferment,
                              1});
    }
  }
  // for (auto n : neurons_storage)
  //   n.u_mem = 0.0;
  // neurons_storage[label].u_mem = 1.0;
}

void input_layers_worker([[maybe_unused]] const retina_layer_t &layer)
{
  while (!phead->finish.load())
  {
    retina_layer_t::worker();
    mnist_couch_layer_t::worker();
  }
}

// Internal Layers Worker---------------------------------------------------
//
void internal_layer_worker_t::worker([[maybe_unused]] const retina_layer_t &layer)
{
  while (!phead->finish.load())
  {
    if (!events.empty())
    {
      auto e = events.front();
      events.dequeue();
      update_potential(std::move(e));
    }
  }
}

// Events processing------------------------------------------------
//

// Printing---------------------------------------------------------------
//
void print_image(scene_t *pscene)
{
  for (auto i = 0; i < 28; ++i)
  {
    for (auto j = 0; j < 28; ++j)
      std::cout << (*pscene)[i][j] << " ";
    std::cout << std::endl;
  }
}

void head_t::print_output(layer_dim_t layer_num)
{
  for (size_t i = 0; i < layers.at(layer_num)->neurons.size(); ++i)
  {
    for (size_t j = 0; j < layers.at(layer_num)->neurons[i].size(); ++j)
    {
      std::cout << std::dynamic_pointer_cast<actuator_layer_t> //
                   (layers.at(layer_num))->node_ref(i, j).value()
                << " ";
    }
    std::cout << std::endl;
  }
}

// Openers----------------------------------------------------------------

neuro_node_t &neuron_address_t::ref()
{
  auto &_layer = *(phead->layers.at(layer));
  return _layer.node_ref(row, col);
}

void print_weights_t::operator()(layer_dim_t layer_num, layer_dim_t row_num)
{
  std::lock_guard<atomic_mutex> guard(wmutex);
  auto &layer = phead->layers[layer_num];
  auto &neurons = layer->neurons;
  auto &row = neurons[row_num];
  for (auto col = row.begin(); col != row.end(); ++col)
  {
    auto &neuron = layer->node_ref(row_num, col - row.begin());
    for (auto &synapse : neuron.synapses)
      weights_file << synapse.weight << " ";
    weights_file << std::endl;
  }
  weights_file << std::endl;
}
