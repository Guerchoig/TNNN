#include "input_output.h"
#include "brain.h"
#include <iostream>
#include <vector>
#include <utility>
#include <cstdlib>
#include <random>
#include <atomic>
#include <type_traits>

using namespace TNN;
// Nodes ------------------------------------------------------------------
void neuro_node_t::output(event_t &&ev)
{
  phead->put_event(std::move(ev));
}

void actuator_node_t::output([[maybe_unused]] event_t &&ev)
{
  auto t = phead->net_timer.time();
  while()

}

// Optics ------------------------------------------------------------------
void eyes_optics_t::shift(point_t dir, float dist)
{
  int x_proj = upper_left.first + dir.first * dist;
  x_proj = x_proj < 0 ? 0 : x_proj;
  upper_left.first = x_proj >= static_cast<int>((*pscene).size()) ? (*pscene).size() - 1
                                                                  : x_proj;

  auto x_right = upper_left.first + width;
  width = x_right < (*pscene).size() ? x_right : (*pscene).size() - upper_left.first - 1;

  int y_proj = upper_left.second + dir.second * dist;
  y_proj = y_proj < 0 ? 0 : y_proj;
  upper_left.second = y_proj >= static_cast<int>((*pscene)[0].size()) ? (*pscene)[0].size() - 1
                                                                      : y_proj;

  auto y_right = upper_left.second + heigth;
  heigth = y_right < (*pscene)[0].size() ? y_right
                                         : (*pscene)[0].size() - upper_left.second - 1;
  // DN(upper_left.first);
  // DN(upper_left.second);
  // DN(width);
  // DN(heigth);
  // ND;
}

void eyes_optics_t::look_at(scene_t *_pscene, std::pair<unsigned, unsigned> _upper_left)
{
  // DN(__PRETTY_FUNCTION__);
  pscene = _pscene;
  upper_left = _upper_left;
  set_focus(width, heigth);
  print_image(pscene);
}

void eyes_optics_t::saccade(float dist)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> w(0, dist);
  const point_t dir[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
  dist = w(gen);
  point_t pdir = dir[rand() % 4];
  shift(pdir, dist);
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

// Head ------------------------------------------------
head_t::head_t()
{
  p_eyes_optics = std::make_shared<eyes_optics_t>();
  auto nof_events = std::thread::hardware_concurrency() - 2;
  for (unsigned i = 0; i < nof_events; ++i)
  {
    events.emplace_back(events_cirular_buffer_size);
  }
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

potential_t couch_node_t::input(event_t &&e)
{
  return e.target_addr.ref().u_mem * leak_alpha;
}

void update_potential(event_t &&e)
{
  auto time_moment = phead->net_timer.time();
  auto &trg = e.target_addr.ref();
  trg.busy.lock();
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
    for (auto afferr_synapse = trg.synapses.begin();
         afferr_synapse != trg.synapses.end();
         ++afferr_synapse)
    {
      auto &tg = afferr_synapse->target_addr;

      event_t ev{
          e.target_addr,
          static_cast<layer_dim_t>(afferr_synapse - trg.synapses.begin()),
          tg,
          time_moment + afferr_synapse->delay, 0};

      // afferr_synapse->last_fired = time_moment;
      trg.output(std::move(ev));

      // cur_distance = afferr_synapse->delay;
    }
    u_res = u_rest;
  }
  trg.u_mem = u_res;

  if (e.target_addr.layer == 3 && e.source_addr.layer == 4 /*&& e.src_synapse == 5*/)
    std::cout << "Umem: " << trg.u_mem << '\n';
  trg.update_params(std::forward<event_t>(e), time_moment);
  trg.busy.unlock();
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
  set_focus(width, heigth);

  // Init threads
  finish.store(false);
  unsigned nof_event_threads = 3;
  // std::thread::hardware_concurrency() / 2;

  threads.resize(nof_event_threads);

  // Start threads
  threads[0] = std::move(std::thread(&input_layers_worker));
  for (unsigned i = 1; i < nof_event_threads; ++i)
  {
    threads.at(i) = std::move(std::thread(&internal_layer_worker, i));
  }
}

void head_t::go_to_sleep()
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

void retina_layer_t::worker()
{

  if (moving_look.load())
    return;
  auto max_row = static_cast<layer_dim_t>(phead->p_eyes_optics->upper_left.first + phead->p_eyes_optics->width - 1);
  auto max_col = static_cast<layer_dim_t>(phead->p_eyes_optics->upper_left.second + phead->p_eyes_optics->heigth - 1);
  auto &prev = phead->p_eyes_optics->prev_view;

  // Prepare cicle
  auto prev_eye_row = phead->p_eyes_optics->upper_left.first;
  layer_dim_t neuro_row = prev_eye_row;

  auto &scene = *phead->p_eyes_optics->pscene;
  for (layer_dim_t row = prev_eye_row;
       row <= max_row;
       row++, prev_eye_row++, neuro_row++)
  {
    auto prev_eye_col = phead->p_eyes_optics->upper_left.second;
    auto neuro_col = prev_eye_col;

    for (layer_dim_t col = prev_eye_col;
         col <= max_col;
         col++, prev_eye_col++, neuro_col++)
    {
      auto delta_curr = scene[row][col] - prev[row][col];

      if (abs(delta_curr) >= visual_detector_threshold)
      {
        update_potential({{0, 0, 0}, 0, {0, row, col}, 0L, delta_curr});
        prev[row][col] = scene[row][col];
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
      update_potential({{layer_num, i, j},
                        0, // one synapse per every neuron
                        src_ref.synapses[0].target_addr,
                        phead->net_timer.time(),
                        1});
    }
  }
  for (auto n : neurons_storage)
    n.u_mem = 0.0;
  neurons_storage[label].u_mem = 1.0;
}

void input_layers_worker()
{
  while (!phead->finish.load())
  {
    retina_layer_t::worker();
    mnist_couch_layer_t::worker();
  }
}

// Internal Layers Worker---------------------------------------------------

void internal_layer_worker([[maybe_unused]] unsigned id)
{
  while (!phead->finish.load())
  {

    if (!phead->events.empty())
    {
      if (!phead->events[id].buf.empty())
      {

        auto e = phead->events[id].buf.front();

        phead->events[id].rmutex.lock();
        phead->events[id].buf.pop_front();
        phead->events[id].rmutex.unlock();
        update_potential(std::move(e));
      }
    }
  }
}

// Events processing------------------------------------------------
void head_t::put_event(event_t &&event)
{
  event.time_of_arrival = phead->net_timer.time();
  auto ind = event_index.fetch_add(1, std::memory_order_relaxed);
  if (ind >= events.size())
  {
    ind = 0;
    event_index.store(ind, std::memory_order_relaxed);
  }
  events[ind].wmutex.lock();
  events[ind].buf.push_back(event);
  // print_events();
  events[ind].wmutex.unlock();
}

// Printing------------------------------------------------
void print_image(scene_t *pscene)
{
  for (auto i = 0; i < 28; ++i)
  {
    for (auto j = 0; j < 28; ++j)
      std::cout << (*pscene)[i][j] << " ";
    std::cout << std::endl;
  }
}
void head_t::print_events()
{
  for (unsigned i = 0; i < events.size(); ++i)
  {

    std::cout << events[i].buf.size() << " ";
  }
  std::cout << std::endl;
}

void head_t::print_output(layer_dim_t layer_num)
{
  for (size_t i = 0; i < layers.at(layer_num)->neurons.size(); ++i)
  {
    for (size_t j = 0; j < layers.at(layer_num)->neurons[i].size(); ++j)
    {
      std::cout << layers.at(layer_num)->node_ref(i, j).u_mem << " ";
    }
    std::cout << std::endl;
  }
}

// Openers------------------------------------------------
neuro_node_t &neuron_address_t::ref()
{
  auto &_layer = *(phead->layers.at(layer));
  return _layer.node_ref(row, col);
}
