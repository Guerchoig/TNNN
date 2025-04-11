#pragma once
#include "common.h"
#include "tracer.h"
#include "atomic_queue.h"
#include "eyes_optics.h"
#include <boost/circular_buffer.hpp>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <map>
#include <atomic>
#include <memory>
#include <iostream>
#include <fstream>
#include <type_traits>
#include <tuple>

using input_val_t = int;

// Synapse's params-----------------------------------------------
constexpr double spike_velocity = 1.0;

// Neuron's params-----------------------------------------------
constexpr potential_t delta_u_mem = 0.05;
constexpr potential_t initial_neuron_threshold = 0.7;
constexpr potential_t u_rest = 0.00;
constexpr potential_t leak_alpha = 0.3;

// Retina's neuron  params------------------------------------------------
constexpr input_val_t delta_i_input_min = 1;
constexpr clock_count_t reasonnable_t_acc_max = 1000; // ms

// Visual detector's  params------------------------------------------------
constexpr input_val_t visual_detector_threshold = 2;
constexpr potential_t detector_alpha = initial_neuron_threshold / reasonnable_t_acc_max / delta_i_input_min; // 1e-4

// Weights update  params------------------------------------------------
constexpr potential_t weigth_alpha = 0.1;			  // synapse.weight *= (weigth_alpha + 1)
constexpr clock_count_t weigth_correlation_time = 10; // ms

// Actuator  params-------------------------------------------------
constexpr clock_count_t actuator_tau = 1000; // ms

//
//
// Synapses-----------------------------------------------
struct synapse_t
{
	weight_t weight;
	TNN::ferment_t ferment;

	// @brief is calculated at net building time
	clock_count_t delay;

	neuron_address_t target_addr;

	synapse_t() = default;
	synapse_t(/* clock_count_t last_fired, */
			  weight_t weight,
			  TNN::ferment_t ferment,
			  clock_count_t delay,
			  neuron_address_t &&_target_addr) : // last_fired(last_fired),
												 weight(weight),
												 ferment(ferment),
												 delay(delay),
												 target_addr(_target_addr)
	{
	}

	synapse_t(const synapse_t &other) : // last_fired(other.last_fired),
										weight(other.weight),
										ferment(other.ferment),
										delay(other.delay),
										target_addr(other.target_addr)
	{
	}
};

// Nodes-------------------------------------------------

/**
 * @brief This class represents a neuron in a neural network, with properties
 * and methods to manage its state and behavior
 */
struct neuron_t
{
	potential_t u_mem;
	potential_t threshold;
	clock_count_t last_fired;
	std::vector<synapse_t> synapses;

	neuron_t &operator=(const neuron_t &other)
	{
		u_mem = other.u_mem;
		threshold = other.threshold;
		last_fired = other.last_fired;
		synapses = other.synapses;
		return *this;
	}

	neuron_t() {}

	neuron_t(potential_t u_mem,
			 potential_t threshold,
			 clock_count_t last_fired) : u_mem(u_mem),
										 threshold(threshold),
										 last_fired(last_fired) {}

	neuron_t(const neuron_t &other) : u_mem(other.u_mem),
									  threshold(other.threshold),
									  last_fired(other.last_fired)
	{
		synapses = other.synapses;
	}
	neuron_t(const neuron_t &&other) : u_mem(other.u_mem),
									   threshold(other.threshold),
									   last_fired(other.last_fired)
	{
		synapses = other.synapses;
	}
};

// Workers's types and params -----------------------------------------------

constexpr uint32_t events_q_size = 1024;
constexpr uint32_t weigths_q_size = 1024;

using events_pack_t = std::vector<neuron_event_t>;
using weights_pack_t = std::vector<weight_event_t>;

// @brief arrays of output packs: one pack per each layer

using events_output_buf_t = std::unordered_map<address_t, std::unique_ptr<events_pack_t>>;
using weights_output_buf_t = std::unordered_map<address_t, std::unique_ptr<weights_pack_t>>;

// Worker areas -----------------------------------------------

// See also in input_output.h:
struct area_descr_t
{
	layer_dim_t layer_num;
	layer_dim_t row;
	layer_dim_t col;
	layer_dim_t n_by_rows;
	layer_dim_t n_by_cols;
};
using one_worker_areas_descr_t = std::vector<area_descr_t>;
using areas_descr_coll_t = std::vector<one_worker_areas_descr_t>;

struct worker_area_t
{
	uint16_t layer;
	uint16_t top;
	uint16_t bottom;
	uint16_t left;
	uint16_t right;
};

using one_worker_areas_t = std::vector<worker_area_t>;
using worker_areas_coll_t = std::vector<one_worker_areas_t>;

struct area_bounds_t
{
	layer_dim_t row_side = 0;
	layer_dim_t col_side = 0;
	area_bounds_t() {}
	area_bounds_t(layer_dim_t _row_side,
				  layer_dim_t _col_side) : row_side(_row_side),
										   col_side(_col_side) {}
};

// Base layer -----------------------------------------------

struct layer_t
{
	TNN::layer_type ltype;
	vector_2D_t<neuron_t> neurons;

	neuron_t &neuron_ref([[maybe_unused]] layer_dim_t row, [[maybe_unused]] layer_dim_t col)
	{
		return neurons.at(row).at(col);
	}
};

// Input and output worker queues -----------------------------------------------
template <typename T, size_t S>
struct input_worker_queue_t
{
	atomic_queue::AtomicQueue2<std::unique_ptr<T>, S> queue;
	bool try_push(std::unique_ptr<T> &&p_pack)
	{
		return queue.try_push(std::move(p_pack));
	}

	bool try_pop(std::unique_ptr<T> &p_pack)
	{
		return queue.try_pop(p_pack);
	}
};

// Workers -----------------------------------------------
template <typename Derived> // Curiously Recurring Template Pattern
struct tworker_t
{
	one_worker_areas_t areas;

	// @brief input packs queues
	input_worker_queue_t<events_pack_t, events_q_size> input_events;
	input_worker_queue_t<weights_pack_t, weigths_q_size> input_weights;

	std::thread worker_thread;

	events_output_buf_t output_events_buf;
	weights_output_buf_t output_weights_buf;

	void clear_output_buffers()
	{
		for (auto it = output_events_buf.begin(); it != output_events_buf.end(); ++it)
			it->second->clear();
		for (auto it = output_weights_buf.begin(); it != output_weights_buf.end(); ++it)
			it->second->clear();
	}

	void put_event_to_output_buf(neuron_event_t &&ev);

	// template <typename OutputBufType>
	// void move_output_packs_to_workers(OutputBufType &output_buf);
	void move_output_packs_to_workers(events_output_buf_t &output_buf);
	void move_output_packs_to_workers(weights_output_buf_t &output_buf);

	void move_signals_n_weights_packs_to_workers()
	{
		move_output_packs_to_workers(output_events_buf);
		move_output_packs_to_workers(output_weights_buf);
	}

	void cortex_proc([[maybe_unused]] layer_dim_t area_num);
	void visual_scene_proc([[maybe_unused]] layer_dim_t area_num);
	void mnist_couch_proc([[maybe_unused]] layer_dim_t area_num);

	void pass_event_to_synapses(neuron_t &firing_neuron, neuron_address_t &&addr,
								clock_count_t time_moment);

	void pass_weight_event_to_output_buf(neuron_address_t &src_neuron,
										 layer_dim_t synapse_num,
										 clock_count_t spike_time);

	void execute()
	{
		static_cast<Derived *>(this)->worker();
	}

	tworker_t(const one_worker_areas_t &areas);

	~tworker_t()
	{
		worker_thread.join();
	}
};

struct retina_worker_t : public tworker_t<retina_worker_t> // Curiously Recurring Template Pattern
{
	void worker();
};

struct cortex_worker_t : public tworker_t<cortex_worker_t> // Curiously Recurring Template Pattern
{
	void worker();
};

struct mnist_couch_worker_t : public tworker_t<mnist_couch_worker_t> // Curiously Recurring Template Pattern
{
	void worker();
};

// struct actuator_worker_t : public tworker_t<actuator_worker_t> // Curiously Recurring Template Pattern
// {
// 	void worker(layer_dim_t area_num);
// };

potential_t predict_retina_neuron_potential(const neuron_t &neuron, scene_signal_t signal, scene_signal_t &memory_signal);
potential_t predict_cortex_neuron_potential(neuron_t &neuron, synapse_t &synapse);
void hebb_update_weight(neuron_t &neuron, synapse_t &synapse, clock_count_t afferent_spike_time);

using layers_t = std::vector<std::shared_ptr<layer_t>>;

template <typename T>
concept Is_layer = std::is_base_of_v<layer_t, T>;

template <Is_layer T>
void create_neurons(T *layer, layer_dim_t rows, layer_dim_t cols);

// Derived layers------------------------------------------------

struct retina_layer_t : public layer_t
{
	std::shared_ptr<eyes_optics_t> p_eyes_optics;
	scene_t scene_memories;

	void set_eyes_optics(std::shared_ptr<eyes_optics_t> _p_eyes_optics) { p_eyes_optics = _p_eyes_optics; }
	retina_layer_t();
	retina_layer_t(layer_dim_t rows, layer_dim_t cols);
};

struct cortex_layer_t : layer_t
{
	cortex_layer_t();
	cortex_layer_t(layer_dim_t rows, layer_dim_t cols);
};

struct mnist_couch_layer_t : layer_t
{
	unsigned char label;

	void set_label(layer_dim_t i_label = 0);
	mnist_couch_layer_t();
	mnist_couch_layer_t(layer_dim_t rows, layer_dim_t cols);
};

// head_t ------------------------------------------------
#define cast_to_pretina_worker(p) (std::reinterpret_pointer_cast<retina_worker_t>(p))
struct head_t
{
	layers_t layers;
	std::vector<area_bounds_t> area_bounds_of_layers;

	retina_layer_t *pretina;
	std::shared_ptr<eyes_optics_t> p_eyes_optics;
	net_timer_t net_timer;
	const clock_count_t stdp_delay = 5L;

	std::unordered_map<address_t, std::shared_ptr<void>> workers;

	std::atomic<bool> finish;
	conn_descr_coll_t connections;

	void make_worker_areas(worker_areas_coll_t &areas_coll, const areas_descr_coll_t &descr_areas_coll);

	address_t area_address(neuron_address_t &neuron_addr)
	{
		return address_t(neuron_addr.layer, neuron_addr.row / area_bounds_of_layers[neuron_addr.layer].row_side,
						 neuron_addr.col / area_bounds_of_layers[neuron_addr.layer].col_side);
	}

	std::shared_ptr<void> worker_by_addr(address_t area_addr)
	{
		return workers.find(area_addr)->second;
	}

	void wake_up(scene_t *pscene, unsigned width, unsigned heigth);
	void go_to_sleep();
	// void print_output(layer_dim_t layer_num);
	head_t();
};

inline std::shared_ptr<head_t> phead;

// Global Tracer --------------------------------------------------
inline std::shared_ptr<tracer_t> ptracer;

// Openers --------------------------------------------------

void change_scene(scene_t *_pscene, layer_dim_t _left = 0, layer_dim_t _top = 0);

void print_image(scene_t *pscene);

// struct print_weights_t
// {

// 	std::fstream weights_file;
// 	// ("../networks/weigths.out", std::ios::out | std::ios::trunc);
// 	void operator()(layer_dim_t layer_num, layer_dim_t row_num);
// 	print_weights_t() : wmutex()
// 	{
// 		weights_file.open("../networks/weigths.out", std::ios::out | std::ios::trunc);
// 		weights_file.precision(2);
// 	}
// 	~print_weights_t()
// 	{
// 		weights_file.close();
// 	}
// };
// inline print_weights_t print_weights;

// struct actuator_layer_t : layer_t
// {
// 	vector_2D_t<clock_circular_buffer> clocks;
// 	clock_count_t value(address_t &&addr)
// 	{
// 		return clocks.at(addr.row).at(addr.col).size();
// 	}

// 	actuator_layer_t();
// 	actuator_layer_t(layer_dim_t rows, layer_dim_t cols);
// };
