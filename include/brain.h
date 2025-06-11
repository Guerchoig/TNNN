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
constexpr potential_t membrana_resistance = 0.02;
constexpr potential_t max_neuron_threshold = membrana_resistance * 30;
constexpr potential_t neuron_threshold_alpha = 1.0E-2;
constexpr potential_t u_rest = 0.0;
constexpr potential_t cortex_tau = 160.0; //  ms
constexpr potential_t cortex_leak_alpha = 1 / cortex_tau;
constexpr potential_t scene_memory_leak_alpha = 1 / (cortex_tau * 20); // 20 times slower than cortex_tau;

// Retina's neuron  params------------------------------------------------
constexpr input_val_t delta_signal_min = 1;

// Visual detector's  params------------------------------------------------
// constexpr input_val_t visual_detector_threshold = 1;
constexpr potential_t detector_alpha = 2.0 / 255.0;

// Weights update  params------------------------------------------------
constexpr potential_t max_delta_weight = 0.2; //
constexpr potential_t weight_tau_plus = 200;  // ms
constexpr potential_t weight_tau_minus = 200; // ms
constexpr potential_t weight_leak_alpha_pos = 1 / weight_tau_plus;
constexpr potential_t weight_leak_alpha_neg = 1 / weight_tau_minus;
// constexpr clock_count_t hebb_correlation_time = 2; // tics

constexpr clock_count_t empty_time = 0;
constexpr brain_coord_t empty_area = -1;

//
//
// Synapses-----------------------------------------------

struct synapse_t
{
	weight_t weight;
	TNN::ferment_t ferment;

	neuron_address_t target_addr;

	synapse_t() = default;
	synapse_t(/* clock_count_t last_fired, */
			  weight_t weight,
			  TNN::ferment_t ferment,
			  neuron_address_t &&_target_addr) : // last_fired(last_fired),
												 weight(weight),
												 ferment(ferment),
												 target_addr(_target_addr)
	{
	}

	synapse_t(const synapse_t &other) : // last_fired(other.last_fired),
										weight(other.weight),
										ferment(other.ferment),
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
	clock_count_t last_fired = 0LL;
	clock_count_t last_processed = 0LL; // To calculate leaks
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

// Worker areas -----------------------------------------------

// See also in input_output.h:
struct area_descr_t
{
	brain_coord_t layer_num;
	brain_coord_t row;
	brain_coord_t col;
	brain_coord_t n_by_rows;
	brain_coord_t n_by_cols;
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
	brain_coord_t row_side = 0;
	brain_coord_t col_side = 0;
	area_bounds_t() {}
	area_bounds_t(brain_coord_t _row_side,
				  brain_coord_t _col_side) : row_side(_row_side),
											 col_side(_col_side) {}
};

// Base layer -----------------------------------------------

struct layer_t
{
	TNN::layer_type ltype;
	vector_2D_t<neuron_t> neurons;

	neuron_t &neuron_ref([[maybe_unused]] brain_coord_t row, [[maybe_unused]] brain_coord_t col)
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

potential_t retina_leak_and_input(neuron_t &neuron, scene_signal_t signal,
								  std::pair<scene_signal_t, clock_count_t> &timed_memory_signal,
								  clock_count_t time_moment);
potential_t cortex_leak_and_input(neuron_t &neuron, synapse_t &synapse, clock_count_t time_moment);

void stdp_weight_update(neuron_t &neuron, synapse_t &synapse, clock_count_t afferent_spike_time);

using layers_t = std::vector<std::shared_ptr<layer_t>>;

template <typename T>
concept Is_layer = std::is_base_of_v<layer_t, T>;

template <Is_layer T>
void create_neurons(T *layer, layer_place_n_size_t place_n_size);

// Derived layers------------------------------------------------

struct retina_layer_t : public layer_t
{
	std::shared_ptr<eyes_optics_t> p_eyes_optics;
	timed_scene_t scene_memories{};

	// void set_eyes_optics(std::shared_ptr<eyes_optics_t> _p_eyes_optics) { p_eyes_optics = _p_eyes_optics; }
	retina_layer_t();
	retina_layer_t(layer_place_n_size_t place_n_size);
	void clear_scene_memories()
	{
		for (auto &scene_memory : scene_memories)
			scene_memory.fill({0, 0});
	}
};

struct cortex_layer_t : layer_t
{
	cortex_layer_t();
	cortex_layer_t(layer_place_n_size_t place_n_size);
};

/**
 * @brief In this layer we store the label of the current image.
 * the synapses just mirror the input synapses to fasciliate weights update
 * in ajacent layers
 */
struct couching_layer_t : layer_t
{
	std::atomic<unsigned char> label;

	void set_label(brain_coord_t i_label = 0)
	{
		label.store(i_label);
	}
	couching_layer_t();
	couching_layer_t(layer_place_n_size_t place_n_size);
};

// head_t ------------------------------------------------
#define _cast_to_pretina_worker(p) (std::reinterpret_pointer_cast<retina_worker_t>(p))
struct head_t : head_interface_t
{
	layers_t layers;
	std::vector<area_bounds_t> area_bounds_of_layers;

	retina_layer_t *pretina;
	std::shared_ptr<eyes_optics_t> p_eyes_optics;
	net_timer_t net_timer;
	const clock_count_t stdp_delay = 5L;

	std::unordered_map<address_t, std::shared_ptr<void>> workers;
	std::atomic<int> active_workers;

	std::atomic<bool> finish;
	conn_descr_coll_t connections;
	std::atomic<bool> couching_mode = false;

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
	int get_label()
	{
		return std::static_pointer_cast<couching_layer_t>(layers.back())->label.load();
	}

	void wake_up(ptracer_t tracer);
	void go_to_sleep();
	void clear_scene_memory() override final
	{
		pretina->clear_scene_memories();
	}

	neuron_t &neuron_ref(address_t &addr)
	{
		auto &_layer = *(layers.at(addr.layer));
		return _layer.neuron_ref(addr.row, addr.col);
	}

	void save_model_to_file(std::string file_name, [[maybe_unused]] std::shared_ptr<tracer_t> ptracer);
	void read_model_from_file(std::string file_name, [[maybe_unused]] std::shared_ptr<tracer_t> ptracer);

	head_t();
};

using phead_t = std::shared_ptr<head_t>;
using ptracer_t = std::shared_ptr<tracer_t>;

// workers types ------------------------------------------------
template <typename Derived> // Curiously Recurring Template Pattern
struct tworker_t
{
	one_worker_areas_t areas;

	// @brief input packs queues
	input_worker_queue_t<std::vector<neuron_event_t>, events_q_size> input_events;
	input_worker_queue_t<std::vector<weight_event_t>, weigths_q_size> input_weights;

	std::thread worker_thread;

	events_output_buf_t output_events_buf;
	weights_output_buf_t output_weights_buf;

	head_t *phead;
	ptracer_t ptracer;

	void clear_output_buffers()
	{
		for (auto it = output_events_buf.begin(); it != output_events_buf.end(); ++it)
			it->second->clear();
		for (auto it = output_weights_buf.begin(); it != output_weights_buf.end(); ++it)
			it->second->clear();
	}

	void put_event_to_output_buf(neuron_event_t &&ev);
	void put_weight_to_output_buf(weight_event_t &&ev);

	// template <typename OutputBufType>
	// void move_output_packs_to_workers(OutputBufType &output_buf);
	void move_output_events_to_workers();
	void move_output_weights_to_workers();

	void move_signals_n_weights_packs_to_workers()
	{
		move_output_events_to_workers();
		move_output_weights_to_workers();
	}

	void process_cortex_weights(brain_coord_t area_num, clock_count_t time_moment);
	void process_cortex_events([[maybe_unused]] brain_coord_t area_num,
							   [[maybe_unused]] clock_count_t time_moment,
							   bool couching_mode);
	void visual_scene_proc([[maybe_unused]] brain_coord_t area_num, clock_count_t time_moment);
	void cortex_proc([[maybe_unused]] brain_coord_t area_num, [[maybe_unused]] clock_count_t time_moment);
	void couch_proc([[maybe_unused]] brain_coord_t area_num, [[maybe_unused]] clock_count_t time_moment);

	void pass_event_to_synapses(neuron_t &firing_neuron, neuron_address_t &&addr,
								clock_count_t time_moment);

	// void put_weight_to_output_buf(neuron_address_t &src_neuron,
	// 							   brain_coord_t synapse_num,
	// 							   clock_count_t spike_time);

	void execute()
	{
		static_cast<Derived *>(this)->worker();
	}

	tworker_t(head_t *phead, const one_worker_areas_t &areas,
			  ptracer_t &ptracer);
};

struct retina_worker_t : public tworker_t<retina_worker_t> // Curiously Recurring Template Pattern
{
	void worker();
	using tworker_t<retina_worker_t>::tworker_t;
};

struct cortex_worker_t : public tworker_t<cortex_worker_t> // Curiously Recurring Template Pattern
{
	void worker();
	using tworker_t<cortex_worker_t>::tworker_t;
};

struct couch_worker_t : public tworker_t<couch_worker_t> // Curiously Recurring Template Pattern
{
	metrics_t metrics;
	void worker();
	couch_worker_t(head_t *phead,
				   const one_worker_areas_t &areas,
				   ptracer_t &ptracer) : tworker_t<couch_worker_t>(phead,
																   areas,
																   ptracer)
	{
	}
};
