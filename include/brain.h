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

namespace params
{
	// Synapse's params-----------------------------------------------
	constexpr double spike_velocity = 1.0;

	// Visual detector's  params------------------------------------------------
	// constexpr input_val_t visual_detector_threshold = 1;
	constexpr potential_t detector_alpha = 0.09 / 255.0;

	// Retina's neuron  params------------------------------------------------
	constexpr input_val_t delta_signal_min = 1;

	// Cortex's neuron params-----------------------------------------------
	constexpr potential_t membrana_resistance = 0.030;
	constexpr potential_t max_neuron_threshold = membrana_resistance * 25;
	constexpr potential_t neuron_threshold_alpha = 1.0E-2;
	constexpr potential_t u_rest = 0.0;
	constexpr potential_t cortex_leak_tau = 30.0; //  ms
	constexpr potential_t cortex_leak_freq = 1 / cortex_leak_tau;
	constexpr potential_t scene_memory_leak_alpha = 1 / (cortex_leak_tau * 20); // 20 times slower than cortex_leak_tau;

	// Weights update  params------------------------------------------------
	constexpr clock_count_t tau_plus = 100;	   // Time constant for pre-synaptic spike trace
	constexpr clock_count_t tau_minus = 140;   // Time constant for post-synaptic spike trace
	constexpr potential_t ltp_delta_max = 0.1; // LTP rate
	constexpr potential_t ltd_delta_max = 0.1; // LTD rate
	constexpr potential_t w_max = 2.0;		   // Maximum weight value
	constexpr potential_t w_min = -0.2;		   // Minimum weight value
	constexpr potential_t delta_trace = 0.1;   // Trace increase delta
}
constexpr clock_count_t empty_time = 0;
constexpr brain_coord_t empty_area = -1;

struct metrics_t
{
	enum results_t
	{
		PT = 0,
		NT = 1,
		PF = 2,
		NF = 3
	};
	std::array<std::atomic<uint64_t>, 4> results = {0, 0, 0, 0};

	void store_metric(results_t res)
	{
		results[res].fetch_add(1, std::memory_order_relaxed);
	}

	void print_metrics()
	{
		auto accuracy = static_cast<float>(results[results_t::PT] + results[results_t::NT]) /
						(results[results_t::PT] + results[results_t::NT] + results[results_t::PF] + results[results_t::NF]);

		auto precision = static_cast<float>(results[results_t::PT]) /
						 (results[results_t::PT] + results[results_t::PF]);

		std::cout << "Accuracy: " << accuracy << " Precision: " << precision
				  << " PT:" << results[results_t::PT]
				  << " NT:" << results[results_t::NT]
				  << " PF:" << results[results_t::PF]
				  << " NF:" << results[results_t::NF] << std::endl;
	}
	void reset()
	{
		for (size_t it = 0; it < results.size(); ++it)
			results[it].store(0);
	}
};

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
	potential_t trace; // Spike trace for STDP

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
	std::atomic<size_t> buffer_size = 0;

	bool try_push(std::unique_ptr<T> &&p_pack)
	{
		DN(buffer_size.load());
		buffer_size++;
		DN(buffer_size.load());
		return queue.try_push(std::move(p_pack));
	}

	bool try_pop(std::unique_ptr<T> &p_pack)
	{
		buffer_size--;
		return queue.try_pop(p_pack);
	}
	input_worker_queue_t()
	{
		buffer_size.store(0);
	}
};

potential_t retina_leak_and_input(neuron_t &neuron, scene_signal_t signal,
								  std::pair<scene_signal_t, clock_count_t> &timed_memory_signal,
								  clock_count_t time_moment);
potential_t cortex_leak_and_input(neuron_t &neuron, synapse_t &synapse, clock_count_t time_moment);

void stdp_weight_update(neuron_t &neuron, neuron_t &post_neuron, synapse_t &synapse, clock_count_t afferent_spike_time);

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
	unsigned int get_label() { return label.load(); }
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
	metrics_t metrics;

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

using events_input_buf_t = input_worker_queue_t<std::vector<neuron_event_t>, events_q_size>;
using weights_input_buf_t = input_worker_queue_t<std::vector<weight_event_t>, weigths_q_size>;

// workers types ------------------------------------------------
template <typename Derived> // Curiously Recurring Template Pattern
struct tworker_t
{
	one_worker_areas_t areas;

	// @brief input packs queues
	events_input_buf_t input_events;
	weights_input_buf_t input_weights;

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

	template <typename T, auto BufPtr, auto AddrPtr>
	void put_to_output_buf(T &&ev);

	template <typename Tout, auto MemberPtr>
	void move_to_workers(Tout &output_buf);

	void process_cortex_weights(brain_coord_t area_num, clock_count_t time_moment);
	template <bool JustInput>
	void process_cortex_events([[maybe_unused]] brain_coord_t area_num,
							   [[maybe_unused]] clock_count_t time_moment,
							   bool couching_mode);
	void visual_scene_proc([[maybe_unused]] brain_coord_t area_num, clock_count_t time_moment);
	template <bool JustInput>
	void cortex_proc([[maybe_unused]] brain_coord_t area_num, [[maybe_unused]] clock_count_t time_moment);
	void couch_proc([[maybe_unused]] brain_coord_t area_num, [[maybe_unused]] clock_count_t time_moment);

	void pass_event_to_synapses(neuron_t &firing_neuron, neuron_address_t &&addr,
								clock_count_t time_moment);
	clock_count_t do_empty_input_events_q();

	void execute();

	tworker_t(head_t *phead, const one_worker_areas_t &areas,
			  ptracer_t &ptracer);
};

struct retina_worker_t : public tworker_t<retina_worker_t> // Curiously Recurring Template Pattern
{
	void worker(brain_coord_t area_num);
	using tworker_t<retina_worker_t>::tworker_t;
};

struct cortex_worker_t : public tworker_t<cortex_worker_t> // Curiously Recurring Template Pattern
{
	void worker(brain_coord_t area_num);
	using tworker_t<cortex_worker_t>::tworker_t;
};

struct couch_worker_t : public tworker_t<couch_worker_t> // Curiously Recurring Template Pattern
{
	void worker(brain_coord_t area_num);
	couch_worker_t(head_t *phead,
				   const one_worker_areas_t &areas,
				   ptracer_t &ptracer) : tworker_t<couch_worker_t>(phead,
																   areas,
																   ptracer)
	{
	}
};
