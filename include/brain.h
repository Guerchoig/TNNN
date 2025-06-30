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

	// Cortex's neuron params-----------------------------------------------
	constexpr potential_t initial_neuron_threshold = 0.9;
	constexpr potential_t threshold_rally_rate = 1.0E-1;
	constexpr potential_t u_rest = 0.0;
	constexpr potential_t inc_threshold_after_fired = 0.5;
	constexpr potential_t cortex_leak_tau = 50.0; //  tics
	constexpr potential_t cortex_leak_freq = 1 / cortex_leak_tau;

	// Visual detector's  params------------------------------------------------
	// constexpr input_val_t visual_detector_threshold = 1;
	constexpr potential_t detector_alpha = initial_neuron_threshold / 2 / 255.0;

	// Weights update  params------------------------------------------------
	constexpr clock_count_t tau_plus = 50;		// Time constant for pre-synaptic spike trace
	constexpr clock_count_t tau_minus = 5;		// Time constant for post-synaptic spike trace
	constexpr potential_t ltp_delta_max = 0.01; // LTP rate
	constexpr potential_t ltd_delta_max = 0.01; // LTD rate
	constexpr potential_t w_max = 2.0;			// Maximum weight value
	constexpr potential_t w_min = 0.0;			// Minimum weight value
	constexpr potential_t delta_trace = 0.2;	// Trace increase delta
}
constexpr clock_count_t empty_time = std::numeric_limits<clock_count_t>::max();

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
		auto positive_total = results[results_t::PT] + results[results_t::PF];
		auto total = positive_total + results[results_t::NT] + results[results_t::NF];

		auto accuracy = 0.0f;
		auto precision = 0.0f;
		if (total)
			accuracy = static_cast<float>(results[results_t::PT] + results[results_t::NT]) / total;
		if (positive_total)
			precision = static_cast<float>(results[results_t::PT]) / positive_total;

		std::cout << "Accuracy: " << accuracy << " Precision: " << precision
				  << " PT:" << results[results_t::PT]
				  << " NT:" << results[results_t::NT]
				  << " PF:" << results[results_t::PF]
				  << " NF:" << results[results_t::NF]
				  << " Total:" << total
				  << std::endl;
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
	std::atomic<size_t> buffer_size;

	bool try_push(std::unique_ptr<T> &&p_pack)
	{

		auto res = queue.try_push(std::move(p_pack));
		// if (res)
		// {
		// 	buffer_size++;
		// 	}
		return res;
	}

	bool try_pop(std::unique_ptr<T> &p_pack)
	{
		auto res = queue.try_pop(p_pack);
		// if (res)
		// {
		// 	buffer_size--;
		// 	D("--");
		// 	DN(buffer_size.load());
		// }
		return res;
	}
	input_worker_queue_t()
	{
		buffer_size.store(0);
	}
};

potential_t retina_leak_and_input(neuron_t &neuron, scene_signal_t signal,
								  //   std::pair<scene_signal_t, clock_count_t> &timed_memory_signal,
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
	// timed_scene_t scene_memories{};

	// void set_eyes_optics(std::shared_ptr<eyes_optics_t> _p_eyes_optics) { p_eyes_optics = _p_eyes_optics; }
	retina_layer_t();
	retina_layer_t(layer_place_n_size_t place_n_size);
	// void clear_scene_memories()
	// {
	// 	for (auto &scene_memory : scene_memories)
	// 		scene_memory.fill({0, 0});
	// }
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
struct head_t
{
	layers_t layers;

	retina_layer_t *pretina;
	std::shared_ptr<eyes_optics_t> p_eyes_optics;
	net_timer_t net_timer;
	const clock_count_t stdp_delay = 5L;

	std::vector<std::shared_ptr<void>> workers;
	std::atomic<int> active_workers;

	std::atomic<bool> finish;
	conn_descr_coll_t connections;
	std::atomic<bool> couching_mode = false;
	metrics_t metrics;

	int get_label()
	{
		return std::static_pointer_cast<couching_layer_t>(layers.back())->label.load();
	}
#ifdef TRACER_DEBUG
	void wake_up(ptracer_t tracer);
#else
	void wake_up();
#endif
	void go_to_sleep();
	// void clear_scene_memory() override final
	// {
	// 	pretina->clear_scene_memories();
	// }

	neuron_t &neuron_ref(address_t &addr)
	{
		auto &_layer = *(layers.at(addr.layer));
		return _layer.neuron_ref(addr.row, addr.col);
	}

	void print_workers_counters();

#ifdef TRACER_DEBUG
	void save_model_to_file(std::string file_name, [[maybe_unused]] std::shared_ptr<tracer_t> ptracer);
	void read_model_from_file(std::string file_name, [[maybe_unused]] std::shared_ptr<tracer_t> ptracer);
#else
	void save_model_to_file(std::string file_name);
	void read_model_from_file(std::string file_name);
#endif

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
	// input packs queues
	events_input_buf_t input_events;
	weights_input_buf_t input_weights;

	std::thread worker_thread;

	events_output_buf_t output_events_buf;
	weights_output_buf_t output_weights_buf;

	head_t *phead;
	brain_coord_t layer_num;
	ptracer_t ptracer;
	counter_t events_counter;
	counter_t weight_events_counter;

	template <auto OutputBuf, auto InputBuf, auto Counter>
	void move_to_workers();

	template <typename T, auto BufPtr, auto AddrPtr>
	void put_to_output_buf(T &&ev);

	void process_input_weights(clock_count_t time_moment);

	void cortex_process_input_events([[maybe_unused]] clock_count_t time_moment,
									 bool couching_mode);
	void cortex_process([[maybe_unused]] clock_count_t time_moment = empty_time);
	void retina_process_input_events(clock_count_t time_moment);
	void retina_process(clock_count_t time_moment);

	void pass_event_to_synapses(neuron_t &firing_neuron, neuron_address_t &&addr,
								clock_count_t time_moment);
	clock_count_t empty_input_buf_get_time();

	void execute();
#ifdef TRACER_DEBUG
	tworker_t(head_t *phead, brain_coord_t layer_num, ptracer_t &ptracer);
#else
	tworker_t(head_t *phead, brain_coord_t layer_num);
#endif
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
	void worker();
	using tworker_t<couch_worker_t>::tworker_t;
};
