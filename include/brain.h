#pragma once
#include "common.h"
#include "tracer.h"
#include "atomic_queue.h"
#include "eyes_optics.h"
#include "counters.h"
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
#include <bitset>

using input_val_t = int;
using prev_spikes_t = std::bitset<8>;

namespace params
{
	// Synapse's params-----------------------------------------------
	constexpr double spike_velocity = 1.0;

	// Cortex's neuron params-----------------------------------------------
	constexpr potential_t u_rest = 0.0;
	constexpr potential_t initial_neuron_threshold = 6;
	// constexpr potential_t threshold_rally_rate = 0.4;
	// constexpr potential_t inc_threshold_after_fired = 1.5;
	// constexpr potential_t threshold_increment = 0.1; // M_PI;
	constexpr potential_t high_threshold_base = 5.0; // M_PI;
	constexpr potential_t low_threshold_base = -1;
	constexpr potential_t normal_threshold_base = 0.0;

	constexpr uint nof_classes = 10;
	constexpr potential_t max_firing_percentage = 1.0 / nof_classes;
	constexpr potential_t min_firing_percentage = 0.03;

	constexpr potential_t cortex_leak_tau = 20.0; //  tics
	constexpr potential_t cortex_leak_freq = 1 / cortex_leak_tau;

	// Visual detector's  params------------------------------------------------
	// constexpr input_val_t visual_detector_threshold = 1;
	constexpr potential_t detector_alpha = initial_neuron_threshold * 1.0 / 255.0;

	// Weights update  params------------------------------------------------
	// constexpr clock_count_t tau_plus = 50;		// Time constant for pre-synaptic spike trace
	// constexpr clock_count_t tau_minus = 5;		// Time constant for post-synaptic spike trace
	// constexpr potential_t ltp_delta_max = 0.1;	// LTP rate
	// constexpr potential_t ltd_delta_max = 0.01; // LTD rate
	// constexpr potential_t w_max = 2.0;			// Maximum weight value
	// constexpr potential_t w_min = 0.0;			// Minimum weight value
	// constexpr potential_t delta_trace = 0.2;	// Trace increase delta
	constexpr potential_t dw_max = 0.8;
	constexpr potential_t dw_min = -0.5;
	constexpr potential_t zero_dt = 3.5;
	constexpr clock_count_t max_dt = sizeof(prev_spikes_t) - 1;
	constexpr potential_t neg_dw_rate = dw_min / (max_dt - zero_dt);
	constexpr potential_t pos_dw_rate = dw_max / zero_dt;
	constexpr clock_count_t infinite_delay = max_dt + 1;
}

// slowdown values
constexpr int slowdown_apply = 1;
constexpr int slowdown_donothing = 0;
constexpr int slowdown_cancel = -1;

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

	void store_one_metric(const results_t &res)
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
	synapse_t(weight_t weight,
			  TNN::ferment_t ferment,
			  neuron_address_t &&_target_addr) : weight(weight),
												 ferment(ferment),
												 target_addr(_target_addr)
	{
	}

	synapse_t(const synapse_t &other) : weight(other.weight),
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

	// Bitmap of current and previous 15 spikes for STDP
	prev_spikes_t prev_spikes{0x01};

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
	// avg_counter_t avg_out_ev_counter;
	counters_t<double> double_counters;
	std::atomic<int> slowdown = 0;
	std::atomic<potential_t> threshold_base = params::normal_threshold_base;

	neuron_t &neuron_ref([[maybe_unused]] brain_coord_t row, [[maybe_unused]] brain_coord_t col)
	{
		return neurons.at(row).at(col);
	}
	layer_t()
	{
		double_counters.add<counters_t<double>::sum>("spikes_counter");
		double_counters.add<counters_t<double>::sum>("weights_counter");
	}
};

template <typename T, size_t S>
using input_worker_queue_t = atomic_queue::AtomicQueue2<std::unique_ptr<T>, S>;
// Input and output worker queues -----------------------------------------------
// template <typename T, size_t S>
// struct input_worker_queue_t
// {
// 	atomic_queue::AtomicQueue2<std::unique_ptr<T>, S> queue;
// 	// std::atomic<size_t> buffer_size;

// 	bool try_push(std::unique_ptr<T> &&p_pack)
// 	{

// 		auto res = queue.try_push(std::move(p_pack));

// 		return res;
// 	}

// 	bool try_pop(std::unique_ptr<T> &p_pack)
// 	{
// 		auto res = queue.try_pop(p_pack);

// 		return res;
// 	}
// 	// input_worker_queue_t()
// 	// {
// 	// 	buffer_size.store(0);
// 	// }
// };

potential_t retina_leak_and_input(neuron_t &neuron, scene_signal_t signal,
								  //   std::pair<scene_signal_t, clock_count_t> &timed_memory_signal,
								  clock_count_t delta_time);
potential_t cortex_leak_and_input(neuron_t &neuron, synapse_t &synapse, clock_count_t delta_time);

// void stdp_weight_update(neuron_t &neuron,
// 						[[maybe_unused]] neuron_t &post_neuron,
// 						synapse_t &synapse,
// 						clock_count_t postsynaptic_spike_time,
// 						[[maybe_unused]] brain_coord_t synapse_num);

using layers_t = std::vector<std::shared_ptr<layer_t>>;

template <typename T>
concept Is_layer = std::is_base_of_v<layer_t, T>;

template <Is_layer T>
void create_neurons(T *layer, const layer_place_n_size_t &place_n_size);

// Derived layers------------------------------------------------

struct retina_layer_t : public layer_t
{
	std::shared_ptr<eyes_optics_t> p_eyes_optics;
	// timed_scene_t scene_memories{};

	// void set_eyes_optics(std::shared_ptr<eyes_optics_t> _p_eyes_optics) { p_eyes_optics = _p_eyes_optics; }
	retina_layer_t();
	retina_layer_t(const layer_place_n_size_t &place_n_size);
	// void clear_scene_memories()
	// {
	// 	for (auto &scene_memory : scene_memories)
	// 		scene_memory.fill({0, 0});
	// }
};

struct cortex_layer_t : layer_t
{
	cortex_layer_t();
	cortex_layer_t(const layer_place_n_size_t &place_n_size);
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
	couching_layer_t(const layer_place_n_size_t &place_n_size);
};

// Worker's interface class ------------------------------------------------
struct head_t;
using phead_t = std::shared_ptr<head_t>;
using ptracer_t = std::shared_ptr<tracer_t>;

using events_input_buf_t = input_worker_queue_t<std::vector<neuron_event_t>, events_q_size>;
using weights_input_buf_t = input_worker_queue_t<std::vector<weight_event_t>, weigths_q_size>;

// constexpr size_t zero_dt_index = 25;
// constexpr size_t nof_dt_counters = 50;
struct worker_base_t
{
	brain_coord_t layer_num;
	std::thread worker_thread;

	// input packs queues
	events_input_buf_t input_events;
	weights_input_buf_t input_weights;

	// output packs queues
	events_output_buf_t output_events_buf;
	weights_output_buf_t output_weights_buf;

	head_t *phead;

	virtual ~worker_base_t() = default;
	virtual void execute() = 0;
	virtual const char *type_name() const = 0;
};

// head_t ------------------------------------------------
// #define _cast_to_pretina_worker(p) (std::reinterpret_pointer_cast<retina_worker_t>(p))
struct head_t
{
	layers_t layers;

	retina_layer_t *pretina;
	std::shared_ptr<eyes_optics_t> p_eyes_optics;
	net_timer_t net_timer;
	const clock_count_t stdp_delay = 5L;

	std::vector<std::shared_ptr<worker_base_t>> workers;

	std::atomic<int> active_workers;

	std::atomic<bool> finish;
	conn_descr_coll_t connections;
	std::atomic<bool> couching_mode = false;
	metrics_t metrics;

	int get_label()
	{
		return std::static_pointer_cast<couching_layer_t>(layers.back())->label.load();
	}
#ifdef DEBUG_TRACER
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

	void print_counters(uint iteration) const;

#ifdef DEBUG_TRACER
	void save_model_to_file(std::string file_name, [[maybe_unused]] std::shared_ptr<tracer_t> ptracer);
	void read_model_from_file(std::string file_name, [[maybe_unused]] std::shared_ptr<tracer_t> ptracer);
#else
	void save_model_to_file(std::string file_name);
	void read_model_from_file(std::string file_name);
#endif
	void save_choosen_weights(const neuron_address_t &addr,
							  const std::pair<brain_coord_t, brain_coord_t> &direction,
							  const brain_coord_t distance) const;
	head_t();
};

// workers types ------------------------------------------------
struct retina_worker_t;
struct cortex_worker_t;
struct couch_worker_t;

template <typename Derived> // Curiously Recurring Template Pattern
struct tworker_t : public worker_base_t
{
#ifdef DEBUG_TRACER
	ptracer_t ptracer;
#endif
	template <auto OutputBuf, auto InputBuf>
	void move_to_workers();

	template <typename T, auto BufPtr, auto AddrPtr>
	void put_to_output_buf(T &&ev);

	void process_input_weights();

	void cortex_process_input_events(bool couching_mode);
	void cortex_process();
	void retina_process_input_events();
	void retina_process();

	void create_synapses_events(neuron_t &firing_neuron, neuron_address_t &&addr,
								clock_count_t time_moment);

	void execute() override;

	int64_t calc_delta_time(neuron_t &neuron, clock_count_t afferent_spike_time);
	void calc_threshold_base();

	const char *
	type_name() const override
	{
		if constexpr (std::is_same_v<Derived, retina_worker_t>)
			return "Retina";
		else if constexpr (std::is_same_v<Derived, cortex_worker_t>)
			return "Cortex";
		else if constexpr (std::is_same_v<Derived, couch_worker_t>)
			return "Couch";
		else
			return "Unknown";
	}

	void stdp_weight_update(neuron_t &neuron,
							[[maybe_unused]] neuron_t &post_neuron,
							synapse_t &synapse,
							clock_count_t afferent_spike_time,
							[[maybe_unused]] brain_coord_t synapse_num);

	void store_one_metric(neuron_event_t &e, bool couching_mode, bool fired);

#ifdef DEBUG_TRACER
	tworker_t(head_t *phead, brain_coord_t layer_num, ptracer_t ptracer);
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
