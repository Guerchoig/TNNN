#pragma once

#include "common.h"
#include "tracer.h"
// #include "atomic_queue.h"
#include "eyes_optics.h"
#include "counters.h"
// #include "vector2d.h"
#include "metrics.h"
#include "mnist_set.h"
#include <bitset>
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
#include <condition_variable>

namespace params
{
	constexpr brain_coord_t nof_cathegories = 10;
	constexpr brain_coord_t usual_nof_pieces_per_layer = 16;
	constexpr brain_coord_t nof_pieces_in_last_layer = 1;

	// Synapse's params-----------------------------------------------
	// constexpr double spike_velocity = 1.0;

	// Cortex's neuron params-----------------------------------------------
	constexpr potential_t u_rest = 0.0;
	constexpr potential_t initial_neuron_threshold = 1.0;

	constexpr potential_t normal_threshold_base = 0.0;

	constexpr potential_t cortex_leak_tau = 10.0; //  tics
	constexpr potential_t cortex_leak_freq = 1 / cortex_leak_tau;

	// Visual detector's  params------------------------------------------------
	// constexpr input_val_t visual_detector_threshold = 1;
	constexpr scene_signal_t max_scene_amplitude = 255;
	constexpr potential_t detector_alpha = 0.2 * initial_neuron_threshold / max_scene_amplitude;

	// Weights update  params------------------------------------------------
	constexpr potential_t dw_max = 0.1;
	constexpr clock_count_t dw_plus_time = 2;
	constexpr clock_count_t dw_minus_time = 2;
	constexpr potential_t dw_alpha_plus = 1.0 / dw_plus_time;
	constexpr potential_t dw_alpha_minus = 1.0 / dw_minus_time;
}

using input_val_t = int;
constexpr size_t spiking_history_len = 8;
using history_spikes_t = std::bitset<spiking_history_len>;

constexpr clock_count_t empty_time = std::numeric_limits<clock_count_t>::max();

// Synapses-----------------------------------------------

class synapse_t
{
private:
	weight_t weight;
	TNN::ferment_t ferment;
	brain_coord_t src_index;
	brain_coord_t trg_index;

public:
	void set_weight(weight_t w) { weight = w; }
	weight_t get_weight() const { return weight; }
	void set_ferment(TNN::ferment_t f) { ferment = f; }
	TNN::ferment_t get_ferment() const { return ferment; }
	void set_src_index(brain_coord_t src_index) { this->src_index = src_index; }
	brain_coord_t get_src_index() const { return src_index; }
	void set_trg_index(brain_coord_t trg_index) { this->trg_index = trg_index; }
	brain_coord_t get_trg_index() const { return trg_index; }

	synapse_t() = default;
	synapse_t(weight_t weight,
			  TNN::ferment_t ferment,
			  brain_coord_t src_index,
			  brain_coord_t trg_index) : weight(weight),
										 ferment(ferment),
										 src_index(src_index),
										 trg_index(trg_index)
	{
	}

	synapse_t(const synapse_t &other) : weight(other.weight),
										ferment(other.ferment),
										src_index(other.src_index),
										trg_index(other.trg_index)
	{
	}
};

// Nodes-------------------------------------------------

/**
 * @brief This class represents a neuron in a neural network, with properties
 * and methods to manage its state and behavior
 */
class neuron_t
{
private:
	std::atomic<potential_t> u_mem = params::u_rest;
	potential_t threshold = params::initial_neuron_threshold;
	clock_count_t last_fired = 0LL;

	// Bitmap of current and previous spikes for STDP
	history_spikes_t spiking_history{0x01};

	// Synapses references
	std::vector<brain_coord_t> input_synapse_indexes;
	std::vector<brain_coord_t> output_synapse_indexes;

	// Spike trace for STDP
	potential_t trace = 0.0;

	std::atomic<bool> must_fire = false;

public:
	void set_u_mem(potential_t u_mem) { this->u_mem.store(u_mem); }
	potential_t get_u_mem() const { return u_mem; }
	void fetch_add_u_mem(potential_t u_mem) { this->u_mem.fetch_add(u_mem); }
	void set_threshold(potential_t threshold) { this->threshold = threshold; }
	potential_t get_threshold() const { return threshold; }
	void set_last_fired(clock_count_t last_fired) { this->last_fired = last_fired; }
	clock_count_t get_last_fired() const { return last_fired; }
	void set_spiking_history(history_spikes_t spiking_history) { this->spiking_history = spiking_history; }
	history_spikes_t get_spiking_history() const { return spiking_history; }
	std::vector<brain_coord_t> &get_input_synapse_indexes() { return input_synapse_indexes; }
	std::vector<brain_coord_t> &get_output_synapse_indexes() { return output_synapse_indexes; }
	void set_trace(potential_t trace) { this->trace = trace; }
	potential_t get_trace() const { return trace; }
	void set_must_fire(bool must_fire) { this->must_fire = must_fire; }
	bool get_must_fire() const { return must_fire; }

	// Default operations
	neuron_t() : u_mem(0.0), threshold(params::initial_neuron_threshold) {}
	neuron_t &operator=(const neuron_t &other) = default;

	// Custom constructor
	neuron_t(potential_t u_mem,
			 potential_t threshold,
			 clock_count_t last_fired) : u_mem(u_mem),
										 threshold(threshold),
										 last_fired(last_fired) {}

	// Explicit move operations
	neuron_t(neuron_t &&other) noexcept : threshold(std::exchange(other.threshold, params::initial_neuron_threshold)),
										  last_fired(std::exchange(other.last_fired, 0LL)),
										  spiking_history(std::exchange(other.spiking_history, history_spikes_t{0x01})),
										  trace(std::exchange(other.trace, 0))
	{
		u_mem.store(other.u_mem.exchange(0.0));
		other.must_fire.store(must_fire.exchange((other.must_fire.load())));
		input_synapse_indexes = std::move(other.input_synapse_indexes);
		output_synapse_indexes = std::move(other.output_synapse_indexes);
	}

	// Move assignment operator
	neuron_t &operator=(neuron_t &&other) noexcept
	{
		u_mem.exchange(other.u_mem);
		threshold = std::exchange(other.threshold, params::initial_neuron_threshold);
		last_fired = std::exchange(other.last_fired, 0LL);
		spiking_history = std::exchange(other.spiking_history, history_spikes_t{0x01});
		trace = std::exchange(other.trace, 0);
		other.must_fire.store(must_fire.exchange((other.must_fire.load())));
		input_synapse_indexes = std::move(other.input_synapse_indexes);
		output_synapse_indexes = std::move(other.output_synapse_indexes);
		return *this;
	}
};

enum state_t : int
{
	ready_to_be_processed,
	in_process
};

class head_t;

class piece_t
{
private:
	TNN::layer_type type;
	brain_coord_t layer_num;
	brain_coord_t first_index;
	brain_coord_t size;
	bool last_piece_in_layer = false;

	std::atomic<state_t> state = state_t::ready_to_be_processed;
	std::atomic<potential_t> threshold_base = params::normal_threshold_base;
	std::atomic<bool> has_a_must_fire = false;
	clock_count_t time_moment = 0LL;

	std::shared_ptr<head_t> phead;
	std::vector<neuron_t> &neurons;
#ifdef DEBUG_TRACER
	std::shared_ptr<tracer_t> ptracer;
#endif

public:
	void set_type(TNN::layer_type type) { this->type = type; }
	TNN::layer_type get_type() const { return type; }
	void set_layer_num(brain_coord_t layer_num) { this->layer_num = layer_num; }
	brain_coord_t get_layer_num() const { return layer_num; }
	void set_first_index(brain_coord_t first_index) { this->first_index = first_index; }
	brain_coord_t get_first_index() const { return first_index; }
	void set_size(brain_coord_t size) { this->size = size; }
	brain_coord_t get_size() const { return size; }

	void set_state(state_t s) { state.store(s); }
	state_t get_state() { return state.load(); }
	state_t exchange_state(state_t s) { return state.exchange(s); }

	void set_threshold_base(potential_t threshold_base) { this->threshold_base.store(threshold_base); }
	potential_t get_threshold_base() { return threshold_base.load(); }
	void set_has_a_must_fire(bool has_a_must_fire) { this->has_a_must_fire = has_a_must_fire; }
	bool get_has_a_must_fire() { return has_a_must_fire; }

	void set_time_moment(clock_count_t time_moment) { this->time_moment = time_moment; }
	clock_count_t get_time_moment() { return time_moment; }

	void update_postsynaptic_neuron(neuron_t &cur_neuron, head_t &phead, int64_t delta_time);
	std::shared_ptr<head_t> get_phead() { return phead; }
	std::vector<neuron_t> &get_neurons() { return neurons; }
#ifdef DEBUG_TRACER
	std::shared_ptr<tracer_t> get_ptracer() { return ptracer; }
#endif
	bool is_last_piece_in_layer() const { return last_piece_in_layer; }

	piece_t(TNN::layer_type,
			brain_coord_t layer_num,
			brain_coord_t first_index,
			brain_coord_t size,
			std::shared_ptr<head_t> phead,
			std::vector<neuron_t> &neurons
#ifdef DEBUG_TRACER
			,
			std::shared_ptr<tracer_t> ptracer
#endif
	);

	piece_t(piece_t &&other) noexcept : type(other.type),
										layer_num(other.layer_num),
										first_index(other.first_index),
										size(other.size),
										last_piece_in_layer(other.last_piece_in_layer),
										time_moment(other.time_moment),
										phead(other.phead),
										neurons(other.neurons)
#ifdef DEBUG_TRACER
										,
										ptracer(other.ptracer)
#endif
	{
		other.state.store(state.exchange((other.state.load())));
		other.threshold_base.store(threshold_base.exchange((other.threshold_base.load())));
	}

	piece_t &operator=(piece_t &&other) noexcept;
};

// Base layer -----------------------------------------------

clock_count_t calc_delta_time(neuron_t &neuron, clock_count_t afferent_spike_time);

struct head_t; // forward declaration

class layer_t
{
private:
	TNN::layer_type ltype;
	brain_coord_t rows; // nof neurons rows
	brain_coord_t cols; // nof neurons cols
	brain_coord_t first_neurons_index;

public:
	brain_coord_t
	get_rows() const { return rows; }
	brain_coord_t get_cols() const { return cols; }
	brain_coord_t get_first_neurons_index() const { return first_neurons_index; }
	virtual void process_neurons(piece_t &piece) = 0;

	layer_t(TNN::layer_type ltype, brain_coord_t rows,
			brain_coord_t cols,
			brain_coord_t first_neurons_index) : ltype(ltype), rows(rows),
												 cols(cols),
												 first_neurons_index(first_neurons_index) {}
};

// Free functions------------------------------------------------
potential_t retina_leak_and_input(neuron_t &neuron, scene_signal_t signal,
								  //   std::pair<scene_signal_t, clock_count_t> &timed_memory_signal,
								  clock_count_t delta_time);
potential_t cortex_leak_and_input(neuron_t &neuron,
								  potential_t &weight,
								  clock_count_t delta_time);

// void stdp_weight_update(neuron_t &neuron, neuron_t &post_neuron, synapse_t &synapse, clock_count_t afferent_spike_time);
potential_t fire_neuron(neuron_t &cur_neuron, piece_t &piece, clock_count_t delta_time);

void process_input_weights(neuron_t &cur_neuron, head_t &phead, clock_count_t post_synaptic_spike_time);
void update_postsynaptic_neurons(neuron_t &cur_neuron, head_t &phead,
								 potential_t threshold_base,
								 clock_count_t delta_time);
void stdp_weight_update(neuron_t &cur_neuron,
						neuron_t &post_neuron,
						synapse_t &synapse,
						clock_count_t post_synaptic_spike_time);

// Derived layers------------------------------------------------

class retina_layer_t : public layer_t
{
public:
	void process_neurons(piece_t &piece) override final;
	using layer_t::layer_t;
};

class cortex_layer_t : public layer_t
{
public:
	void process_neurons(piece_t &piece) override;
	using layer_t::layer_t;
};

/**
 * @brief In this layer we store the label of the current image.
 * the synapses just mirror the input synapses to fasciliate weights update
 * in ajacent layers
 */
class couching_layer_t : public cortex_layer_t
{
public:
	void process_neurons(piece_t &piece) override final;
	using cortex_layer_t::cortex_layer_t;
};

// Worker's class ------------------------------------------------

class worker_t
{
private:
	std::thread worker_thread;
	std::shared_ptr<head_t> phead;

public:
	std::thread &get_worker_thread() { return worker_thread; }
	void execute();
	void calc_threshold_base();
	worker_t(std::shared_ptr<head_t> phead);
};

template <typename T>
	requires std::integral<T>
std::pair<T, T> twod_from1(T k, T M)
{
	T i = k / M; // Row index (integer division)
	T j = k % M; // Column index (remainder)
	return {i, j};
}

template <typename T>
	requires std::integral<T>
T oned_from2(T i, T j, T M)
{
	return i * M + j;
}

// head_t ------------------------------------------------
class head_t
{
private:
	conn_descr_coll_t connections_descr;
	std::vector<std::unique_ptr<layer_t>> layers_descr;
	std::map<brain_coord_t, brain_coord_t> layer_num_by_1st_neuron_index;
	std::vector<piece_t> pieces;
	std::vector<synapse_t> synapses;
	std::vector<neuron_t> neurons;

	std::shared_ptr<eyes_optics_t> p_eyes_optics;
	net_timer_t net_timer;

	std::vector<std::shared_ptr<worker_t>> workers;

	std::atomic<unsigned char> label;
	std::atomic<int> nof_active_workers;
	std::atomic<bool> finish;
	std::atomic<bool> couching_mode = false;
	metrics_t metrics;
	std::atomic<brain_coord_t> last_piece_in_process = -1;
	atomic_mutex mtx_pieces;
	std::atomic<bool> finished_processing_an_image = false;
	std::mutex moving_gaze;
	uint64_t scene_index = 0;

public:
	std::condition_variable cv_processing_image;
	conn_descr_coll_t &get_connections_descr() { return connections_descr; }
	std::vector<std::unique_ptr<layer_t>> &get_layers_descr() { return layers_descr; }
	std::vector<piece_t> &get_pieces() { return pieces; }
	std::vector<synapse_t> &get_synapses() { return synapses; }
	std::vector<neuron_t> &get_neurons() { return neurons; }
	std::shared_ptr<eyes_optics_t> get_p_eyes_optics() { return p_eyes_optics; }
	net_timer_t &get_net_timer() { return net_timer; }
	void set_finish() { finish.store(true); }
	bool get_finish() { return finish.load(); }
	void fetch_add_nof_active_workers(int n) { nof_active_workers.fetch_add(n); }
	void set_label(int l) { label.store(l); }
	uint32_t get_label() const { return label.load(); }
	void set_couching_mode(bool couching_mode) { this->couching_mode.store(couching_mode); }
	bool get_couching_mode() const { return couching_mode.load(); }
	metrics_t &get_metrics() { return metrics; }
	void store_one_metric(bool fired, brain_coord_t col);
	bool get_finished_processing_an_image() { return finished_processing_an_image.load(); }
	void set_finished_processing_an_image(bool val) { finished_processing_an_image.store(val); }

	void set_scene_index(uint64_t scene_index) { this->scene_index = scene_index; }
	uint64_t get_scene_index() const { return scene_index; }

	brain_coord_t get_first_piece_index_in_layer(brain_coord_t layer) const;
	clock_count_t get_net_time() { return net_timer.just_time(); }
	clock_count_t inc_net_time() { return net_timer.time_and_inc(); }

	// public:
	void add_connections(brain_coord_t src_layer, brain_coord_t trg_layer,
						 TNN::ferment_t ferment, TNN::connection_type connection_type);

	void add_layer(TNN::layer_type ltype, brain_coord_t rows, brain_coord_t cols,
				   brain_coord_t &first_neuron_index, brain_coord_t nof_pieces,
				   std::shared_ptr<head_t> phead
#ifdef DEBUG_TRACER
				   ,
				   std::shared_ptr<tracer_t> ptracer
#endif
	);

	abs_address_t abs_address(brain_coord_t addr);
	brain_coord_t neuron_index(abs_address_t &&addr);
	neuron_t &neuron_ref(const brain_coord_t &addr)
	{
		return neurons.at(addr);
	}
	piece_t *get_a_piece_to_process();

	void wake_up();
	void go_to_sleep();

	void print_counters(uint iteration) const;

#ifdef DEBUG_TRACER
	void save_model_to_file(std::string file_name, [[maybe_unused]] std::shared_ptr<tracer_t> ptracer);
	void read_model_from_file(std::string file_name, [[maybe_unused]] std::shared_ptr<tracer_t> ptracer);
#else
	void save_model_to_file(std::string file_name);
	void read_model_from_file(std::string file_name);
#endif

	scene_t *get_locked_scene()
	{
		moving_gaze.lock();
		auto scene = p_eyes_optics->get_scene();
		return scene;
	}

	void unlock_scene() { moving_gaze.unlock(); }

	scene_t *next_image(std::shared_ptr<mnist_set> pmnist);

	std::uint8_t get_signal(brain_coord_t i)
	{
		return p_eyes_optics->get_signal(i);
	}
	void print_nof_synapses_per_neuron();
	head_t();
};

potential_t exp_term(clock_count_t delta_time, potential_t alpha);
