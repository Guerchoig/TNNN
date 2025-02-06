#pragma once
#include "common.h"
#include "clock_circular_buffer.h"
#include "tracer.h"
#include "tread_safe_queue.h"
#include <boost/circular_buffer.hpp>
#include <cstring>
#include <vector>
#include <atomic>
#include <memory>
#include <iostream>
#include <fstream>
#include <type_traits>

using input_val_t = int;

// Nof connections per neuron = exp(-alpha * (distance - 1)) * (sqr(distance) * 3 + distance * 6 + 4 ) * 2
constexpr float alpha = 0.79f;
constexpr layer_dim_t max_distance = 6; // 1 < distance <= max_distance

// input_curr =cur_scene_val - old_scene_val;  can take value 1--1E6 both
// delta_input_potential = detector_alpha * input_curr * delta_time
// initial_neuron_threshold = detector_alpha * delta_i_input_min * reasonnable_t_acc_max
// Meta- parameters
constexpr input_val_t delta_i_input_min = 1;

constexpr clock_count_t reasonnable_t_acc_max = 1000; // ms
constexpr potential_t u_mem_max = 1.0;
constexpr potential_t delta_u_mem = 0.05;
constexpr potential_t initial_neuron_threshold = 0.7;
constexpr potential_t delta_threshold = 0.005;
constexpr potential_t u_rest = 0.00;
constexpr potential_t detector_alpha = initial_neuron_threshold / reasonnable_t_acc_max / delta_i_input_min; // 1e-4
constexpr input_val_t visual_detector_threshold = 2;

// u_i+1 = u_i * (1-leak_alpha) + detector_alpha * delta_i_input * delta_time; | u_i+1  < u_threshold
// u_threshold_i+1 =  u_threshold_i * (1-th_alpha);                            |
//
// u_i+1 = u_rest;                                      | u_i+1  >= u_threshold
// u_threshold_i+1 =  u_threshold_i + delta_threshold;  |
// u_i can take value 0.0--100.0
// initial_threshold 1--10

constexpr potential_t leak_alpha = 0.3;
constexpr potential_t th_alpha = 0.6;

// Weights
constexpr potential_t delta_weight = 0.1;
constexpr potential_t weigth_alpha = 0.1;

// Actuator
constexpr clock_count_t actuator_tau = 1000; // ms

// View field
constexpr unsigned view_field_def_width = 28;
constexpr unsigned view_field_def_heigth = 28;

// Synapses-----------------------------------------------

struct synapse_t
{
	// clock_count_t last_fired;
	weight_t weight;
	TNN::ferment_t ferment;
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

struct base_worker_t;

/**
 * @brief This class represents a neuron in a neural network, with properties
 * and methods to manage its state and behavior
 */
struct neuro_node_t
{
	potential_t u_mem;
	potential_t threshold;
	clock_count_t last_fired;
	atomic_mutex busy;
	std::vector<synapse_t> synapses;

	virtual potential_t input([[maybe_unused]] event_t &&e) { return 0.0; }

	virtual void update_params([[maybe_unused]] event_t &&e,
							   [[maybe_unused]] clock_count_t time_moment) {}
	virtual void output(base_worker_t &worker, [[maybe_unused]] event_t &&ev);

	neuro_node_t &operator=(const neuro_node_t &other)
	{
		u_mem = other.u_mem;
		threshold = other.threshold;
		last_fired = other.last_fired;

		return *this;
	}

	neuro_node_t() : busy() {}

	neuro_node_t(potential_t u_mem,
				 potential_t threshold,
				 clock_count_t last_fired) : u_mem(u_mem),
											 threshold(threshold),
											 last_fired(last_fired),
											 busy() {}

	neuro_node_t(const neuro_node_t &&other)
		: u_mem(other.u_mem), threshold(other.threshold), last_fired(other.last_fired) {}

	neuro_node_t(const neuro_node_t &other)
		: u_mem(other.u_mem), threshold(other.threshold), last_fired(other.last_fired) {}
};

struct cortex_node_t : neuro_node_t
{
	potential_t input(event_t &&e) override final;
	void update_params(event_t &&e,
					   clock_count_t time_moment) override final;
	using neuro_node_t::neuro_node_t;
};

struct retina_node_t : neuro_node_t
{
	potential_t input(event_t &&e) override final;
	void update_params([[maybe_unused]] event_t &&e,
					   [[maybe_unused]] clock_count_t time_moment) override final {}
	using neuro_node_t::neuro_node_t;
};

struct couch_node_t : neuro_node_t
{
	potential_t input(event_t &&e) override final;
	void update_params([[maybe_unused]] event_t &&e,
					   [[maybe_unused]] clock_count_t time_moment) override final {}
	using neuro_node_t::neuro_node_t;
};

struct actuator_node_t : neuro_node_t
{
	clock_circular_buffer clocks;
	// using neuro_node_t::neuro_node_t;
	potential_t input(event_t &&e) override final;
	void output([[maybe_unused]] base_worker_t &worker, [[maybe_unused]] event_t &&ev) override final;
	unsigned value();
	actuator_node_t() : neuro_node_t::neuro_node_t() {}

	actuator_node_t(potential_t u_mem,
					potential_t threshold,
					clock_count_t last_fired) : neuro_node_t::neuro_node_t(u_mem,
																		   threshold,
																		   last_fired)
	{
		;
	}

	actuator_node_t(const actuator_node_t &&other)
	{
		u_mem = other.u_mem;
		threshold = other.threshold;
		last_fired = other.last_fired;
	}

	actuator_node_t(const neuro_node_t &other)
	{
		u_mem = other.u_mem;
		threshold = other.threshold;
		last_fired = other.last_fired;
	}
};

// Optics------------------------------------------------

struct eyes_optics_t
{

	scene_t *pscene;
	layer_dim_t left;
	layer_dim_t top;
	layer_dim_t right;
	layer_dim_t bottom;
	scene_t prev_view;

	void set_focus(int _left, int _top, layer_dim_t _width, layer_dim_t _heigth);

	void look_at(scene_t *_pscene, layer_dim_t _left = 0, layer_dim_t _top = 0);

	void shift(int dx, int dy, float dist);

	void saccade(float dist);

	eyes_optics_t(layer_dim_t width = view_field_def_width,
				  layer_dim_t heigth = view_field_def_heigth) : left{0}, top{0},
																right(width - 1),
																bottom(heigth - 1)
	{
		for (layer_dim_t i = 0; i < width; i++)
			for (layer_dim_t j = 0; j < heigth; j++)
				prev_view[i][j] = 0;
	}
};

// Layers------------------------------------------------

template <typename T>
using vector_2D_t = std::vector<std::vector<T>>;

struct layer_t
{
	using node_t = neuro_node_t;
	TNN::layer_type type;
	vector_2D_t<size_t> neurons;
	atomic_mutex busy;
	virtual node_t &node_ref([[maybe_unused]] layer_dim_t row, [[maybe_unused]] layer_dim_t col) = 0;
	layer_t() = default;
	layer_t(TNN::layer_type _type) : type(_type) {}
	virtual ~layer_t() = default;
};

using layers_t = std::vector<std::shared_ptr<layer_t>>;

template <typename T>
concept Is_layer = std::is_base_of_v<layer_t, T>;


template <Is_layer T>
void create_neurons(T *layer, layer_dim_t rows, layer_dim_t cols);

struct cortex_layer_t : layer_t
{
	using node_t = cortex_node_t;
	static std::vector<cortex_node_t> neurons_storage;
	// static vector_2D_t<synapse_t> synapses_storage;
	node_t &node_ref(layer_dim_t row, layer_dim_t col) override final
	{
		return neurons_storage.at(neurons.at(row).at(col));
	};
	using layer_t::layer_t;
	cortex_layer_t() = default;
	cortex_layer_t(TNN::layer_type _type) : layer_t(_type) {}
	cortex_layer_t(layer_dim_t rows, layer_dim_t cols, TNN::layer_type _type);
};

inline std::vector<cortex_node_t> cortex_layer_t ::neurons_storage;




struct retina_layer_t : layer_t
{
	struct worker : base_worker_t
	{
		worker : base_worker_t(0) {}
		void worker(const retina_layer_t &layer) override final;
	};

	using node_t = retina_node_t;
	std::shared_ptr<eyes_optics_t> p_eyes_optics;
	static std::vector<retina_node_t> neurons_storage;
	static std::atomic<bool> moving_look;
	node_t &node_ref(layer_dim_t row, layer_dim_t col) override final
	{
		return neurons_storage.at(neurons.at(row).at(col));
	};

	retina_layer_t();
	retina_layer_t(TNN::layer_type _type);
	retina_layer_t(retina_layer_t &&other);
	retina_layer_t(layer_dim_t rows, layer_dim_t cols, TNN::layer_type _type);
};
inline std::atomic<bool> retina_layer_t::moving_look;
inline std::vector<retina_node_t> retina_layer_t ::neurons_storage;

struct mnist_couch_layer_t : layer_t
{
	using node_t = couch_node_t;
	static std::vector<couch_node_t> neurons_storage;
	static unsigned char label;
	node_t &node_ref(layer_dim_t row, layer_dim_t col) override final
	{
		return neurons_storage.at(neurons.at(row).at(col));
	};
	static void worker();
	void set_label(layer_dim_t i_label = 0);
	mnist_couch_layer_t() = default;
	mnist_couch_layer_t(TNN::layer_type _type) : layer_t(_type) {}
	mnist_couch_layer_t(layer_dim_t rows, layer_dim_t cols, TNN::layer_type _type);
};

inline std::vector<couch_node_t> mnist_couch_layer_t ::neurons_storage;
inline unsigned char mnist_couch_layer_t ::label;

struct actuator_layer_t : layer_t
{
	using node_t = actuator_node_t;
	static std::vector<actuator_node_t> neurons_storage;

	// static vector_2D_t<synapse_t> synapses_storage;
	node_t &node_ref(layer_dim_t row, layer_dim_t col) override final
	{
		return neurons_storage.at(neurons.at(row).at(col));
	}

	actuator_layer_t() = default;
	actuator_layer_t(TNN::layer_type _type) : layer_t(_type) {}
	actuator_layer_t(layer_dim_t rows, layer_dim_t cols, TNN::layer_type _type);
};

inline std::vector<actuator_node_t> actuator_layer_t::neurons_storage;

// Events -------------------------------------------------

struct events_set
{
	boost::circular_buffer<event_t> buf;
	atomic_mutex wmutex;
	atomic_mutex rmutex;
	events_set(const size_t size) : buf{size}, wmutex(), rmutex() {}
	events_set(events_set &&other) : buf{other.buf}, rmutex() {}
};

// Workers ------------------------------------------------
struct base_worker_t
{
	unsigned id;
	ThreadSafeQueue<event_t> input_events;
	std::thread thread;
	virtual void worker([[maybe_unused]] const retina_layer_t &layer) = 0;
	base_worker_t(unsigned id) : id(id)
	{
		thread = std::thread(&base_worker_t::worker, this, id);
	}
	~base_worker_t()
	{
		thread.join();
	}
};



struct internal_layer_worker_t : base_worker_t
{
	internal_layer_worker_t(unsigned id) : base_worker_t(id) {}
	void worker([[maybe_unused]] const retina_layer_t &layer) override final;
};

// Head ------------------------------------------------


struct head_t
{
	layers_t layers;
	retina_layer_t *pretina;
	std::shared_ptr<eyes_optics_t> p_eyes_optics;
	net_timer_t net_timer;
	const clock_count_t stdp_delay = 5L;

	std::vector<std::thread> threads;
	std::atomic<bool> finish;

	std::atomic<unsigned> event_index = 0;

	conn_descr_coll_t connections;

	void look_at(scene_t *pscene)
	{
		if (layers[0]->type == TNN::RETINA)
			std::static_pointer_cast<retina_layer_t>(layers[0])->moving_look = true;
		p_eyes_optics->look_at(pscene, 0, 0);
		if (layers[0]->type == TNN::RETINA)
			std::static_pointer_cast<retina_layer_t>(layers[0])->moving_look = false;
	}

	void set_focus(unsigned width, unsigned heigth, int left, int top)
	{
		p_eyes_optics->set_focus(width, heigth, left, top);
	}

	void saccade(float dist)
	{
		p_eyes_optics->saccade(dist);
	}

	void wake_up(scene_t *pscene, unsigned width, unsigned heigth);
	void do_sleep();
	void print_output(layer_dim_t layer_num);
	head_t();
};

inline std::shared_ptr<head_t> phead;

// Openers --------------------------------------------------

void update_potential(event_t &&e);

void print_image(scene_t *pscene);

void internal_layer_worker([[maybe_unused]] unsigned id);
void input_layers_worker();
struct print_weights_t
{
	atomic_mutex wmutex;
	std::fstream weights_file;
	// ("../networks/weigths.out", std::ios::out | std::ios::trunc);
	void operator()(layer_dim_t layer_num, layer_dim_t row_num);
	print_weights_t() : wmutex()
	{
		weights_file.open("../networks/weigths.out", std::ios::out | std::ios::trunc);
		weights_file.precision(2);
	}
	~print_weights_t()
	{
		weights_file.close();
	}
};
inline print_weights_t print_weights;

// Global Tracer --------------------------------------------------
inline std::shared_ptr<tracer_t> ptracer;
