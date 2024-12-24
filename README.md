# SNN framework

## Network
The network includes the visual detectiors layer **Retina** and the cerebral network **Cortex**,
both containing layers of LIF **Neurons**.
The inputs of retinal neurons represent input currents proportional to image changes
The input data of the neurons of the **Cortex** are spikes of the neurons from the previous layer 
Each neural spike is stored in the common **Events buffer** as a set of **Events**, addressed to one of neurons.

The spike goes not immediately each of them corresponding to an appropriate integer *fire_delay* of the synapse (1, 2, ...). The connection del

## Neuron (*neuro_node_t*)
stores:
> - current membrane potential, 
> - current membrane potential threshold, 
> - time of last fire 
> - and a vector of short groups (vectors) of **Connections**, each group corresponding to an appropriate (integer) interneuronal distance

## Spike Event (*event_t*)
The event stores 
> - the generalized address of the neuron which had spiked 
> - planned time of arrival of the spike to the next neuron
> - the group of connections

## Events Buffer (*events_buffer*)
The *events_buffer* is a special kind of a circular buffer. It is composed of short vectors of events,
each of them corresponding to a the events, happened in the time interval between two consequitive "time indexes".
The temporal volume of the time buffer is calculated as 
> - *time_steps* = max_neurons_spiking_contemporally * eye_reaction_time =  1E4 * 1E2  = 1E6 it*ms
There are *time_steps* (1E6) items in the *events_buffer*, each item outstanding by  
 
## To register an **Event** in the event buffer 
> - at the upper level of buffer we go to the  
*upper_level_time_index = planned_time_of_arriving % time_steps* 
> - if we find an event in the vector, having negative time of arrival or time of arrival < *last_processed_time*, then we fill it with:
>> - planned_time_of_arrival
>> - generalized neuron address
>> - connection distance group index
> - if the appropriate event were not found, then we add a new item to the vector
> - we sort vector by time of arrival, include negative onces

## Processing of an **Event** in the event buffer 
> we calculate time index as
*upper_level_time_index = last_processed_time % time_steps*  
> in a vector, residing at that index we search for the first non-negative time being greater than last_processed_time
> we go to the neuron address, find the appropriate connection group and update connected neurons according to the rules:
> #### If projected k+1_membrana_potential < detector_threshold (ajustable)
> - *k+1_membrana_potential = k_membrana_potential(1-leak_alpha) + synapse_weight;*  mV
> - *k+1_detector_threshold = k_detector_threshold * (1 - th_alpha)*
> STDP

> #### If projected k+1_membrana_potential >= detector_threshold (ajustable) 
> - *k+1_membrana_potential = u_rest*
> - *k+1_detector_threshold = k_detector_threshold + delta_threshold*

## Parameters
> - *delta_i_input_min* = 1; 
> - *reasonnable_t_acc_max* = 1000; // ms
> - *initial_detector_threshold* = 0.1 mV
> - *detector_alpha = initial_detector_threshold / reasonnable_t_acc_max / delta_i_input_min*; // 1e-4
> - *visual_detector_threshold* = 1;
> - *u_rest = 10.0;*
> - *leak_alpha = 0.06*
> - *th_alpha = 0.06;*
> - *delta_threshold = 50.0*
> - *time_steps = 1000000*

## Retina
Retina consists of three arrays:  
> - current view - the current image from camera  
>
> - previous view - previous time step image from camera 
> 
> - ajustable threshold-LIF neurons layer  

### Processing retina
At every point of current view the difference between current view and previous view is calculated  
If the result is greater than *visual_detector_threshold*, then the appropriate inferent synapse of retina neuron is updated as described in **Processing of an Event in the event buffer** whith a different way of calculating of the membrane sub-threshold potential: 
> - *input_current = current_val - prev_val*;  can take values 0--32K
> - *k+1_membrana_potential = k_membrana_potential(1-leak_alpha) + detector_alpha * input_current * delta_time*; mV

## Learning
We use STDP learning rules as follows:





