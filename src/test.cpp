#include "brain.h"
#include "input_output.h"
#include "mnist_set.h"
#include "tracer.h"

#include <vector>
#include <array>
#include <sstream>
#include <fstream>
#include <optional>
#include <csignal>
#include <stdio.h>

using namespace TNN;
constexpr unsigned nof_saccades = 50;

inline std::atomic<bool> stop = false;
/**
 * @brief Handles CTRL-C signal to softly shutdown the server
 * @param _signal
 */
void SIGINT_handler([[maybe_unused]] int _signal)
{
    stop.store(true); // stop the coro loop
    // getchar();
}

void main_loop(phead_t phead, ptracer_t ptracer)
{
    struct sigaction handler;
    handler.sa_handler = SIGINT_handler;
    sigemptyset(&handler.sa_mask);
    handler.sa_flags = 0;
    sigaction(SIGINT, &handler, NULL);

    auto first_time = true;
#ifdef TRACER_DEBUG
    while (ptracer->poll_for_closed_event())
#else
    while (!stop.load())
#endif
    {
        std::pair<scene_t *, uint8_t> p;

        // Establish CTRL-C handler

        p = pmnist->next_image();

        if (p.first == nullptr)
            break; // No more images

        // Set appropriate scene
        phead->p_eyes_optics->set_scene(p.first);
#ifdef TRACER_DEBUG
        auto index = phead->p_eyes_optics->scene_index;
        ptracer->set_scene_index(index);
#endif
        phead->p_eyes_optics->zoom(0, 0, mnist_size, mnist_size);

        // Set appropriate label
        auto pl = phead->layers.back();
        auto p_couch = std::static_pointer_cast<mnist_couch_layer_t>(pl);
        p_couch->set_label(p.second);

        // Think about scene
        if (first_time)
        {
            phead->wake_up(ptracer);
            first_time = false;
        }

        std::chrono::milliseconds timespan(200);
        // for (unsigned i = 0; i < nof_saccades; ++i)
        // {
        std::this_thread::sleep_for(timespan);
        // phead->p_eyes_optics->saccade(1.0);
        // }
    }

    // Signal stop processing to workers
    phead->go_to_sleep();
}

/**
 * @brief This is the main function
 * @return int
 */

int main()
{
#ifdef TRACER_DEBUG
    auto ptracer = std::move(std::make_shared<tracer_t>(1720, 1050));
#endif
    {
        auto phead = std::move(std::make_shared<head_t>());
#ifdef TRACER_DEBUG
        ptracer->phead = phead;
#endif
        pmnist = std::make_shared<mnist_set>();
        pmnist->init_set("../MNIST/train-images-idx3-ubyte",
                         "../MNIST/train-labels-idx1-ubyte", true);

        // quick_exit(0);
#ifndef READ_NET_FROM_FILE
        // Create network by description
        create_net(phead.get(), network_descr_t({{TNN::RETINA, mnist_size, mnist_size},
                                                 {TNN::CORTEX, mnist_size, mnist_size},
                                                 {TNN::CORTEX, mnist_size, mnist_size},
                                                 //  {TNN::CORTEX, 14, 14},
                                                 {TNN::COUCHING, 1, 10}},

                                                {
                                                    {0, 1, TNN::DOPHAMINE, 14},
                                                    {1, 2, TNN::DOPHAMINE, 14},
                                                    {2, 3, TNN::DOPHAMINE, 10},
                                                    //  {3, 4, TNN::DOPHAMINE, 1}
                                                }));

#else
        // Read network from file
        phead->read_model_from_file("../networks/net.out", nullptr);
#endif

#ifdef TRACER_DEBUG
        main_loop(phead, ptracer);
        phead->save_model_to_file("../networks/net.out", ptracer);
#else
        main_loop(phead, nullptr);
        phead->save_model_to_file("../networks/net.out", nullptr);
#endif
    }
    std::cout << "Done" << std::endl;
    return 0;
}
