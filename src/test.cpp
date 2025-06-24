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
using namespace std;

constexpr unsigned nof_saccades = 50;

inline std::atomic<bool> stopp = false;
/**
 * @brief Handles CTRL-C signal to softly shutdown the server
 * @param _signal
 */
void fSIGINT_handler([[maybe_unused]] int _signal)
{
    stopp.store(true); // stop the coro loop
}

void main_loop(phead_t phead, ptracer_t ptracer)
{
    // Block SIGINT in all threads (including future threads)
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    pthread_sigmask(SIG_BLOCK, &mask, nullptr);

    // Setup signal handler
    struct sigaction sa;
    sa.sa_handler = fSIGINT_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);

    phead->couching_mode.store(true);
    // phead->couching_mode.store(false);

    auto first_time = true;
    size_t epoque = 0;
    size_t pos_in_test_set = 0;
    // int times = 1;
#ifdef TRACER_DEBUG
    while (ptracer->poll_for_closed_event())
#else
    while (!stopp.load())
#endif
    {
        std::pair<scene_t *, uint8_t> p;

        p = pmnist->next_image();

        if (p.first == nullptr)
            break; // No more images
        // if (!times--)
        //     break;
        // Set appropriate scene
        phead->p_eyes_optics->set_scene(p.first);
#ifdef TRACER_DEBUG
        auto index = phead->p_eyes_optics->scene_index;
        ptracer->set_scene_index(index);
#else
        if (!(phead->p_eyes_optics->scene_index % 100))
            std::cout << phead->p_eyes_optics->scene_index << std::endl;
#endif
        phead->p_eyes_optics->zoom(0, 0, mnist_size, mnist_size);

        // Set appropriate label
        auto pl = phead->layers.back();
        auto p_couch = std::static_pointer_cast<couching_layer_t>(pl);
        p_couch->set_label(p.second);

        // Think about scene
        if (first_time)
        {
            // Start workers
            phead->wake_up(ptracer);
            // Unblock SIGINT only in main thread
            pthread_sigmask(SIG_UNBLOCK, &mask, nullptr);
            first_time = false;
        }

        std::chrono::milliseconds timespan(image_show_delay);
        // for (unsigned i = 0; i < nof_saccades; ++i)
        // {
        std::this_thread::sleep_for(timespan);
        // phead->p_eyes_optics->saccade(1.0);
        // }

        // Change couching mode
        if (phead->couching_mode.load())
            if (epoque == nof_images_in_learning_epoque)
            {
                phead->couching_mode.store(false);
                pos_in_test_set = 0;
            }
            else
            {
                ++epoque;
            }
        else
        {
            if (pos_in_test_set == nof_images_in_test_set)
            {
                phead->couching_mode.store(true);
                // phead->couching_mode.store(false);
                epoque = 0;
                phead->metrics.reset();
            }
            else
            {
                phead->metrics.print_metrics();
                ++pos_in_test_set;
            }
        }
    }

    // Stop workers
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
                                                 {TNN::CORTEX, 14, 14},
                                                 {TNN::CORTEX, 8, 8},
                                                 //  {TNN::CORTEX, mnist_size / 4, mnist_size / 4},
                                                 {TNN::COUCHING, 1, 10}},

                                                {{0, 1, TNN::DOPHAMINE, 14 / 2},
                                                 {1, 2, TNN::DOPHAMINE, 8 / 2},
                                                 //  {2, 3, TNN::DOPHAMINE, 10 / 2}
                                                 {2, 3, TNN::DOPHAMINE, 10 / 2}}));

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
#ifdef DEBUG
        phead->net_timer.print_avg_tick();
#endif
        phead->print_worker_counters();
    }
    std::cout << "Done" << std::endl;
    return 0;
}
