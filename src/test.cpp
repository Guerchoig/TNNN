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

void main_loop(std::shared_ptr<head_t> phead
#ifdef DEBUG_TRACER
               ,
               std::shared_ptr<tracer_t> ptracer
#endif
)

{
        std::mutex cv_mutex; // mutex for head->cv
        std::unique_lock<std::mutex> lock(cv_mutex);

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

        phead->set_couching_mode(true);

        auto first_time = true;
        size_t epoque = 1;
        size_t pos_in_test_set = 1;
        size_t loop_num = 1;

#ifdef DEBUG_TRACER
        while (ptracer->poll_for_closed_event())
#else
        while (!stopp.load())
#endif
        {
                // Pass to next image
                if (phead->next_image(pmnist) == nullptr)
                        break; // No more images

                auto index = phead->get_scene_index();
#ifdef DEBUG_TRACER
                ptracer->set_scene_index(index);
#endif
                if (!(index % nof_images_in_learning_set))
                        std::cout << "Scene " << index << std::endl;

                if (first_time)
                {
                        // Start workers
                        phead->wake_up();

                        // Unblock SIGINT only in main thread
                        pthread_sigmask(SIG_UNBLOCK, &mask, nullptr);
                        first_time = false;
                }

                // Think about scene
                for (auto j = 0; j < iterations_per_image; ++j)
                {
                        // Wait until the end of scene processing
                        phead->cv_processing_image.wait(lock, [phead]()
                                                        { return phead->get_finished_processing_an_image(); });
                        DF;
                        phead->set_finished_processing_an_image(false);
                        loop_num++;
                        phead->inc_net_time();
                }

                if (phead->get_couching_mode())
                {
                        // Rules for couching mode
                        if (epoque == nof_images_in_learning_set)
                        {
                                phead->set_couching_mode(false);
                                pos_in_test_set = 1;
                        }
                        else
                        {
                                ++epoque;
                        }
                }
                else
                {
                        // Rules for test mode
                        if (pos_in_test_set == nof_images_in_test_set)
                        {
                                phead->set_couching_mode(true);
                                epoque = 1;
                                phead->get_metrics().reset();
                        }
                        else
                        {
                                phead->get_metrics().print_metrics();
                                std::cout << " Time moment:" << phead->get_net_time() << " Loop num:" << loop_num << std::endl;
                                ++pos_in_test_set;
                        }
                }
        }

        phead->go_to_sleep();
}

/**
 * @brief This is the main function
 * @return int
 */
int main()
{

#ifdef DEBUG_TRACER
        auto ptracer = std::move(std::make_shared<tracer_t>(1720, 1050));
#endif
        {
                auto phead = std::move(std::make_shared<head_t>());
#ifdef DEBUG_TRACER
                ptracer->phead = phead;
#endif
                pmnist = std::make_shared<mnist_set>();
                pmnist->init_set("../MNIST/train-images-idx3-ubyte",
                                 "../MNIST/train-labels-idx1-ubyte", true);

                brain_coord_t first_neuron_index = 0;
                phead->add_layer(TNN::RETINA, mnist_size, mnist_size, first_neuron_index,
                                 params::usual_nof_pieces_per_layer, phead
#ifdef DEBUG_TRACER
                                 ,
                                 ptracer
#endif
                );
                phead->add_layer(TNN::CORTEX, mnist_size, mnist_size, first_neuron_index, params::usual_nof_pieces_per_layer, phead
#ifdef DEBUG_TRACER
                                 ,
                                 ptracer
#endif
                );

                phead->add_layer(TNN::CORTEX, mnist_size , mnist_size , first_neuron_index, params::usual_nof_pieces_per_layer, phead
#ifdef DEBUG_TRACER
                                 ,
                                 ptracer
#endif
                );

                phead->add_layer(TNN::COUCHING, 1, params::nof_cathegories, first_neuron_index, params::nof_pieces_in_last_layer, phead
#ifdef DEBUG_TRACER
                                 ,
                                 ptracer
#endif
                );

                phead->add_connections(0, 1, TNN::DOPHAMINE, TNN::FULLY_CONNECTED);
                phead->add_connections(1, 2, TNN::DOPHAMINE, TNN::FULLY_CONNECTED);
                phead->add_connections(2, 3, TNN::DOPHAMINE, TNN::FULLY_CONNECTED);

                // phead->print_nof_synapses_per_neuron();
                // exit(0);

#ifdef DEBUG_TRACER
                main_loop(phead, ptracer);
                // phead->save_model_to_file("../networks/net.out", ptracer);
#else
                main_loop(phead);
                // phead->save_model_to_file("../networks/net.out");
#endif
        }

        // logger.dump_to_file(true, false);

        std::cout << "Done" << std::endl;
        return 0;
}
