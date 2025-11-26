#include "brain.h"
#include "input_output.h"
#include "mnist_set.h"
#include "tracer.h"
#include "tests.h"

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

inline std::ofstream out;

inline std::atomic<bool> stopp = false;
/**
 * @brief Handles CTRL-C signal to softly shutdown the server
 * @param _signal
 */
void fSIGINT_handler([[maybe_unused]] int _signal)
{
        stopp.store(true); // stop the coro loop
}

std::streambuf *redirect_output_to_file(std::string file_name)
{
        out.open(file_name);
        if (!out)
        {
                std::cerr << "Can't open file for std::cout redirection\n";
                return nullptr;
        }
        std::streambuf *old_cout = std::cout.rdbuf(out.rdbuf()); // перенаправляем std::cout
        return old_cout;
}

void redirect_output_to_console(std::streambuf *old_cout)
{
        std::cout.rdbuf(old_cout); // возвращаем std::cout обратно в консоль
}

void main_loop(std::shared_ptr<head_t> phead TRACE_PARAM)

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

        WHILE_TRACER(ptracer)
        {
                // Pass to next image
                if (phead->next_image(pmnist) == nullptr)
                        break; // No more images

                TRACE_STMT(auto index = phead->get_scene_index(); ptracer->set_scene_index(index););

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

                        phead->set_finished_processing_an_image(false);
                        loop_num++;
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
                                // std::cout << " Time moment:" << phead->get_net_time() << " Loop num:" << loop_num << std::endl;
                                ++pos_in_test_set;
                        }
                }
                DEBUG_STMT(potential_t sumw = 0.0; potential_t maxw = 0.0; potential_t minw = 1000.0; //
                           for (auto synapse : phead->get_synapses()) {
                                        auto val = synapse.get_weight();
                                        sumw += val;
                                        maxw = std::max(maxw, val);
                                        minw = std::min(minw, val); }                             //
                               * plogger
                           << "Weights \t" << sumw / phead->get_synapses().size() << '\t' << minw << '\t' << maxw << std::endl;
                           //    sumw = 0.0; maxw = 0.0; minw = 1000.0;                                                                  //
                           //    for (auto pneuron = phead->get_neurons().begin(); pneuron != phead->get_neurons().end(); ++pneuron) {
                           //         auto val = pneuron->get_threshold_adaptation();
                           //            sumw += val;
                           //            maxw = std::max(maxw, val);
                           //            minw = std::min(minw, val); } //
                           //        * plogger
                           //    << "Th_adapt \t" << "Val: \t" << phead->get_label() << "\t" << "Stat: \t" << sumw / phead->get_neurons().size() << '\t' << minw << '\t' << maxw << std::endl
                );
        }

        phead->go_to_sleep();
}

/**
 * @brief This is the main function
 * @return int
 */
int main()
{

        text_logger logger("../log.txt");
        logger << std::endl
               << "BEGIN" << std::endl;
        plogger = &logger;

        TRACE_DECL(auto ptracer = std::move(std::make_shared<tracer_t>(1720, 1050, params));)
        {
                TRACE_STMT(ptracer->init_param_fields(););
                auto phead = std::move(std::make_shared<head_t>());
                TRACE_STMT(ptracer->phead = phead;);
                pmnist = std::make_shared<mnist_set>();
                pmnist->init_set("../MNIST/train-images-idx3-ubyte",
                                 "../MNIST/train-labels-idx1-ubyte", true);

                brain_coord_t first_neuron_index = 0;
                phead->add_layer(TNN::RETINA, mnist_size, mnist_size, first_neuron_index, 4 * 4, phead TRACE_ARG);
                phead->add_layer(TNN::REFERENCE, 1, params.nof_cathegories.load(), first_neuron_index, params.nof_pieces_in_last_layer.load(), phead TRACE_ARG);
                phead->add_layer(TNN::CORTEX, mnist_size, mnist_size, first_neuron_index, 4 * 4, phead TRACE_ARG);
                phead->add_layer(TNN::CORTEX, mnist_size, mnist_size, first_neuron_index, 4 * 4, phead TRACE_ARG);
                phead->add_layer(TNN::OUTPUT, 1, params.nof_cathegories.load(), first_neuron_index, params.nof_pieces_in_last_layer.load(), phead TRACE_ARG);

                phead->add_connections(0, 2, TNN::GLUTAMATE, TNN::FULLY_CONNECTED);
                phead->add_connections(1, 4, TNN::GLUTAMATE, TNN::ONE_TO_ONE);
                phead->add_connections(2, 3, TNN::GLUTAMATE, TNN::FULLY_CONNECTED);
                phead->add_connections(3, 4, TNN::GLUTAMATE, TNN::FULLY_CONNECTED);


                // for (auto p = phead->get_neurons().begin(); p != phead->get_neurons().end(); ++p)
                //         logger<< p->get_nof_inputs() << std::endl;
                // logger.dump_to_file(); // Writes buffered lines to file
                // exit(0);

                main_loop(phead TRACE_ARG);
        }

        logger.dump_to_file();
        // redirect_output_to_console(old_buf);
        std::cout << "Done" << std::endl;
        return 0;
}
