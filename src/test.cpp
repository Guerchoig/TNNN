#include "brain.h"
#include "input_output.h"
#include "mnist_set.h"
#include "tracer.h"

#include <vector>
#include <array>
#include <sstream>
#include <fstream>
using namespace TNN;

void print_couch()
{
    auto couch_layer = std::dynamic_pointer_cast<mnist_couch_layer_t>(phead->layers[4]);
    std::cout << "Label: " << static_cast<int>(couch_layer->label) << " Val: ";
    for (size_t i = 0; i < 10; ++i)
        std::cout << couch_layer->node_ref(0, i).u_mem << " ";
    std::cout << std::endl;
}

/**
 * @brief This is the main function
 * @return int
 */
int main()
{

    phead = std::move(std::make_shared<head_t>());
    ptracer = std::move(std::make_shared<tracer_t<5, 28, 9>>(1920, 1200));

    create_net({5,
                {{TNN::RETINA, 28, 28},
                 {TNN::CORTEX, 14, 14},
                 {TNN::CORTEX, 7, 7},
                 {TNN::ACTUATOR, 1, 10},
                 {TNN::COUCHING, 1, 10}},

                {{0, 0, TNN::GAMK, 1, 1},
                 {0, 1, TNN::DOPHAMINE, 1, 1},
                 {1, 2, TNN::DOPHAMINE, 1, 1},
                 {2, 3, TNN::DOPHAMINE, 1, 1},
                 //
                 {4, 3, TNN::DOPHAMINE, 0, 1}}});

    // std::fstream ifile("../networks/net.out", std::ios::in);
    // ifile >> *phead;

    // Create a new mnist set
    auto pmnist = std::make_shared<mnist_set>();

    pmnist->init_set("../MNIST/train-images-idx3-ubyte",
                     "../MNIST/train-labels-idx1-ubyte", true);

    auto first_time = true;
    int times = 1;
    while (times-- != 0)
    {
        auto p = pmnist->next_image();

        if (p.first == nullptr)
            break;
        auto pp = p.first;

        // Set appropriate label
        auto pl = phead->layers.back();
        auto p_couch = std::dynamic_pointer_cast<mnist_couch_layer_t>(pl);
        p_couch->set_label(p.second);

        // std::fstream ofile("../networks/net.out", std::ios::out | std::ios::trunc);
        // ofile << *phead;
        // exit(0);

        if (first_time)
        {
            phead->wake_up(pp, 28, 28);
            first_time = false;
        }
        else
        {
            phead->look_at(pp);
            phead->set_focus(28, 28);
        }

        std::chrono::milliseconds timespan(100);

        for (auto i = 0; i < 100; ++i)
        {
            // phead->saccade(2.0);
            std::this_thread::sleep_for(timespan);
        }
    }

    std::fstream ofile("../networks/net.out", std::ios::out | std::ios::trunc);
    ofile << *phead;
    phead->print_output(3);
    std::cout << "nof events: " << nof_events.load() << std::endl;
    phead->do_sleep();

    return 0;
}
