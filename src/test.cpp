#include "brain.h"
#include "input_output.h"
#include "mnist_set.h"
#include "tracer.h"

#include <vector>
#include <array>
#include <sstream>
#include <fstream>
#include <optional>

using namespace TNN;

void main_loop(phead_t phead, ptracer_t ptracer)
{

    auto first_time = true;

    while (ptracer->poll_for_closed_event())
    {
        std::pair<scene_t *, uint8_t> p;

        p = pmnist->next_image();

        if (p.first == nullptr)
            break; // No more images

        // Set appropriate scene
        phead->p_eyes_optics->set_scene(p.first);
        auto index = phead->p_eyes_optics->scene_index;
        ptracer->set_scene_index(index);

        phead->p_eyes_optics->zoom(0, 0, mnist_size, mnist_size);

        // Set appropriate label
        auto pl = phead->layers.back();
        auto p_couch = std::static_pointer_cast<mnist_couch_layer_t>(pl);
        p_couch->set_label(p.second);

        // Think about scene
        if (first_time)
        {
            phead->wake_up();
            first_time = false;
        }

        std::chrono::milliseconds timespan(100);
        std::this_thread::sleep_for(timespan);

        // phead->p_eyes_optics->saccade(1.0);
    }

    try
    {
        std::fstream ofile("../networks/net.out", std::ios::out | std::ios::trunc);
        ofile << *phead;
    }
    catch (...)
    {
        std::cout << "Error saving network" << std::endl;
    }
    phead->go_to_sleep();
}

/**
 * @brief This is the main function
 * @return int
 */

int main()
{
    auto ptracer = std::move(std::make_shared<tracer_t>(1720, 1050));
    {
        auto phead = std::move(std::make_shared<head_t>());
        ptracer->phead = phead;

        pmnist = std::make_shared<mnist_set>();
        pmnist->init_set("../MNIST/train-images-idx3-ubyte",
                         "../MNIST/train-labels-idx1-ubyte", true);

        // quick_exit(0);

        create_net(network_descr_t({{TNN::RETINA, mnist_size, mnist_size},
                                    {TNN::CORTEX, mnist_size / 2, mnist_size / 2},
                                    {TNN::CORTEX, mnist_size / 2, mnist_size / 2},
                                    //  {TNN::CORTEX, 14, 14},
                                    {TNN::COUCHING, 1, 10}},

                                   {
                                       {0, 1, TNN::DOPHAMINE, 14, 1},
                                       {1, 2, TNN::DOPHAMINE, 14, 1},
                                       {2, 3, TNN::DOPHAMINE, 10, 1},
                                       //  {3, 4, TNN::DOPHAMINE, 1, 1}
                                   }));

        main_loop(phead, ptracer);
    }
    return 0;
}
