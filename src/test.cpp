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

void main_loop()
{

    auto first_time = true;

    while (ptracer->poll_for_closed_event())
    {
        std::pair<scene_t *, uint8_t> p;
        std::shared_ptr<scene_t> pp;

        p = pmnist->next_image();

        if (p.first == nullptr)
            break;
        pp = std::shared_ptr<scene_t>(p.first);
        phead->p_eyes_optics->set_scene(pp);
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

        std::chrono::milliseconds timespan(50);
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
    // phead->print_output(3);
    // std::cout << "nof events: " << ptracer->nof_events.load() << std::endl;
    phead->go_to_sleep();
}

/**
 * @brief This is the main function
 * @return int
 */

int main()
{
    ptracer = std::make_shared<tracer_t>(1720, 1050);

    phead = std::move(std::make_shared<head_t>());
    
    pmnist = std::make_shared<mnist_set>();
    pmnist->init_set("../MNIST/train-images-idx3-ubyte",
                     "../MNIST/train-labels-idx1-ubyte", true);

    // quick_exit(0);

    create_net({3,
                {{TNN::RETINA, mnist_size, mnist_size},
                 {TNN::CORTEX, 14, 14},
                 //  {TNN::CORTEX, 7, 7},
                 //  {TNN::CORTEX, 14, 14},
                 {TNN::COUCHING, 1, 10}},

                {
                    {0, 1, TNN::DOPHAMINE, 1, 1},
                    {1, 2, TNN::DOPHAMINE, 1, 1} //,
                    //  {2, 3, TNN::DOPHAMINE, 1, 1},
                    //  {3, 4, TNN::DOPHAMINE, 1, 1}
                }});

    main_loop();

    return 0;
}
