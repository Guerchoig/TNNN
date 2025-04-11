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

// void print_couch()
// {
//     auto couch_layer = std::dynamic_pointer_cast<mnist_couch_layer_t>(phead->layers[4]);
//     std::cout << "Label: " << static_cast<int>(couch_layer->label) << " Val: ";
//     for (size_t i = 0; i < 10; ++i)
//         std::cout << couch_layer->neuron_ref(0, i).u_mem << " ";
//     std::cout << std::endl;
// }

void test_draw()
{
    sf::Sprite sp;
    sf::Texture tx;
    tx.create(mnist_size, mnist_size * 5);
    std::array<std::array<std::array<rgba_t, mnist_size>, mnist_size>, 5> ar;
    for (uint i = 0; i < 5; i++)
    {
        for (uint j = 0; j < mnist_size; j++)
            for (uint k = 0; k < mnist_size; k++)
            {
                ar[i][j][k].r = i % 2 ? 0xFF : 0x00;
                ar[i][j][k].g = i % 2 ? 0x00 : 0xFF;
                ar[i][j][k].b = 0x00;
                ar[i][j][k].a = tr::no_attenuation;
            }
    }
    tx.update(reinterpret_cast<std::uint8_t *>(ar.data()));
    sp.setTexture(tx);
    sp.setScale(9, 9);
    for (uint i = 0; i < 5; i++)
    {
        sp.setTextureRect(sf::IntRect(0, i * mnist_size, mnist_size, mnist_size));
        sp.setPosition(200 + i * 27 * 9, 200);
        tr::window.draw(sp);
    }

    sf::Text text;
    text.setFont(ptracer->font);
    text.setCharacterSize(24);
    text.setPosition(100, 100);
    std::stringstream ss;
    ss << "Hello World";
    text.setString(ss.str());
    tr::window.draw(text);
    tr::window.display();
}

void main_loop()
{
    ptracer = std::move(std::make_shared<tracer_t>(1720, 1050, phead->p_eyes_optics->pscene,
                                                   &phead->pretina->scene_memories));
    // test_draw();
    // return;

    auto first_time = true;
    // while (first_time)
    while (tr::window.isOpen())
    {
        sf::Event event;
        // check all the window's events that were triggered since the last iteration of the loop
        while (tr::window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                tr::window.close();
            }
        }
        if (!tr::window.isOpen())
            break;



        std::pair<scene_t *, uint8_t> p;
        scene_t *pp = nullptr;

        p = pmnist->next_image();

        if (p.first == nullptr)
            break;
        pp = p.first;
        phead->p_eyes_optics->set_scene(pp);
        phead->p_eyes_optics->zoom(0, 0, mnist_size, mnist_size);
        if (first_time)
        {
            phead->wake_up(pp, mnist_size, mnist_size);
            first_time = false;
            // for(auto it= phead->workers.begin(); it != phead->workers.end(); ++it){
            //     DN(cast_to_pretina_worker(it->second)->input_events.get());
            // }
        }
        // Set appropriate label
        auto pl = phead->layers.back();
        auto p_couch = std::static_pointer_cast<mnist_couch_layer_t>(pl);
        p_couch->set_label(p.second);
        ptracer->scenes[0] = phead->p_eyes_optics->pscene;
        ptracer->show();
        std::chrono::milliseconds timespan(5000);
        std::this_thread::sleep_for(timespan);

        phead->p_eyes_optics->saccade(1.0);
    }

    std::fstream ofile("../networks/net.out", std::ios::out | std::ios::trunc);
    ofile << *phead;
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

    phead = std::move(std::make_shared<head_t>());
    pmnist = std::make_shared<mnist_set>();
    pmnist->init_set("../MNIST/train-images-idx3-ubyte",
                     "../MNIST/train-labels-idx1-ubyte", true);

    // quick_exit(0);

    create_net({5,
                {{TNN::RETINA, mnist_size, mnist_size},
                 {TNN::CORTEX, 14, 14},
                 {TNN::CORTEX, 7, 7},
                 {TNN::CORTEX, 14, 14},
                 {TNN::COUCHING, 1, 10}},

                {{0, 1, TNN::DOPHAMINE, 1, 1},
                 {1, 2, TNN::DOPHAMINE, 1, 1},
                 {2, 3, TNN::DOPHAMINE, 1, 1},
                 {3, 4, TNN::DOPHAMINE, 1, 1}}});

    main_loop();

    return 0;
}
