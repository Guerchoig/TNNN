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

void print_couch()
{
    auto couch_layer = std::dynamic_pointer_cast<mnist_couch_layer_t>(phead->layers[4]);
    std::cout << "Label: " << static_cast<int>(couch_layer->label) << " Val: ";
    for (size_t i = 0; i < 10; ++i)
        std::cout << couch_layer->node_ref(0, i).u_mem << " ";
    std::cout << std::endl;
}

void test_draw()
{
    sf::Sprite sp;
    sf::Texture tx;
    tx.create(28, 28 * 5);
    std::array<std::array<std::array<rgba_t, 28>, 28>, 5> ar;
    for (uint i = 0; i < 5; i++)
    {
        for (uint j = 0; j < 28; j++)
            for (uint k = 0; k < 28; k++)
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
        sp.setTextureRect(sf::IntRect(0, i * 28, 28, 28));
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
                                                   &phead->p_eyes_optics->prev_view));
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

        // ptracer->update({0, 7, 7});
        // continue;

        std::pair<scene_t *, uint8_t> p;
        scene_t *pp = nullptr;
        // if (first_time)
        // {
        p = pmnist->next_image();

        if (p.first == nullptr)
            break;
        pp = p.first;
        phead->look_at(pp);
        phead->set_focus(0, 0, 28, 28);
        if (first_time)
        {
            phead->wake_up(pp, 28, 28);
            first_time = false;
        }
        // Set appropriate label
        auto pl = phead->layers.back();
        auto p_couch = std::dynamic_pointer_cast<mnist_couch_layer_t>(pl);
        p_couch->set_label(p.second);
        ptracer->scenes[0] = phead->p_eyes_optics->pscene;
        // }
        // std::fstream ofile("../networks/net.out", std::ios::out | std::ios::trunc);
        // ofile << *phead;
        // exit(0);
        ptracer->show();
        std::chrono::milliseconds timespan(5000);
        std::this_thread::sleep_for(timespan);

        // phead->saccade(10.0);
    }

    std::fstream ofile("../networks/net.out", std::ios::out | std::ios::trunc);
    ofile << *phead;
    // phead->print_output(3);
    // std::cout << "nof events: " << ptracer->nof_events.load() << std::endl;
    phead->do_sleep();
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

    main_loop();

    return 0;
}
