#pragma once
#include "common.h"
#include <atomic_queue.h>
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <array>
#include <sstream>
#include <memory>

namespace tr
{
    constexpr int inter_sells = 5;  // pixels
    constexpr int left_margin = 20; // pixels
    constexpr int top_margin = 400; // pixels
    constexpr int text_top = 300;   // pixels
    constexpr int char_size = 20;
    constexpr int dubb_len = 300;
    constexpr int label_len = 150;
    constexpr float decrease = 0.001;
    constexpr std::uint8_t transparent = 0xFF;
    constexpr unsigned nof_layers = 5;
    constexpr unsigned scene_width = mnist_size;
    constexpr unsigned magnification = 9;
    constexpr std::uint8_t no_attenuation = 0xFF;

    inline sf::RenderWindow window;
}


struct tracer_t
{
    const sf::Color text_fill_color = sf::Color::Green;
    unsigned h_resolution;
    unsigned v_resolution;
    unsigned vidgets_in_row;
    std::atomic<unsigned long long int> nof_events = 0;
    std::array<const scene_t *, 2> scenes; // scene, prev_scene

    // layer's activity representation
    std::array<sf::Sprite, tr::nof_layers> sprites;
    sf::Texture sprites_texture;
    std::array<sf::RectangleShape, tr::nof_layers> squares;
    std::array<std::array<std::array<rgba_t, tr::scene_width>, tr::scene_width>, tr::nof_layers> colors;

    // dubbs representation
    std::array<sf::Text, 1> texts;
    std::array<sf::Text, 1> ltexts;
    sf::Font font;
    std::array<std::string, 1> labels = {{"Events: "}};
    std::array<std::string, 1> strings = {{""}};

    // drawing queue

    atomic_queue::AtomicQueue2<std::pair<neuron_address_t, unsigned long long int>, 1000> queue;

    // std::mutex m;

    void mt(sf::Text &text, int len, std::string &str)
    {
        text.setFont(font);
        text.setCharacterSize(tr::char_size);
        text.setFillColor(sf::Color::Red);
        // text.setScale(tr::magnification, tr::magnification);
        text.setPosition(len, tr::text_top);
        text.setString(str);
    }

    tracer_t(uint32_t h_resolution,
             uint32_t v_resolution,
             const scene_t *scene,
             const scene_t *prev_scene) : h_resolution(h_resolution),
                                          v_resolution(v_resolution),
                                          scenes{scene, prev_scene}
    {
        for (unsigned i = 0; i < tr::nof_layers; ++i)
            for (unsigned j = 0; j < tr::scene_width; ++j)
                for (unsigned k = 0; k < tr::scene_width; ++k)
                    colors.at(i).at(j).at(k).a = tr::no_attenuation;

        vidgets_in_row = (h_resolution - tr::left_margin) / (tr::inter_sells + tr::scene_width * tr::magnification);
        tr::window.create(sf::VideoMode(h_resolution, v_resolution), "TNNN tracer");

        sprites_texture.create(tr::scene_width, tr::scene_width * tr::nof_layers);

        for (unsigned i = 0; i < tr::nof_layers; ++i)
        {
            // Draw vidget
            auto xpos = tr::left_margin + (i % vidgets_in_row) * (tr::scene_width * tr::magnification + tr::inter_sells);
            auto ypos = tr::top_margin + (i / vidgets_in_row) * (tr::scene_width * tr::magnification + tr::inter_sells);
            sprites.at(i).setPosition(xpos, ypos);
            sprites.at(i).setTexture(sprites_texture);
            sprites.at(i).setTextureRect(sf::IntRect(0, i * tr::scene_width, tr::scene_width, tr::scene_width));
            sprites.at(i).setScale(tr::magnification, tr::magnification);

            squares.at(i).setPosition(xpos - 1, ypos - 1);
            squares.at(i).setSize(sf::Vector2f(tr::scene_width * tr::magnification + 2, tr::scene_width * tr::magnification + 2));
            squares.at(i).setFillColor(sf::Color::Transparent);
            squares.at(i).setOutlineColor(sf::Color::Green);
            squares.at(i).setOutlineThickness(1);

            // tr::window.draw(sprites.at(i));
            // tr::window.draw(squares.at(i));
        }

        font.loadFromFile("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf");

        for (unsigned i = 0; i < texts.size(); ++i)
        {
            mt(ltexts.at(i), tr::left_margin + tr::dubb_len * i, labels.at(i));
            strings.at(i).resize(tr::dubb_len - tr::label_len, ' ');
            mt(texts.at(i), tr::left_margin + tr::dubb_len * i + tr::label_len, strings.at(i));
            // tr::window.draw(ltexts.at(i));
        }

        // tr::window.display();
        // tr::window.setActive(false); // Required for multithread!!
    }

    void update([[maybe_unused]] const neuron_address_t &addr)
    {
        if(!queue.try_push(std::pair<neuron_address_t, unsigned long long int>(addr, nof_events.fetch_add(1, std::memory_order_relaxed))))
            throw std::runtime_error("Tracer: queue is full");
    }

    void show()
    {
        tr::window.setActive(true);
        std::pair<neuron_address_t, unsigned long long int> item;
        while (queue.try_pop(item))
        {
            tr::window.clear(sf::Color::Black);

            // update scene & prev scene
            for (unsigned j = 0; j < tr::scene_width; ++j)
                for (unsigned k = 0; k < tr::scene_width; ++k)
                {
                    colors.at(0).at(j).at(k).g = scenes[0]->at(j).at(k);
                    colors.at(1).at(j).at(k).g = scenes[1]->at(j).at(k);
                }

            // update brain
            for (unsigned j = 0; j < tr::scene_width; ++j)
                for (unsigned k = 0; k < tr::scene_width; ++k)
                {
                    auto layer = item.first.layer + 2;
                    auto &val = colors.at(layer).at(j).at(k).b;
                    val = val != 0 ? val * (1 - tr::decrease) : 0x00;
                    colors.at(layer).at(j).at(k).a = tr::no_attenuation;
                }

            colors.at(item.first.layer + 2).at(item.first.row).at(item.first.col).b = 0xFF;
            colors.at(item.first.layer + 2).at(item.first.row).at(item.first.col).a = tr::no_attenuation;

            sprites_texture.update(reinterpret_cast<std::uint8_t *>(colors.data()));

            for (unsigned i = 0; i < tr::nof_layers; ++i)
            {
                tr::window.draw(sprites.at(i));
                tr::window.draw(squares.at(i));
            }

            strings.at(0) = std::to_string(item.second);
            texts[0].setString(strings.at(0));
            tr::window.draw(texts.at(0));
            for (unsigned i = 0; i < texts.size(); ++i)
            {
                tr::window.draw(ltexts.at(i));
                tr::window.draw(texts.at(i));
            }
        }
        tr::window.display();
        tr::window.setActive(false);
    }
};
