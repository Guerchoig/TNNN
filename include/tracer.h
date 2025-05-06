#pragma once
#include "common.h"
// #include <atomic_queue.h>
#include <queue>
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <array>
#include <sstream>
#include <memory>

constexpr size_t queue_size = 60000;
constexpr uint8_t DARK = 0;

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

}

std::shared_ptr<scene_t> get_locked_scene();
scene_t &get_memories_scene();
void unlock_scene();

struct tracer_t
{
    const sf::Color text_fill_color = sf::Color::Green;
    unsigned h_resolution;
    unsigned v_resolution;
    unsigned vidgets_in_row;
    std::atomic<unsigned long long int> nof_events = 0;
    
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
    sf::RenderWindow window;

    // drawing queue
    std::queue<std::pair<neuron_address_t, unsigned long long int>> queue;
    // atomic_queue::AtomicQueue2<std::pair<neuron_address_t, unsigned long long int>, queue_size> queue;

    std::mutex tracer_mutex;

    void make_text_box(sf::Text &text, int len, std::string &str)
    {
        text.setFont(font);
        text.setCharacterSize(tr::char_size);
        text.setFillColor(sf::Color::Red);
        // text.setScale(tr::magnification, tr::magnification);
        text.setPosition(len, tr::text_top);
        text.setString(str);
    }

    tracer_t(uint32_t h_resolution,
             uint32_t v_resolution) : h_resolution(h_resolution),
                                                    v_resolution(v_resolution)

    {
        for (unsigned i = 0; i < tr::nof_layers; ++i)
            for (unsigned j = 0; j < tr::scene_width; ++j)
                for (unsigned k = 0; k < tr::scene_width; ++k)
                    colors.at(i).at(j).at(k).a = tr::no_attenuation;

        vidgets_in_row = (h_resolution - tr::left_margin) / (tr::inter_sells + tr::scene_width * tr::magnification);
        window.create(sf::VideoMode(h_resolution, v_resolution), "TNNN tracer");

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
        }

        font.loadFromFile("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf");

        for (unsigned i = 0; i < texts.size(); ++i)
        {
            make_text_box(ltexts.at(i), tr::left_margin + tr::dubb_len * i, labels.at(i));
            strings.at(i).resize(tr::dubb_len - tr::label_len, ' ');
            make_text_box(texts.at(i), tr::left_margin + tr::dubb_len * i + tr::label_len, strings.at(i));
        }
    }
    bool poll_for_closed_event()
    {
        std::lock_guard<std::mutex> lock(tracer_mutex);
        sf::Event event;

        window.setActive(true);
        // check all the window's events that were triggered since the last iteration of the loop
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                window.close();
            }
        }

        auto res = window.isOpen();

        window.setActive(false);
        return res;
    }

    // void set_scene(std::shared_ptr<scene_t> _pscene)
    // {
    //     std::lock_guard<std::mutex> lock(tracer_mutex);
    //     scenes[0] = _pscene;
    // }

    void fade_out_sprite(unsigned num)
    {
        for (unsigned j = 0; j < tr::scene_width; ++j)
            for (unsigned k = 0; k < tr::scene_width; ++k)
            {
                // colors.at(num).at(j).at(k).a = tr::no_attenuation;
                colors.at(num).at(j).at(k).g = DARK;
            }
    }

    
    void show_scene()
    {
        std::lock_guard<std::mutex> lock(tracer_mutex);

        window.setActive(true);
        
        // Lock n process scenes
        auto pscene = get_locked_scene();
        auto &old_scene = get_memories_scene();

        for (unsigned j = 0; j < tr::scene_width; ++j)
            for (unsigned k = 0; k < tr::scene_width; ++k)
            {
                colors.at(0).at(j).at(k).g = pscene->at(j).at(k);
                colors.at(1).at(j).at(k).g = old_scene.at(j).at(k);
            }
        unlock_scene();
        
        // Update screen
        sprites_texture.update(reinterpret_cast<std::uint8_t *>(colors.data()));
        for (unsigned i = 0; i < 2; ++i)
        {
            window.draw(sprites.at(i));
            window.draw(squares.at(i));
        }
        window.display();
        window.setActive(false);
    }

    void show_output_events_buffer(events_output_buf_t &buffer)
    {
        if (!buffer.size())
            return;
        std::lock_guard<std::mutex> lock(tracer_mutex);
        window.setActive(true);
        std::pair<neuron_address_t, unsigned long long int> item;
        std::pair<address_t, std::unique_ptr<events_pack_t>> events_key_n_pack;

        for (auto it = buffer.begin(); it != buffer.end(); ++it)
        {
            // draw brain
            if (it->second.get() == nullptr)
                continue;
            for (auto it2 = it->second->begin(); it2 != it->second->end(); ++it2)
            {
                auto &item = *it2;

                auto layer = item.target_addr.layer + 2;
                // fade_out_sprite(layer);
                // update brain
                for (unsigned j = 0; j < tr::scene_width; ++j)
                    for (unsigned k = 0; k < tr::scene_width; ++k)
                    {

                        auto &val = colors.at(layer).at(j).at(k).b;
                        val = val != 0 ? val * (1 - tr::decrease) : 0x00;
                        colors.at(layer).at(j).at(k).a = tr::no_attenuation;
                    }

                colors.at(layer).at(item.target_addr.row).at(item.target_addr.col).b = 0xFF;
                colors.at(layer).at(item.target_addr.row).at(item.target_addr.col).a = tr::no_attenuation;

                sprites_texture.update(reinterpret_cast<std::uint8_t *>(colors.data()));

                for (unsigned i = 0; i < tr::nof_layers; ++i)
                {
                    window.draw(sprites.at(i));
                    window.draw(squares.at(i));
                }

                // strings.at(0) = std::to_string(item.second);
                // texts[0].setString(strings.at(0));
                // window.draw(texts.at(0));
                // for (unsigned i = 0; i < texts.size(); ++i)
                // {
                //     window.draw(ltexts.at(i));
                //     window.draw(texts.at(i));
                // }
            }
        }
        window.display();
        window.setActive(false);
    }
};
