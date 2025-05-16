#pragma once
#include "common.h"
// #include <atomic_queue.h>
#include <queue>
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <array>
#include <sstream>
#include <memory>
#include <mutex>

constexpr size_t queue_size = 60000;
constexpr uint8_t DARK = 0;

namespace tr
{
    constexpr int inter_sells = 5;  // pixels
    constexpr int left_margin = 20; // pixels
    constexpr int top_margin = 400; // pixels
    constexpr int text_top = 300;   // pixels
    constexpr int char_size = 20;
    constexpr unsigned nof_dubbs = 2;
    constexpr int dubb_len = 300;
    constexpr int label_len = 150;
    constexpr float decrease = 0.001;
    constexpr std::uint8_t transparent = 0xFF;
    constexpr unsigned nof_sprites = 6;
    constexpr unsigned scene_width = mnist_size;
    constexpr unsigned magnification = 9;
    constexpr std::uint8_t no_attenuation = 0xFF;
    constexpr int scene_index_width = 300;
}
enum dubbs_t
{
    SCENE_INDEX = 0
};

struct tracer_t
{
    const sf::Color text_fill_color = sf::Color::Green;
    unsigned h_resolution;
    unsigned v_resolution;
    unsigned vidgets_in_row;

    // layer's activity representation
    std::array<sf::Sprite, tr::nof_sprites> sprites{};
    sf::Texture sprites_texture;
    std::array<sf::RectangleShape, tr::nof_sprites> squares;
    std::array<std::array<std::array<rgba_t, tr::scene_width>, tr::scene_width>, tr::nof_sprites> colors{};

    // dubbs representation
    std::array<sf::Text, tr::nof_dubbs> texts{};
    std::array<sf::Text, tr::nof_dubbs> ltexts{};
    sf::Font font;
    std::array<sf::String, tr::nof_dubbs> labels = {{"Scene No: "}};
    std::array<sf::String, tr::nof_dubbs> strings = {{""}};
    uint64_t scene_index = 0LL;
    // Black mask to erase previous text
    sf::RectangleShape black_mask;
    sf::RenderWindow window;
    std::shared_ptr<void> phead;

    // drawing queue
    // std::queue<std::shared_ptr<tracer_buf_t>> queue;
    // std::mutex queue_mutex;
    // atomic_queue::AtomicQueue2<std::pair<neuron_address_t, unsigned long long int>, queue_size> queue;

    std::mutex sfml_mutex;

    void make_text_box(sf::Text &text, int len, sf::String &str)
    {
        text.setFont(font);
        text.setCharacterSize(tr::char_size);
        text.setFillColor(sf::Color::Red);
        // text.setScale(tr::magnification, tr::magnification);
        text.setPosition(len, tr::text_top);
        text.setString(str);
    }

    void make_black_mask(sf::RectangleShape &mask, int len, sf::Vector2f size)
    {
        mask.setPosition(len, tr::text_top);
        mask.setSize(size);
        mask.setFillColor(sf::Color::Black);
    }

    bool poll_for_closed_event()
    {
        std::lock_guard<std::mutex> lock(sfml_mutex);
        sf::Event event;

        window.setActive(true);
        // check all the window's events that were triggered since the last iteration of the loop
        bool res = true;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                res = false;
                break;
            }
        }
        window.setActive(false);
        return res;
    }

    // void close_window()
    // {
    //     std::lock_guard<std::mutex> lock(sfml_mutex);
    //     window.close();
    // }

    void fade_out_sprites()
    {
        for (unsigned i = 2; i < tr::nof_sprites; ++i)
            for (unsigned j = 0; j < tr::scene_width; ++j)
                for (unsigned k = 0; k < tr::scene_width; ++k)
                {
                    colors.at(i).at(j).at(k).g /= 2;
                }
    }

    void display_tracer_buf(std::shared_ptr<tracer_buf_t> item)
    {
        std::lock_guard<std::mutex> lock(sfml_mutex);

        window.setActive(true);

        // Fade out sprites
        fade_out_sprites();

        // Update signals
        for (auto it = item->begin(); it != item->end(); ++it)
        {
            auto addr = it->first;
            colors.at(addr.layer).at(addr.row).at(addr.col).g = it->second;
        }

        // Update screen
        sprites_texture.update(reinterpret_cast<std::uint8_t *>(colors.data()));
        for (unsigned i = 0; i < tr::nof_sprites; ++i)
        {
            window.draw(sprites.at(i));
            window.draw(squares.at(i));
        }

        draw_scene_index();

        window.display();
        window.setActive(false);
    }

    void set_scene_index(uint64_t index) { scene_index = index; }

    void draw_scene_index()
    {
        auto &text = texts.at(dubbs_t::SCENE_INDEX);

        std::stringstream ss;
        ss << scene_index;

        strings.at(dubbs_t::SCENE_INDEX) = ss.str();
        text.setString(strings.at(dubbs_t::SCENE_INDEX));

        window.draw(black_mask);
        window.draw(text);
    }
    tracer_t(uint32_t h_resolution,
             uint32_t v_resolution) : h_resolution(h_resolution),
                                                                   v_resolution(v_resolution)

    {
        for (unsigned i = 0; i < tr::nof_sprites; ++i)
            for (unsigned j = 0; j < tr::scene_width; ++j)
                for (unsigned k = 0; k < tr::scene_width; ++k)
                    colors.at(i).at(j).at(k).a = tr::no_attenuation;

        vidgets_in_row = (h_resolution - tr::left_margin) / (tr::inter_sells + tr::scene_width * tr::magnification);
        window.create(sf::VideoMode(h_resolution, v_resolution), "TNNN tracer");

        sprites_texture.create(tr::scene_width, tr::scene_width * tr::nof_sprites);

        for (unsigned i = 0; i < tr::nof_sprites; ++i)
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

        auto font_loaded = font.loadFromFile("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf");
        if (!font_loaded)
        {
            std::cerr << "Tracer font not loaded" << std::endl;
            exit(2);
        }

        for (unsigned i = 0; i < texts.size(); ++i)
        {
            make_text_box(ltexts.at(i), tr::left_margin + tr::dubb_len * i, labels.at(i));
            window.draw(ltexts.at(i));
            // strings.at(i).replace().resize(tr::dubb_len - tr::label_len, ' ');
            make_text_box(texts.at(i), tr::left_margin + tr::dubb_len * i + tr::label_len, strings.at(i));
        }
        make_black_mask(black_mask,
                        tr::left_margin + tr::label_len,
                        sf::Vector2f(tr::scene_index_width, tr::char_size));
    }
    // ~tracer_t()
    // {
    //     close_window();
    // }
};

using ptracer_t = std::shared_ptr<tracer_t>;
