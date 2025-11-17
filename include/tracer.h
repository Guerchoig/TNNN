#pragma once
#include "common.h"
struct params_t; // forward declaration of params struct defined in brain.h
#include "number_inp_fld.h"
// #include <atomic_queue.h>
#include <queue>
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <array>
#include <sstream>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>

namespace tr
{
    constexpr int inter_sells = 5;  // pixels
    constexpr int left_margin = 20; // pixels
    constexpr int top_margin = 200; // pixels
    constexpr int text_top = 100;   // pixels
    constexpr int char_size = 20;
    constexpr unsigned nof_dubbs = 2;
    constexpr int dubb_len = 300;
    constexpr int label_len = 150;
    constexpr float fade_out_rate = 0.1;
    constexpr std::uint8_t transparent = 0xFF;
    constexpr unsigned nof_sprites = 18;
    constexpr unsigned scene_width = mnist_size;
    constexpr unsigned magnification = 9;
    constexpr std::uint8_t no_attenuation = 0xFF;
    constexpr int scene_index_width = 300;
    constexpr clock_count_t period = tracer_period;
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

    std::mutex sfml_mutex;

    params_t *pparams = nullptr;

    // Param input fields container
    struct param_input_field_iface
    {
        virtual ~param_input_field_iface() = default;
        virtual void handleEvent(const sf::Event &event) = 0;
        virtual void draw(sf::RenderWindow &window) const = 0;
        virtual void updateParam() = 0; // push value to params
        virtual void setPosition(const sf::Vector2f &pos) = 0;
    };

    template <typename T>
    struct param_input_field_impl : param_input_field_iface
    {
        NumberInputField<T> field;
        std::atomic<T> *param_ptr;

        param_input_field_impl(const sf::Vector2f &pos, const sf::Vector2f &size,
                               const sf::Font &font, std::atomic<T> &param_ref,
                               T minVal, T maxVal, const std::string &label)
            : field(pos, size, font, minVal, maxVal, label), param_ptr(&param_ref)
        {
            field.setValue(param_ptr->load());
        }

        void handleEvent(const sf::Event &event) override { field.handleEvent(event); }
        void draw(sf::RenderWindow &window) const override { field.draw(window); }
        void updateParam() override
        {
            auto v = field.getValue();
            if (param_ptr->load() != v)
                param_ptr->store(v);
        }
        void setPosition(const sf::Vector2f &pos) override { field.setPosition(pos); }
    };

    std::vector<std::unique_ptr<param_input_field_iface>> param_fields;
    void init_param_fields();
    void update_param_field_positions();

    // Layout for param fields
    float param_scroll_y = 0.0f;
    float param_scroll_x = 0.0f;
    float param_field_width = 120.0f; // 120.0f;
    float param_field_height = 28.0f;
    float param_start_x = tr::left_margin + 450.0f;
    float param_start_y = 10.0f;
    float param_spacing_y = 34.0f;
    float param_column_spacing = 200.0f;
    int param_fields_per_column = 0;
    int param_num_columns = 0;

    void make_text_box(sf::Text &text, int len, sf::String &str);
    void make_black_mask(sf::RectangleShape &mask, int len, sf::Vector2f size);
    bool poll_for_closed_event();
    void fade_out_sprites();
    void lock_screen();
    void unlock_screen();
    void display_tracer_buf(std::shared_ptr<tracer_buf_t> item, clock_count_t time_moment);
    void set_scene_index(uint64_t index);
    void draw_scene_index();
    std::shared_ptr<tracer_buf_t> get_tracer_buf();
    void push_to_buf(std::shared_ptr<tracer_buf_t> pbuf, abs_address_t addr,
                     uint8_t color, clock_count_t time_moment);
    float compute_params_columns_total_width() const;
    tracer_t(uint32_t h_resolution,
             uint32_t v_resolution, params_t &params);
};
