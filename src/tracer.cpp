#include "tracer.h"
#include "brain.h" // for params_t and pparams access
#include <sstream>

using namespace std;

void tracer_t::make_text_box(sf::Text &text, int len, sf::String &str)
{
    text.setFont(font);
    text.setCharacterSize(tr::char_size);
    text.setFillColor(sf::Color::Red);
    text.setPosition(len, tr::text_top);
    text.setString(str);
}

void tracer_t::make_black_mask(sf::RectangleShape &mask, int len, sf::Vector2f size)
{
    mask.setPosition(len, tr::text_top);
    mask.setSize(size);
    mask.setFillColor(sf::Color::Black);
}

bool tracer_t::poll_for_closed_event()
{
    std::lock_guard<std::mutex> lock(sfml_mutex);
    sf::Event event;

    window.setActive(true);
    bool res = true;
    while (window.pollEvent(event))
    {
        if (event.type == sf::Event::Closed)
        {
            res = false;
            break;
        }

        for (auto &p : param_fields)
            p->handleEvent(event);
        if (event.type == sf::Event::MouseWheelScrolled)
        {
            // Map vertical wheel to horizontal scroll of columns
            float delta = event.mouseWheelScroll.delta;
            float step = (param_field_width + param_column_spacing) * 0.5f;
            param_scroll_x -= delta * step; // subtract so wheel up moves left
            // clamp
            float totalWidth = compute_params_columns_total_width();
            float availWidth = static_cast<float>(h_resolution) - tr::left_margin - 40.0f;
            if (totalWidth > availWidth)
            {
                if (param_scroll_x < 0)
                    param_scroll_x = 0;
                float maxScroll = totalWidth - availWidth;
                if (param_scroll_x > maxScroll)
                    param_scroll_x = maxScroll;
            }
            else
                param_scroll_x = 0;
            update_param_field_positions();
        }
    }

    // After processing events, push updated values to params
    for (auto &p : param_fields)
        p->updateParam();

    window.setActive(false);
    return res;
}

void tracer_t::fade_out_sprites()
{
    for (unsigned i = 1; i < tr::nof_sprites; ++i)
        for (unsigned j = 0; j < tr::scene_width; ++j)
            for (unsigned k = 0; k < tr::scene_width; ++k)
            {
                colors.at(i).at(j).at(k).g *= (1 - tr::fade_out_rate);
            }
}

void tracer_t::lock_screen() { sfml_mutex.lock(); }
void tracer_t::unlock_screen() { sfml_mutex.unlock(); }

void tracer_t::display_tracer_buf(std::shared_ptr<tracer_buf_t> item, clock_count_t time_moment)
{
    if (time_moment % tr::period != 0)
        return;

    std::lock_guard<std::mutex> lock(sfml_mutex);

    window.setActive(true);

    // Fade out sprites
    fade_out_sprites();

    // Update signals
    for (auto it = item->begin(); it != item->end(); ++it)
    {
        auto addr = it->first;
        colors.at(addr.layer).at(addr.abs_row).at(addr.abs_col).g = it->second;
    }

    // Update screen
    sprites_texture.update(reinterpret_cast<std::uint8_t *>(colors.data()));
    for (unsigned i = 0; i < tr::nof_sprites; ++i)
    {
        window.draw(sprites.at(i));
        window.draw(squares.at(i));
    }

    draw_scene_index();

    // Draw parameter input fields
    for (auto &p : param_fields)
        p->draw(window);

    window.display();
    window.setActive(false);
    item->clear();
}

void tracer_t::set_scene_index(uint64_t index) { scene_index = index; }

void tracer_t::draw_scene_index()
{
    auto &text = texts.at(dubbs_t::SCENE_INDEX);

    std::stringstream ss;
    ss << scene_index;

    strings.at(dubbs_t::SCENE_INDEX) = ss.str();
    text.setString(strings.at(dubbs_t::SCENE_INDEX));

    window.draw(black_mask);
    window.draw(text);
}

std::shared_ptr<tracer_buf_t> tracer_t::get_tracer_buf()
{
    return std::make_shared<tracer_buf_t>();
}

void tracer_t::push_to_buf(std::shared_ptr<tracer_buf_t> pbuf, abs_address_t addr,
                           uint8_t color, clock_count_t time_moment)
{
    if (time_moment % tr::period == 0)
        pbuf->push_back(std::make_pair<abs_address_t, uint8_t>(std::move(addr),
                                                               std::move(color)));
}

tracer_t::tracer_t(uint32_t h_resolution,
                   uint32_t v_resolution, params_t &params) : h_resolution(h_resolution),
                                                              v_resolution(v_resolution), pparams(&params)
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
        make_text_box(texts.at(i), tr::left_margin + tr::dubb_len * i + tr::label_len, strings.at(i));
    }
    make_black_mask(black_mask,
                    tr::left_margin + tr::label_len,
                    sf::Vector2f(tr::scene_index_width, tr::char_size));
}

void tracer_t::init_param_fields()
{
    if (!pparams)
        return;

    sf::Vector2f fieldSize(param_field_width, param_field_height);
    float startX = param_start_x;
    float startY = param_start_y;
    float spacingY = param_spacing_y;

#define ADD_FIELD(type, member, minv, maxv, label) \
    param_fields.emplace_back(std::make_unique<param_input_field_impl<type>>(sf::Vector2f(0, 0), fieldSize, font, pparams->member, minv, maxv, label));

    // Network dynamic params --------------------------------
    // Detector
    ADD_FIELD(potential_t, detector_alpha, (potential_t)0.0, (potential_t)100.0, "detector_alpha");
    // Threshold
    ADD_FIELD(potential_t, th_adapt_inc_per_spike, (potential_t)0.0, (potential_t)10.0, "th_inc_per_spike");
    ADD_FIELD(potential_t, threshold_decrease_time, (potential_t)0.0, (potential_t)100.0, "th_decrease_time");
    // Membrana
    ADD_FIELD(potential_t, umem_decrease_time, (potential_t)0.0, (potential_t)1000.0, "umem_decrease_time");
    ADD_FIELD(potential_t, u_rest, (potential_t)0.0, (potential_t)10.0, "u_rest");

    // Network static params -------------------------------
    ADD_FIELD(brain_coord_t, nof_cathegories, (brain_coord_t)1, (brain_coord_t)256, "nof_cathegories");
    ADD_FIELD(brain_coord_t, usual_nof_pieces_per_layer, (brain_coord_t)1, (brain_coord_t)256, "usual_nof_pieces_per_layer");
    ADD_FIELD(brain_coord_t, nof_pieces_in_last_layer, (brain_coord_t)1, (brain_coord_t)256, "nof_pieces_in_last_layer");

    // Neuron thresholds
    ADD_FIELD(potential_t, initial_neuron_threshold, (potential_t)0.0, (potential_t)10.0, "init_th_per_inp");
    ADD_FIELD(potential_t, normal_threshold_base, (potential_t)-10.0, (potential_t)10.0, "normal_th_base");

    // Visual detector params
    ADD_FIELD(scene_signal_t, max_scene_amplitude, (scene_signal_t)0, (scene_signal_t)65535, "max_scene_amplitude");
    ADD_FIELD(potential_t, amplitude_quant_size, (potential_t)1.0, (potential_t)265.0, "amplitude_quant_size");

    // Weights update
    ADD_FIELD(potential_t, dw_max, (potential_t)0.0, (potential_t)1.0, "dw_max");
    ADD_FIELD(potential_t, dw_plus_time, (potential_t)0.0, (potential_t)100.0, "dw_plus_time");
    ADD_FIELD(potential_t, dw_minus_time, (potential_t)0.0, (potential_t)100.0, "dw_minus_time");

#undef ADD_FIELD

    // Compute layout and place fields
    update_param_field_positions();
}

void tracer_t::update_param_field_positions()
{
    // Determine area height based on top margin
    float areaHeight = tr::top_margin - param_start_y - 2; // provide small margin
    if (areaHeight <= 0)
        areaHeight = param_field_height * 4;

    // compute fields per column
    param_fields_per_column = static_cast<int>(std::floor(areaHeight / param_spacing_y));
    if (param_fields_per_column <= 0)
        param_fields_per_column = 1;

    int total = static_cast<int>(param_fields.size());
    param_num_columns = (total + param_fields_per_column - 1) / param_fields_per_column;

    for (int idx = 0; idx < total; ++idx)
    {
        int col = idx / param_fields_per_column;
        int row = idx % param_fields_per_column;
        float xpos = param_start_x + col * (param_field_width + param_column_spacing);
        float ypos = param_start_y + row * param_spacing_y - param_scroll_y;
        // Set position adjusted by horizontal scroll
        sf::Vector2f pos(xpos - param_scroll_x, ypos);
        param_fields[idx]->setPosition(pos);
        // Instead, set the field's internal position via dynamic cast and setPosition
        // We can rely on the stored NumberInputField being the first member of wrapper
        // but safer: provide a method on param_input_field_iface to set position — add it
    }
}

float tracer_t::compute_params_columns_total_width() const
{
    if (param_num_columns <= 0)
        return 0.0f;
    return param_num_columns * (param_field_width + param_column_spacing) - param_column_spacing;
}
