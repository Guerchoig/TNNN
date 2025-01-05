#include "tracer.h"

template <unsigned NofLayers, unsigned SceneWidth, int Magnification>
void tracer_t<NofLayers, SceneWidth, Magnification>::update(const neuron_address_t &addr)
{
    auto &layer = values[addr.layer];
    for (layer_dim_t i = 0; i < SceneWidth; ++i)
        for (layer_dim_t j = 0; j < SceneWidth; ++j)
        {
            layer[i][j].b = (layer[i][j].b > 0) ? layer[i][j].b * (1 - decrease) : 0;
            layer[i][j].a = 0xFF;
        }

    layer[addr.row][addr.col].b = 0xFF;

    textures[addr.layer].update(&layer);
    window.draw(sprites[addr.layer]);
    window.display();
}

template <unsigned NofLayers, unsigned SceneWidth, int Magnification>
tracer_t<NofLayers,
         SceneWidth,
         Magnification>::tracer_t(uint32_t h_resolution,
                                  uint32_t v_resolution) : h_resolution(h_resolution),
                                                           v_resolution(v_resolution),
                                                           window(sf::VideoMode(h_resolution,
                                                                                v_resolution),
                                                                  "TNNN tracer"),
                                                           sells_in_row{(h_resolution - left_margin) / (inter_sells + SceneWidth)}
{
    for (auto i = 0; i < NofLayers; ++i)
    {
        textures[i].create(SceneWidth, SceneWidth);
        sprites[i].setTexture(textures[i], true);
        sprites[i].setScale(Magnification, Magnification);
        auto xpos = i % sells_in_row;
        auto ypos = i / sells_in_row;
        sprites[i].setPosition(left_margin + xpos * (SceneWidth + inter_sells),
                               top_margin + ypos * (SceneWidth + inter_sells));
    }
}