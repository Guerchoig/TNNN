#include "common.h"
#include <SFML/Graphics.hpp>

constexpr int inter_sells = 5;  // pixels
constexpr int left_margin = 20; // pixels
constexpr int top_margin = 400; // pixels
constexpr float decrease = 0.1;

template <unsigned NofLayers, unsigned SceneWidth, int Magnification>
struct tracer_t
{
    unsigned h_resolution;
    unsigned v_resolution;
    sf::RenderWindow window;
    std::array<sf::Texture, NofLayers> textures;
    std::array<sf::Sprite, NofLayers> sprites;
    std::array<sf::Color[SceneWidth][SceneWidth], NofLayers> values;
    unsigned sells_in_row;

    tracer_t(uint32_t h_resolution, uint32_t v_resolution);
    ~tracer_t() { window.close(); }
    void update(const neuron_address_t &addr);
};
inline std::shared_ptr<tracer_t<5, 28, 9>> ptracer;