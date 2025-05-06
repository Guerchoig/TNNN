#pragma once
#include "common.h"
#include "tracer.h"
#include <atomic>

// View field
constexpr unsigned view_field_def_width = mnist_size;
constexpr unsigned view_field_def_heigth = mnist_size;

// Optics------------------------------------------------
struct eyes_optics_t
{

    std::shared_ptr<scene_t> pscene;
    layer_dim_t left;
    layer_dim_t top;
    layer_dim_t right;
    layer_dim_t bottom;
    std::mutex moving_gaze;
    void (*notify_scene_change)();

    void zoom(int _left, int _top, layer_dim_t _width, layer_dim_t _heigth)
    {
        std::lock_guard<std::mutex> l(moving_gaze);
        left = _left;
        top = _top;
        right = left + _width - 1;
        bottom = top + _heigth - 1;
    }

    void shift(int dx, int dy, float dist)
    {
        assert(dist > 0);
        std::lock_guard<std::mutex> l(moving_gaze);
        int delta_x = dx * dist;
        int delta_y = dy * dist;
        left += delta_x;
        right += delta_x;
        top += delta_y;
        bottom += delta_y;
    }

    void set_scene(std::shared_ptr<scene_t> _pscene)
    {
        std::lock_guard<std::mutex> l(moving_gaze);
        pscene = _pscene;
    }

    std::shared_ptr<scene_t> get_locked_scene()
    {
        moving_gaze.lock();
        auto res = std::shared_ptr<scene_t>(pscene);
        return res;
    }

    void unlock_scene()
    {
        moving_gaze.unlock();
    }

    eyes_optics_t(layer_dim_t width = view_field_def_width,
                  layer_dim_t heigth = view_field_def_heigth,
                  void (*_notify_scene_change)() = nullptr) : left{0}, top{0},
                                                              right(width - 1),
                                                              bottom(heigth - 1),
                                                              notify_scene_change(_notify_scene_change)
    {
    }

    void saccade(float dist)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> w(1, dist);
        const std::pair<int, int> dir[] = {{0, 1}, {0, -1}, {1, 0}, {1, 1}, //
                                           {1, -1},
                                           {-1, 1},
                                           {-1, -1},
                                           {-1, 0}};
        dist = w(gen);
        auto pdir = dir[rand() % (sizeof(dir) / sizeof(dir[0]))];
        zoom(pdir.first * dist, pdir.second * dist, view_field_def_width, view_field_def_heigth);
    }
};