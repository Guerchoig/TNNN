#pragma once

#include "common.h"
#include "tracer.h"
#include <atomic>
#include <mutex>
#include <cassert>

// View field
constexpr unsigned view_field_def_width = mnist_size;
constexpr unsigned view_field_def_heigth = mnist_size;

// Optics------------------------------------------------
struct eyes_optics_t
{

    scene_t *pscene;
    uint64_t scene_index;
    brain_coord_t left;
    brain_coord_t top;
    brain_coord_t right;
    brain_coord_t bottom;

    void zoom(int _left, int _top, brain_coord_t _width, brain_coord_t _heigth)
    {

        left = _left;
        top = _top;
        right = left + _width - 1;
        bottom = top + _heigth - 1;
    }

    void shift(int dx, int dy, float dist)
    {
        assert(dist > 0);

        int delta_x = dx * dist;
        int delta_y = dy * dist;
        left += delta_x;
        right += delta_x;
        top += delta_y;
        bottom += delta_y;
    }

    void set_scene(scene_t *_pscene)
    {

        pscene = _pscene;
        scene_index++;
        // phead->clear_scene_memory();
    }

    scene_t *get_scene()
    {

        auto res = pscene;
        return res;
    }

    scene_signal_t get_signal(brain_coord_t _i)
    {

        auto res = pscene->at(_i);

        return res;
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

    eyes_optics_t(brain_coord_t width = view_field_def_width,
                  brain_coord_t heigth = view_field_def_heigth) : scene_index{0}, left{0}, top{0},
                                                                  right(width - 1),
                                                                  bottom(heigth - 1)
    {
    }
};