#pragma once
#include "common.h"
#include "tracer.h"
#include <atomic>
#include <mutex>

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
    head_interface_t *phead;
    std::mutex moving_gaze;

    void zoom(int _left, int _top, brain_coord_t _width, brain_coord_t _heigth)
    {
        moving_gaze.lock();
        left = _left;
        top = _top;
        right = left + _width - 1;
        bottom = top + _heigth - 1;
        moving_gaze.unlock();
    }

    void shift(int dx, int dy, float dist)
    {
        assert(dist > 0);
        moving_gaze.lock();
        int delta_x = dx * dist;
        int delta_y = dy * dist;
        left += delta_x;
        right += delta_x;
        top += delta_y;
        bottom += delta_y;
        moving_gaze.unlock();
    }

    void set_scene(scene_t *_pscene)
    {
        moving_gaze.lock();
        pscene = _pscene;
        scene_index++;
        phead->clear_scene_memory();
        moving_gaze.unlock();
    }

    scene_t *get_locked_scene()
    {
        moving_gaze.lock();
        auto res = pscene;
        return res;
    }

    void unlock_scene()
    {
        moving_gaze.unlock();
    }

    scene_signal_t get_signal(brain_coord_t _i, brain_coord_t _j)
    {
        auto i = _i + top;
        auto j = _j + left;
        if (i < 0)
            i = 0;
        if (i >= right)
            i = right - 1;
        if (j < 0)
            j = 0;
        if (j >= bottom)
            j = bottom - 1;

        auto res = pscene->at(i).at(j);

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
                  brain_coord_t heigth = view_field_def_heigth,
                  head_interface_t *phead = nullptr) : scene_index{0}, left{0}, top{0},
                                                       right(width - 1),
                                                       bottom(heigth - 1),
                                                       phead{phead}
    {
    }
    ~eyes_optics_t()
    {
        moving_gaze.try_lock();
        // here mutex is certanly locked
        moving_gaze.unlock();
    }
};