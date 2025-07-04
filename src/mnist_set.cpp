#include "mnist_reader.hpp"
#include "mnist_set.h"
#include "common.h"
#include <tuple>
using namespace mnist;
void mnist_set::init_set(const std::string &img_path,
                         const std::string &lbl_path,
                         bool is_training)
{
    auto limit = is_training ? mnist_training_size : mnist_test_size;
    scenes.resize(limit);
    labels.resize(limit);
    auto &p = *(reinterpret_cast<img_set_t *>(&scenes));
    auto res = read_mnist_image_file_flat<img_set_t>(p, img_path, limit, 0);
    if (!res)
    {
        throw std::runtime_error("Failed to read image file");
    }
    scene_index = 0;
    res = read_mnist_label_file_flat<labels_t>(labels, lbl_path, limit);
    if (!res)
    {
        throw std::runtime_error("Failed to read label file");
    }
    i_label = 0;
}

std::pair<scene_t *, uint8_t> mnist_set::next_image()
{
    if (static_cast<size_t>(scene_index) >= mnist_epoques) // scenes.size())
        return {nullptr, 0};
    return std::pair<scene_t *, uint8_t>(&(scenes[scene_index++]), labels[i_label++]);
}
