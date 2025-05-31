#pragma once
// #include "mnist_reader.hpp"
#include "common.h"
#include <vector>
#include <array>
#include <memory>

using img_t = std::array<uint8_t, mnist_size * mnist_size>;
using img_set_t = std::vector<img_t>;

using labels_t = std::vector<uint8_t>;
constexpr size_t mnist_training_size = 60000;
constexpr size_t mnist_test_size = 10000;

struct mnist_set
{
    std::vector<scene_t> scenes;
    labels_t labels;
    uint64_t scene_index = 0;
    layer_dim_t i_label;
    void init_set(const std::string &img_path,
                  const std::string &lbl_path,
                  bool is_training);
    std::pair<scene_t *, uint8_t> next_image();
};
inline std::shared_ptr<mnist_set> pmnist;
