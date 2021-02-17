//
// Created by Kiki Rizki Arpiandi on 13/11/18.
//

#ifndef FACEBOXES_PRIORBOX_H
#define FACEBOXES_PRIORBOX_H


#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/core/types.hpp>

class PriorBox {
public:
    explicit PriorBox(float im_height,float im_width);

    PriorBox();

    torch::Tensor forward();

private:
    int min_dim = 1024;
    std::vector<std::vector<float>> min_sizes_list = {{32, 64, 128},
                                                      {256},
                                                      {512}};
    std::vector<float> steps = {32, 64, 128};
    float image_size[2];
    bool clip = false;
    std::vector<std::pair<float,float>> feature_map_list;

};


#endif //FACEBOXES_PRIORBOX_H
