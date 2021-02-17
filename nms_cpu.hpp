//
// Created by Kiki Rizki Arpiandi on 13/11/18.
//
#include <torch/torch.h>


torch::Tensor nms_cpu(
        const torch::Tensor &dets,
        const torch::Tensor &scores,
        const float iou_threshold);
