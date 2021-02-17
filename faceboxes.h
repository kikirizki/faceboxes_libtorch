//
// Created by Kiki Rizki Arpiandi on 13/11/18.
//

#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include "../model/model.h"
#include "../priorbox/priorbox.h"

class FaceBoxes : public Model {


public:


    std::vector<cv::Rect> detect_on_image(cv::Mat &image);

    explicit FaceBoxes(const std::string &p_path, bool is_hp);

    explicit FaceBoxes();

    void preprocess_(cv::Mat &image) override;

    torch::Tensor intersect(torch::Tensor &box_a, torch::Tensor &box_b);

    void run();

private:

    std::vector<at::Tensor> forward_model(std::vector<torch::jit::IValue> &inputs) override;

    bool is_hp = false;

    PriorBox priorbox;
    torch::Tensor priors;

    std::vector<torch::jit::IValue> mat_2_ivalue(cv::Mat &mat);
};


#endif //FACE_DETECTOR_H
