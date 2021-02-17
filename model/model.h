#ifndef FACEDETECTION_MODEL_H
#define FACEDETECTION_MODEL_H

#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
class Model {
public:
    explicit Model(std::string &protobuf_path);

    Model();

    virtual void preprocess_(cv::Mat &image) = 0;

    virtual std::vector <at::Tensor> forward_model(std::vector <torch::jit::IValue> &inputs) = 0;

    std::vector <torch::jit::IValue> _Mat2IValues(cv::Mat &mat);

    torch::jit::script::Module script_module;

    torch::DeviceType device_type;

    std::vector <torch::Tensor> feed_image(cv::Mat &image);

    bool model_exist(const std::string &name);

    torch::jit::script::Module anchor_module;
};


#endif //FACEDETECTION_MODEL_H





