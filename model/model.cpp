//
// Created by blackbeard on 08/10/18.
//
#include "model.h"


Model::Model(std::string &protobuf_path) {

}
bool Model::model_exist (const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}
Model::Model() = default;

std::vector<torch::jit::IValue> Model::_Mat2IValues(cv::Mat &mat) {
    std::vector<torch::jit::IValue> inputs;
    auto img_tensor = torch::from_blob(mat.data, {1, mat.rows, mat.cols, 3}).cuda();
    img_tensor = (img_tensor - 127.5) * 0.0078125;
    img_tensor = img_tensor.permute({0, 3, 1, 2});

    inputs.emplace_back(img_tensor);
    return inputs;
}

std::vector<torch::Tensor> Model::feed_image(cv::Mat &image) {
    cv::Mat copy;
    image.copyTo(copy);
    preprocess_(copy);
    //Possible bugs^
    auto inputs = Model::_Mat2IValues(copy);
    auto output = forward_model(inputs);
    return output;
}



