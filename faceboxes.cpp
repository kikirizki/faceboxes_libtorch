//
// Created by Kiki Rizki Arpiandi on 13/11/18.
//

#include "faceboxes.h"
#include "../priorbox/priorbox.h"
#include "nms_cpu.hpp"

torch::Tensor intersect(torch::Tensor &box_a, torch::Tensor &box_b) {
    uint n = box_a.size(0);
    uint A = box_a.size(1);
    uint B = box_b.size(1);
    auto max_xy = torch::min(
            box_a.slice(2, 2, 4).unsqueeze(2).expand({n, A, B, 2}),
            box_b.slice(2, 2, 4).unsqueeze(1).expand({n, A, B, 2}));

    auto min_xy = torch::max(
            box_a.slice(2, 0, 2).unsqueeze(2).expand({n, A, B, 2}),
            box_b.slice(2, 0, 2).unsqueeze(1).expand({n, A, B, 2}));
    auto inter = torch::clamp((max_xy - min_xy), 0);
    return inter.slice(3, 0, 1).squeeze(3) * inter.slice(3, 1, 2).squeeze(3);
}

torch::Tensor decode(torch::Tensor loc, torch::Tensor priors) {
    auto boxes = torch::cat({
                                    priors.slice(1, 0, 2) +
                                    loc.slice(1, 0, 2) * 0.1 * priors.slice(1, 2, priors.size(1)),
                                    priors.slice(1, 2, priors.size(1)) * torch::exp(loc.slice(1, 2, loc.size(1)) * 0.2)
                            },
                            1);

//    boxes[:, :2] -= boxes[:, 2:] / 2
//    boxes[:, 2:] += boxes[:, :2]
    boxes.slice(1, 0, 2) -= boxes.slice(1, 2, boxes.size(1)) / 2;
    boxes.slice(1, 2, boxes.size(1)) += boxes.slice(1, 0, 2);
    return boxes;
}


FaceBoxes::FaceBoxes() {

}

std::vector<torch::jit::IValue> FaceBoxes::mat_2_ivalue(cv::Mat &mat) {
    std::vector<torch::jit::IValue> inputs;
    auto img_tensor = torch::from_blob(mat.data,
                                       {1, mat.rows, mat.cols, 3});
    if (FaceBoxes::is_hp) {
        std::cout<<"Half Precission ... "<<std::endl;
        img_tensor = img_tensor.permute({0, 3, 1, 2}).to(at::kHalf).cuda();
    } else {
        img_tensor = img_tensor.permute({0, 3, 1, 2}).cuda();
    }
    inputs.emplace_back(img_tensor);
    return inputs;
}

torch::Tensor jaccard(torch::Tensor &box_a, torch::Tensor &box_b) {
    bool use_batch = true;
    bool iscrowd = false;
    if (box_a.dim() == 2) {
        use_batch = false;
        //TODO : implement those
        // box_a = box_a[None, ...]
        // box_b = box_b[None, ...]
    }
    auto inter = intersect(box_a, box_b);

//    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
//            (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]


    auto area_a = ((box_a.slice(2, 2, 3).squeeze(2) - box_a.slice(2, 0, 1).squeeze(2))
                   * (box_a.slice(2, 3, 4).squeeze(2) - box_a.slice(2, 1, 2).squeeze(2)))
            .unsqueeze(2).expand_as(inter);

    auto area_b = ((box_b.slice(2, 2, 3).squeeze(2) - box_b.slice(2, 0, 1).squeeze(2))
                   * (box_b.slice(2, 3, 4).squeeze(2) - box_b.slice(2, 1, 2).squeeze(2)))
            .unsqueeze(1).expand_as(inter);

    auto unio = area_a + area_b - inter;

    torch::Tensor out;
    if (iscrowd) {
        out = inter.div(area_a);
    } else {
        out = inter.div(unio);
    }
    if (use_batch) {
        return out;
    } else {
        return out.squeeze(0);
    }
}


torch::Tensor fast_nms(torch::Tensor &boxes, torch::Tensor &scores, float iou_threshold) {
    std::cout << " Boxes size " << boxes.size(0) << " " << boxes.size(1) << std::endl;
    std::cout << " Score size " << scores.size(0) << std::endl;
    return torch::Tensor();
}


std::vector<cv::Rect> FaceBoxes::detect_on_image(cv::Mat &image) {
    float image_width = image.cols;
    float image_height = image.rows;
    cv::Mat copy;
    image.copyTo(copy);
    preprocess_(copy);
    auto inputs = mat_2_ivalue(copy);
    auto outputs = FaceBoxes::forward_model(inputs);
    auto loc = outputs.at(0);
    auto conf = outputs.at(1);


    FaceBoxes::priors = FaceBoxes::priors.cuda();

    float scale_data[4] = {image_width, image_height, image_width, image_height};
    auto scale = torch::from_blob(scale_data, {4}).cuda();
    auto boxes = decode(loc.squeeze(0), priors);
    boxes = boxes * scale;
    boxes = boxes.cpu();
    int top_k = 5000;
    //Revome low confidence
    auto scores = conf.squeeze(0).cpu().select(1, 1).to(torch::kFloat);
    auto keep = scores > 0.5;
    scores = torch::masked_select(scores, keep);
    boxes = torch::masked_select(boxes, keep.unsqueeze(1)).view({-1, 4});

    //Slice the top-K
    std::tuple<torch::Tensor, torch::Tensor> sorted = scores.sort(0, true);
    scores = std::get<0>(sorted).slice(0, 0, top_k);
    torch::Tensor order = std::get<1>(sorted).slice(0, 0, top_k);
    //do NMS
    std::vector<cv::Rect> results;
    keep = nms_cpu(boxes, scores, 0.3);
    auto keep_accesor = keep.accessor<long, 1>();
    auto boxes_accesor = boxes.accessor<float, 2>();
    for (size_t i = 0; i < keep.size(0); i++) {
        int idx = keep_accesor[i];
        cv::Point a(boxes_accesor[idx][0], boxes_accesor[idx][1]);
        cv::Point b(boxes_accesor[idx][2], boxes_accesor[idx][3]);
        cv::Rect item(a, b);
        results.emplace_back(item);
    }
    return results;
}


std::vector<at::Tensor> FaceBoxes::forward_model(std::vector<torch::jit::IValue> &inputs) {
    torch::NoGradGuard no_grad_guard;
    auto out_tensor = Model::script_module.forward(inputs);
    auto result = out_tensor.toTuple()->elements();
    std::vector<at::Tensor> results;
    for (auto &item:result) {
        results.emplace_back(item.toTensor());
    }
    return results;
}

void FaceBoxes::preprocess_(cv::Mat &image) {
//    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    float mean0 = 104;
    float mean1 = 117;
    float mean2 = 123;
    image.convertTo(image, CV_32F);
    image.forEach<cv::Vec3f>
            (
                    [mean0, mean1, mean2](cv::Vec3f &pixel, const int *position) -> void {
                        pixel[0] -= mean0;
                        pixel[1] -= mean1;
                        pixel[2] -= mean2;
                    }
            );

}

torch::Tensor FaceBoxes::intersect(torch::Tensor &box_a, torch::Tensor &box_b) {
    int A = box_a.size(0);
    int B = box_b.size(0);
    auto max_xy = torch::min(
            box_a.slice(1, 2, box_a.size(1)).unsqueeze(1).expand({A, B, 2}),
            box_b.slice(1, 2, box_b.size(1)).unsqueeze(0).expand({A, B, 2})
    );
    auto min_xy = torch::max(
            box_a.slice(1, 0, 2).unsqueeze(1).expand({A, B, 2}),
            box_b.slice(1, 0, 2).unsqueeze(0).expand({A, B, 2})
    );
//      min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
//            box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    auto inter = torch::clamp((max_xy - min_xy), 0);
//    inter = torch.clamp((max_xy - min_xy), min=0)
//    return inter[:, :, 0] * inter[:, :, 1]
    return inter.select(2, 0) * inter.select(2, 1);
}

FaceBoxes::FaceBoxes(const std::string &p_path, bool is_hp) {
    FaceBoxes::is_hp = is_hp;
    if (torch::cuda::is_available()) {
        FaceBoxes::device_type = torch::kCUDA;
    } else {
        FaceBoxes::device_type = torch::kCPU;
    }
    std::cout<<"Loading ... "<<p_path<<std::endl;
    assert(model_exist(p_path));
    Model::script_module = torch::jit::load(p_path);
    if(Model::script_module.i)

    assert(model_exist("/usr/models/anchors.pt"));
    Model::anchor_module = torch::jit::load("/usr/models/anchors.pt");

    std::vector<torch::jit::IValue> inputs;
    auto dummy = torch::ones({1, 1});
    inputs.emplace_back(dummy);
    auto out = anchor_module.forward(inputs);
    FaceBoxes::priors = out.toTensor().cuda();
}
