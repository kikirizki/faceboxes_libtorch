#include <iostream>
#include <opencv2/opencv.hpp>
#include "faceboxes/faceboxes.h"

int main() {
    FaceBoxes faceboxes("/usr/models/face.pt", "/usr/models/anchor.pt");
    auto input_image = cv::imread("sample.jpg");
    auto ori = cv::imread("sample.jpg");
    cv::resize(input_image, input_image, cv::Size(1280, 720));
    faceboxes.detect_on_image(input_image);
    auto results = faceboxes.detect_on_image(input_image);

    for (auto &rect : results) {
        if (rect.x > 0 and rect.y > 0 and rect.width > 0 and rect.height > 0 and
            rect.x + rect.width <= input_image.cols and rect.y + rect.height <= input_image.rows) {
            cv::rectangle(ori, rect, cv::Scalar(8, 218, 254), 2);

        }
    }
    cv::imwrite("result.jpg", input_image);
    return 0;
}
