//
// Created by Kiki Rizki Arpiandi on 13/11/18.
//
#include "../priorbox/priorbox.h"


torch::Tensor PriorBox::forward() {
    int k = 0;
    std::vector<float> anchors_data;
    for (auto &feature_map:PriorBox::feature_map_list) {
        std::vector<float> min_sizes = PriorBox::min_sizes_list.at(k);
        float scale_width ;
        if(k%2==0){
            scale_width = 5.0;
        }else{
            scale_width = 4.5;
        }
        for (int i = 0; i < feature_map.first; i++) {
            for (int j = 0; j < feature_map.second; j++) {
                std::cout << "(i,j) " << i << " " << j << std::endl;
                for (auto min_size:min_sizes) {
                    float s_kx = min_size / PriorBox::image_size[1];
                    float s_ky = min_size / PriorBox::image_size[0];
                    s_kx = s_kx * 4;

                    if (min_size == 32) {
                        float scalex = PriorBox::steps[k] / PriorBox::image_size[1];
                        float scaley = PriorBox::steps[k] / PriorBox::image_size[0];
                        std::vector<float> dense_cx = {
                                scalex * j,
                                static_cast<float>(scalex * (j + .25)),
                                static_cast<float>(scalex * (j + .5)),
                                static_cast<float>(scalex * (j + .75))
                        };
                        std::vector<float> dense_cy = {
                                scaley * i,
                                static_cast<float>(scaley * (i + .25)),
                                static_cast<float>(scaley * (i + .5)),
                                static_cast<float>(scaley * (i + .75))
                        };
                        for (auto cy:dense_cy) {
                            for (auto cx:dense_cx) {
                                anchors_data.emplace_back(cx);
                                anchors_data.emplace_back(cy);
                                anchors_data.emplace_back(s_kx);
                                anchors_data.emplace_back(s_ky);
                            }
                        }


                    } else if (min_size == 64) {
                        float scalex = PriorBox::steps[k] / PriorBox::image_size[1];
                        float scaley = PriorBox::steps[k] / PriorBox::image_size[0];
                        std::vector<float> dense_cx = {
                                scalex * j,
                                static_cast<float>(scalex * (j + .5))
                        };
                        std::vector<float> dense_cy = {
                                scaley * i,
                                static_cast<float>(scaley * (i + .5))
                        };
                        for (auto cy:dense_cy) {
                            for (auto cx:dense_cx) {
                                anchors_data.emplace_back(cx);
                                anchors_data.emplace_back(cy);
                                anchors_data.emplace_back(s_kx);
                                anchors_data.emplace_back(s_ky);
                            }
                        }


                    } else {
                        float cx = (j + 0.5) * PriorBox::steps.at(k) / PriorBox::image_size[1];
                        float cy = (i + 0.5) * PriorBox::steps.at(k) / PriorBox::image_size[0];

                        anchors_data.emplace_back(cx);
                        anchors_data.emplace_back(cy);
                        anchors_data.emplace_back(s_kx);
                        anchors_data.emplace_back(s_ky);
                    }
                }
            }
        }


        k++;
    }
    float sum = 0;
    for (auto &f:anchors_data) {
        sum += f;
    }
    std::cout << sum << std::endl;
    int a = anchors_data.size();
    torch::Tensor output = torch::empty({a});
    float *data = output.data<float>();
    for (auto &f:anchors_data) {
        *data++ = f;
    }
    output = output.view({-1,4});
    return output;
}

PriorBox::PriorBox(float im_height, float im_width) {
    PriorBox::image_size[0] = im_height;
    PriorBox::image_size[1] = im_width;
    std::cout << "Feature map" << std::endl;
    for (auto &step:PriorBox::steps) {
        float x = ceil(PriorBox::image_size[0] / step);
        float y = ceil(PriorBox::image_size[1] / step);
        std::pair<float, float> feature_map;
        feature_map.first = x;
        feature_map.second = y;
        PriorBox::feature_map_list.emplace_back(feature_map);
        std::cout << " [ " << feature_map.first << " , " << feature_map.second << " ] " << std::endl;
    }
}

PriorBox::PriorBox() {}



