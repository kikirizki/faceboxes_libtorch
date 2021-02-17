# Faceboxes libtorch
This is unofficial implementation (inference only) of [FaceBoxes](https://arxiv.org/abs/1708.05234): A CPU Real-time Face Detector with High Accuracy, in libtorch (pythorch C++ API). The original caffe implementation can be found [here](https://github.com/sfzhang15/FaceBoxes)

## Pretrained Model

Move **anchor.pt** and **face.pt** from this repo to `/usr/models/`

## Demo
  -  Download libtorch from pytroch.org extract and put it at `/usr/lib/`
  - `cd faceboxes_libtorch/ && mkdir build`
  - `cd build && cmake -DCMAKE_PREFIX_PATH=/usr/lib/libtorch ..`
  - `make -j$(nproc)`
  - put sample.jpg in build folder
  - run `./faceboxes` result will be saved in `build/result.jpg`
## Info
Tested on **Ubuntu 20.04**, **libtorch 1.7.1** the weight is traced from the weight published here https://github.com/zisianw/FaceBoxes.PyTorch. So the performance expected to be the same

## Todo
- [ ] Deploy on android
- [ ] Use half-precision model
- [x] Libtorch
