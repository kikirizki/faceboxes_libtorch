cmake_minimum_required(VERSION 3.5)
project(faceboxes)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${OpenCV_LIBS})
if (NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif (NOT Torch_FOUND)
message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
include(CPack)

file(GLOB_RECURSE lib
        "${PROJECT_SOURCE_DIR}/lib/image/*"
        "${PROJECT_SOURCE_DIR}/lib/tensor_operation/*"
        )

file(GLOB_RECURSE model
        "${PROJECT_SOURCE_DIR}/model/*.h"
        "${PROJECT_SOURCE_DIR}/model/*.cpp"
        )

file(GLOB_RECURSE faceboxes
        "${PROJECT_SOURCE_DIR}/faceboxes/*.h"
        "${PROJECT_SOURCE_DIR}/faceboxes/*.cpp"
        )
file(GLOB_RECURSE priorbox
        "${PROJECT_SOURCE_DIR}/priorbox/*.h"
        "${PROJECT_SOURCE_DIR}/priorbox/*.cpp"
        )
add_executable(faceboxes main.cpp ${lib} ${model} ${priorbox} ${faceboxes}  )
target_link_libraries(faceboxes ${OpenCV_LIBS} ${TORCH_LIBRARIES})