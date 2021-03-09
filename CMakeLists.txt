cmake_minimum_required(VERSION 3.4)

project(dynmi-dnn)

include_directories(include/dynmi-dnn)

set(TAR_OBJS
    src/train.c
    src/inference.c
    src/data.c)

set(LAYER_OBJS
    src/activation_layer.c
    src/batchnorm_layer.c
    src/convolution_layer.c
    src/dropout_layer.c
    src/maxpooling_layer.c
    src/fc_layer.c)

add_executable(alexnet
    ${TAR_OBJS}
    ${LAYER_OBJS}
    src/matrix.c
    src/alexnet.c)

target_link_libraries(alexnet pthread m)