#!/usr/bin/env sh

./build/tools/caffe test \
    -model models/cvbiouc_zooplanktonet/solver.prototxt \
    -weights model/cvbiouc_zooplanktonet/zooplanktonet_train_iter_N.caffemodel \
    -gpu all 
