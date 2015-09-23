#!/usr/bin/env sh

./build/tools/caffe train \
    -solver models/cvbiouc_zooplanktonet/solver.prototxt \
    -snapshot models/cvbiouc_zooplanktonet/zooplankonet_train_iter_N.solverstate \
    -gpu all
