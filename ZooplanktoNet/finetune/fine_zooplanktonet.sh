#!/usr/bin/env sh

./build/tools/caffe train \
    -solver models/cvbiouc_zooplanktonet/solver.prototxt \
    -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
    -gpu all
