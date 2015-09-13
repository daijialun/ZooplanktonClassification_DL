#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/cvbiouc_zooplanktonet/solver.prototxt
