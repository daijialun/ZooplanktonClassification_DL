#!/usr/bin/env sh

TOOLS=/home/wubin/Workspace/project/caffe/caffe-master_150708/build/tools

$TOOLS/caffe train \
    --solver=/home/wubin/Workspace/project/zooplankton/CaffeTrian/02_train/net_config/ZooPlankton_solver.prototxt

