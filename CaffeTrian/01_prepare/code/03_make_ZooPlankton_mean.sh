#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/wubin/Workspace/project/zooplankton/CaffeTrian/01_prepare/lmdb
DATA=/home/wubin/Workspace/project/zooplankton/CaffeTrian/01_prepare/image_mean
TOOLS=/home/wubin/Workspace/project/caffe/caffe-master_150708/build/tools

$TOOLS/compute_image_mean $EXAMPLE/ZooPlankton_train_lmdb \
  $DATA/ZooPlankton_mean.binaryproto

echo "Done."
