#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

OUT_PATH=/home/wubin/Workspace/project/zooplankton/CaffeTrian/01_prepare/lmdb
DATA=/home/wubin/Workspace/project/zooplankton/CaffeTrian/01_prepare/label/test
TOOLS=/home/wubin/Workspace/project/caffe/caffe-master_150708/build/tools

VAL_DATA_ROOT=/home/wubin/Workspace/project/zooplankton/CaffeTrian/01_prepare/data/test/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
# gray image
GRAY=true

RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=32
  RESIZE_WIDTH=32
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi


if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating val lmdb..."


GLOG_logtostderr=1 
$TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/train.txt \
    $OUT_PATH/ZooPlankton_val_lmdb


echo "Done."
