#!/usr/bin/env sh
# Create the zooplankton lmdb inputs
# N.B. set the path to the zooplankton train + val data dirs

EXAMPLE=examples/imagenet
DATA=data/zooplankton
TOOLS=build/tools

TRAIN_DATA_ROOT=/path/to/caffe-master/data/zooplankton/train/
VAL_DATA_ROOT=/path/to/caffe-master/data/zooplankton/val/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
# Grayscale is on
GRAY=true
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_zooplanktonet.sh to the path" \
       "where the ZooplanktoNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_zoolanktonet.sh to the path" \
       "where the ZooplanktoNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray=$GRAY
    $TRAIN_DATA_ROOT \
    $DATA/train/train.txt \
    $EXAMPLE/zooplanktonet_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray=$GRAY
    $VAL_DATA_ROOT \
    $DATA/val/val.txt \
    $EXAMPLE/zooplanktonet_val_lmdb

echo "Done."
