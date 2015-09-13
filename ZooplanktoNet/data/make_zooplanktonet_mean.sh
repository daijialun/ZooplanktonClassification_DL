#!/usr/bin/env sh
# Compute the mean image from the zooplanktonet training lmdb
# N.B. this is available in data/zooplankton

EXAMPLE=examples/zooplanktonet
DATA=data/zooplankton
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/zooplanktonet_train_lmdb \
  $DATA/zooplanktonet_mean.binaryproto

echo "Done."
