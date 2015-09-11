#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLE=examples/zooplanktonet
DATA=data/zooplankton
DBTYPE=lmdb

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/zooplankton_train_$DBTYPE $EXAMPLE/zooplankton_test_$DBTYPE

./build/examples/cifar10/convert_cifar_data.bin $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."

./build/tools/compute_image_mean -backend=$DBTYPE \
  $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto

echo "Done."
