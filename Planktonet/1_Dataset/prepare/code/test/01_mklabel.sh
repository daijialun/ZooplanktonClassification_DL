#!/bin/sh

DATA_DIR=/home/wubin/Workspace/project/zooplankton/CaffeTrian/01_prepare/data/test
LABEL_DIR=/home/wubin/Workspace/project/zooplankton/CaffeTrian/01_prepare/label/test
label=0
for i in `ls $DATA_DIR`; do
	echo $i >> $LABEL_DIR/synsets.txt;
	for j in `find $DATA_DIR/$i -type f -name *.jpg | sort`; do
#		echo $j;
		#获取文件名
		str=${j##*/};
#		echo $str;
		newstr=${i}"/"${str}" "${label};
		echo $newstr >> $LABEL_DIR/train.txt;
	done
	label=$(($label+1))	
done
