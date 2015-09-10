#!/bin/sh

#DATA_DIR and LABEL_DIR MAYBE THE SAME DIRECTORY
DATA_DIR=<dataset root directory>/train
LABEL_DIR=<dataset root directory>/label

#这里所记录的路径为相对路径，必要时使用绝对路径
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
