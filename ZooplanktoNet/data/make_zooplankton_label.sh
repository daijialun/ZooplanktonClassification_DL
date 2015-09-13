#!/bin/sh

#DATA_DIR and VAL_DIR ARE DIFFERENT DIRECTORY
#YOU CAN JUST CHOOSE TRAIN_DIR WITHOUT VAL_DIR
TRAIN_DIR=<dataset root directory>/train
VAL_DIR=<dataset root directory>/val

#这里所记录的路径为相对路径，必要时使用绝对路径
label=0
for i in `ls $TRAIN_DIR`; do
	echo $i >> $TRAIN_DIR/synsets.txt;
	for j in `find $TRAIN_DIR/$i -type f -name *.jpg | sort`; do
		str_train=${j##*/};
		newstr_train=${i}"/"${str}" "${label};
		echo $newstr >> $TRAIN_DIR/train.txt;
	done
	
	for k in `find $VAL_DIR/$I -type f -name *.jpg | sort`; do
		str_val=${k##*/};
		newstr_val=${i}"/"${name}" "${label};
		echo $newname >> $VAL_DIR/val.txt;
	done
	label=$(($label+1))	
done
