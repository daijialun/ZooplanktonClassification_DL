# ZooplanktoNet Data #

ZooplanktoNet准备数据的相关文件，以下内容对其进行解释

## Data

### make_zooplankton_label.sh

- 根据数据集的train与val，得到相对应的train.txt与val.txt

- 在**train**过程中，train.txt / val.txt使用的为图片相对路径

- 在**finetune**过程中，train.txt / val.txt可能需要使用图片的绝对路径。


### create_zooplanktonet.sh

- 脚本根据" caffe_root/examples/imagenet/create_imagenet.sh"修改

- 如果图片没有经过尺寸变换，则设置**RESIZE=true**，且通过**RESIZE_HEIGHT**和**RESIZE_WIDTH**的数值来调整大小。

- 在**$TOOLS/convert_imageset**的参数中，则生成的lmdb格式，默认通道数为3。在**train**过程中，blob对应的储存格式为N x K x H x W，生成的top的为50 x 3 x 227 x 227，其中3对应通道数为3，则说明网络训练的数量为3；如果需要转换灰度图像为lmdb格式，则在参数中设置**--gray=true**，则lmdb为单通道，在**train**过程中中，top为50 x 1 x 227 x 227。

- 在脚本中，如果要修改**$TOOLS/convert_imageset**的参数，直接删除，**不要使用注释**，由于其语句运行程序，注释将作为输入运行，则会显示错误提示。

- **$DATA/train.txt**，请根据实际情况进行修改，例如修改为**$DATA/train/train.txt**。否则在转换过程中，显示" transform 0 images "，则表示转换出错。

- 注意修改各参数变量名，如“$EXAMPLE/imagenet_val_lmbd”修改为“$EXAMPLE/zooplanktonet_val_lmdb”

### make_zooplanktonet_mean.sh

- 脚本根据"caffe_root/examples/imagenet/make_imagenet_mean.sh"修改

- 如果对应lmdb数据的通道为3，则生成的mean.binaryproto通道也为3；同理，lmdb通道为1，生成的mean.binaryproto通道也为1

- 注意修改各参数变量名，如"$DATA/imagenet_mean.binaryproto"修改为"$DATA/zooplanktonet_mean.binaryproto"
