# 关于ZooplanktoNet的注意事项

caffe中例子主要位于data, example和models，因此使用过程中，主要参考这三个目录下的文件。

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

## Model

### solver.prototxt

- 文件根据"caffe_root/models/bvlc_reference_caffenet/solver.prototxt"修改

- 注意修改**net**与**snapshot_prefix**路径设置 

### train_val.prototxt

- 文件根据"caffe_root/models/bvlc_reference_caffenet/bvlc_caffenet.model"修改

- 修改**name**为**name: ZooplanktoNet**

- 将**tranform_param**参数中，修改**mean_file**路径 

- 修改**data_param**中，**source**路径

### deploy.prototxt

- 结构部署文件，大部分内容与train_val.txt相同，修改其没有实际影响


## Train

### train_zooplanktonet.sh

- 脚本根据"caffe_root/examples/imagenet/train_caffenet.sh"修改

- "--solver=models/cvbiouc_zooplanktonet/solver.prototxt"可修改为"-solver models/cvbiouc_zooplanktonet/solver.prototxt"

- 可添加"-gpu all"，表示使用全部GPU运算

- 注意修改solver.prototxt路径


## Finetune

### fine_zooplanktonet.sh

- -solver表示使用的solver.prototxt文件位置；-weights表示finetune所使用的pre-trained模型位置

### solver.prototxt

- 文件根据"caffe_root/models/finetune_flickr_style/solver.prototxt"修改

- 注意修改**net**与**snapshot_prefix**路径设置

### train_val.prototxt

- 文件根据"caffe_root/models/finetune_flickr_style/train_val.prototxt"修改

- 修改**name**为**name: ZooplanktonCaffeNet**

- 将**tranform_param**参数中，修改**mean_file**路径

- 将**layer**中的**"type:ImageData"**修改为**"type:Data"**，表示不使用图像数据，使用lmdb

- 将**image_data_param**的参数设置，修改为**data_param**；其中修改**source**，删除**new_height**和**new_width**，增加**backend: LMDB**

- 在最后几层，根据自己的实际情况进行修改。注意修改后几层的**name**，**bottom**与**top**


## Other

- 训练数据不够，使用finetune

- 如果训练集收敛不好，可尝试增大stepsize

- 初始learning rate, lr=0.01

- 调试步骤可尝试：逐步调试batchsize, base_lr, stepsize, 以及每一层lr，主要依靠调试经验

- 看caffe代码经验：先搞懂原理，再看代码；清楚数据流，架构模块拆分并不困难

- deplpy.prototxt主要是matlab使用的格式

- --shuffling，表示随机化处理

- NAN表示Not a Number，表示一些特殊数值（无穷与非数值），即可能已经无解了

- weight decay是一个正则项，参考[blog](http://blog.csdn.net/zouxy09/article/details/24971995)

- 可用graphviz画网络结构图

- split_layer的作用是将一个blob复制两份或多份，保持各自的data独立和diff的共享。例如，一个blob作为layer的bottom的时候，就需要通过split将这个blob复制成两份，接收两个回传的梯度并相加 

















