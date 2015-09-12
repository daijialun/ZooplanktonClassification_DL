# ImageNet 相关介绍

这里的内容主要是对caffe中[ImageNet exmaple](https://github.com/BVLC/caffe/tree/master/examples/imagenet)进行参考和分析

ImageNet example的说明是让你可以在自己的模型上训练自己的数据

## Data Preparation

所有指令都是在CAFFE_ROOT目录下完成的。

1. 需要下载好ImageNet的训练数据和验证数据，存储的路径例如：

        /path/to/imagenet/train/n01440764/n01440764_10026.JPEG
        /path/to/imagenet/val/ILSVRC2012_val_00000001.JPEG
        
    /path/to/表示自己所设置的文件目录路径
    
2. 准备训练的辅助数据，即下载`train.txt`与`val.txt`
    
        ./data/ilsvrc12/get_ilsvrc_aux.sh
       
    将会下下载`train.txt`，`val.txt`,`test.txt`，`imagent_mean.binaryproto`，`det_synset_words.txt`，`synsets.txt`与`synset_words.txt`。
     
     训练和验证的输入都在`train.txt`与`val.txt`内，其中列出了所有文件及其label。`synset_words.txt`表明synset与name
的匹配关系。
    
    一般情况下，我们不用预先将图片尺寸调整为256x256，但是如果想要调整，可以通过命令行实现：
    
        for name in /paht/to/imagenet/val/*.JPEG; do
                convert -resize 256x256\! $name $name
        done
        
3. 将zooplankton数据集转换为lmdb格式

        ./example/imagenet/create_imagenet.sh
        
      设置`RESIZE=true`，调整图片尺寸；设置`GRAY=true`，选择灰度图像。
      
      创建好的imagenet的lmdb在`examples/imagenet/ilsvrc12_train_lmdb`和`examples/imagenet/ilsvrc12_val_lmdb`
      
4. 模型需要从每张图像减去均值图像，因此需要计算mean

        ./examples/imagenet/make_imagenet_mean.sh
        
     其结果将在`data/ilsvrc12/imagenet_mean.binaryproto`中。
     
     
## Model Definition

- 模型定义位于`models/bvlc_reference_caffenet/train_val.prototxt`，如果不使用文件中默认的文件路径，则修改`.prototxt`中相关路径。

- 在`models/bvlc_reference_caffenet/train_val.prototxt`中，有部分`include`会被指定为`phase: TRAIN`或`phase: TEST`。这部分允许我们在一个文件中，定义两个很相关的网络：一个网络用来训练，以及一个网络用来测试。这两个网络几乎相同，共享除了标记了`include{phase: TRAIN}`或`include{phase: TEST}` 的所有层。在这个例子中，只有输入层和输出层是不同的。

- **Input layer differences:** *训练网络*的输入数据取自`examples/ilsvrc12_train_leveldb`，并且随机mirror输入图像； *测试网络*的输入数据取自`examples/ilsvrc12_val_leveldb`，不随机mirroring。

- **Output layer differences:** 二者网络都输出`softmax_loss`层，该层在*训练*中用来计算loss function以及初始化反向传播。在*验证*中只报告loss。*测试*网络中，有第二层输出`accuracy`，用来报告测试集的准确率。
在*训练*过程中，偶尔会通过测试集来展示测试网络，例如`Test score #0: XXX`和`Test score #1: XXX`。其中，`score 0`表示准确率，`score 1`表示loss

- `models/bvlc_reference_caffenet/solver.prototxt`包含以下内容：

    - 256 batches，总共450,000次迭代（大约90 epochs）
    - 每1000此迭代，用验证集测试所学习的网络
    - 设置初始学习略为0.01，每100,000此迭代减少一次（大约20 epochs）
    - 每20此迭代展示一次信息
    - momentum为0.9和weight decay为0.0005
    - 每10,000此迭代，对当前state进行一次snapshot
    
 
## Train ImageNet

1. 开始训练网络

        ./build/tools/caffe train --solver=models/bvlc_reference_caffenet/solver.prototxt
        
2. 如果想分析训练时间（可选）

        ./build/tools/caffe time --model=models/bvlc_reference_caffenet/train_val.prototxt   
        
3. 继续训练，选择相应snapshots即可

        ./build/tools/caffe train --solver=models/bvlc_reference_caffenet/solver.prototxt  --snapshot=models/bvlc_reference_caffenet/caffenet_train_iter_10000.solverstate
        
    