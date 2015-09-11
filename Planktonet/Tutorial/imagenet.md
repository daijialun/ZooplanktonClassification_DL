# ImageNet 相关介绍

这里的内容主要是对caffe中[ImageNet exmaple](https://github.com/BVLC/caffe/tree/master/examples/imagenet)进行参考和分析

ImageNet example的说明是让你可以在自己的模型上训练自己的数据

## Data Preparation

所有指令都是在CAFFE_ROOT目录下完成的。

1. 需要下载好ImageNet的训练数据和验证数据，存储的路径例如：

        /path/to/imagenet/train/n01440764/n01440764_10026.JPEG
        /path/to/imagenet/val/ILSVRC2012_val_00000001.JPEG
        
    /path/to/表示自己所设置的文件目录路径
    
2. 准备训练的辅助数据
    
        ./data/ilsvrc12/get_ilsvrc_aux.sh
       
    将会下下载`train.txt`，`val.txt`,`test.txt`，`imagent_mean.binaryproto`，`det_synset_words.txt`，`synsets.txt`与`synset_words.txt`。
     
     训练和验证的输入都在`train.txt`与`val.txt`内，其中列出了所有文件及其label。`synset_words.txt`表明synset与name
的匹配关系。
    
    一般情况下，我们不用预先将图片尺寸调整为256x256，但是如果想要调整，可以通过命令行实现：
    
        for name in /paht/to/imagenet/val/*.JPEG; do
                convert -resize 256x256\! $name $name
        done
        
    