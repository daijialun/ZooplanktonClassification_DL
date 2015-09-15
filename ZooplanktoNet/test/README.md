# ZooplanktoNet Test #

ZooplanktoNet训练命令，以下内容对其进行解释


## Train

### test_zooplanktonet.sh

- 脚本参考[ Caffe代码导读（5）：对数据集进行Testing](http://blog.csdn.net/kkk584520/article/details/41694301)

- "-model models/cvbiouc_zooplanktonet/train_val.prototxt"表示模型结构配置

- "-weights models/cvbiouc_zoolanktonet/zooplanktonet_train_iter_N.caffemodel"表示已训练好的模型，测试其准确率

- "-gpu all"，表示使用全部GPU运算

- 注意修改train_val.prototxt与caffemodel路径
