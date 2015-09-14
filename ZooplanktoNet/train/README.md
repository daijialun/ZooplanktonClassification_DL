# ZooplanktoNet Train #

ZooplanktoNet训练命令，以下内容对其进行解释


## Train

### train_zooplanktonet.sh

- 脚本根据"caffe_root/examples/imagenet/train_caffenet.sh"修改

- "--solver=models/cvbiouc_zooplanktonet/solver.prototxt"可修改为"-solver models/cvbiouc_zooplanktonet/solver.prototxt"

- 可添加"-gpu all"，表示使用全部GPU运算

- 注意修改solver.prototxt路径
