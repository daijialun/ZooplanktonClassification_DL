# ZooplanktoNet Model #

ZooplanktoNet网络结构框架，以下内容对其进行解释


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

