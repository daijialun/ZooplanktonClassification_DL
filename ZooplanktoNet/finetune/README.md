# ZooplanktonCaffeNet Finetune #

ZooplanktonCaffeNet网络结构与实现，以下内容对其进行解释


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