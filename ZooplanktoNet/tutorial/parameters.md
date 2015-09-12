# Caffe网络配置参数说明

## models/bvlc_reference_caffenet/solver.prototxt

- net: "models/bvlc_reference_caffenet/train_val.prototxt"

    网络结构配置文件路径
    
- test_iter: 1000

    测试迭代次数为

- test_interval: 1000

    每1000此迭代，用验证集测试网络accuracy与loss
    
- base_lr: 0.01

    初始学习率为0.01
    
- lr_policy: "step"

- gamma: 0.1

- stepsize: 100,000

    每100,000次迭代改变一次learning rate
    
- display: 20

    每20次迭代显示一次训练信息
    
- max_iter: 450,000

    最大迭代次数为450,000
    
- momoentum: 0.9

- weight_decay: 0.0005

- snapshot: 10000

    每10,000次迭代进行一次state snapshot
    
- snapshot_prefix: "models/bvlc_reference_caffenet/caffenet_train"
 
    snapshot路径前缀
    
- solver_mode: GPU

    GPU模式运行
    

## models/bvlc_reference_caffenet/train_val.prototxt

caffenet具体结构可参考[AlexNet之结构篇](http://blog.sina.com.cn/s/blog_eb3aea990102v47i.html)与下图：

![ ](http://s1.sinaimg.cn/large/004j58Jbzy6NvjDo5UI60&690)

### layer: data

- name: "data"  

    type:"Data"

    该层为数据层，代号为data
    
- top:"data"

    top:"label"
    
    该层有两个输出，分别为"data"与"label"
    
- inclde{ phase: TRAIN }

    这一层标记为TRAIN，只在训练中使用，不参与测试
    
- tranform_param{

    mirror: true
    
     crop_size:227
     
     mean_file: "data/ilsvrc12/imagenet_mean.binaryproto"
     
     }
     
     *mirror*,*crop_size*???
     
     mean_file 均值文件的位置，可用具体mean pixel/channel-wise值代替mean image，例如mean_value:104
     
- data_param{

    source: "examples/imagenet/ilsvrc12_train_lmdb"
    
    batch_size: 256
    
    backend: LMDB
    
    }
    
    source表示数据来源，*batch_size表示每次处理批量为256*??? backend表示数据格式为LMDB
    
### layer: conv1

- name: "conv1"

    type: "Convolution"
    
    该层为卷积层，代号为conv1
    
- bottom: "data"

    top: "conv1"
    
    该层的输入为"data"，输出为"conv1"
    
-  param{

    lr_mult: 1
    
    decay_mult: 1

    }
    
    param{

    lr_mult: 2
    
    decay_mult: 0
    
    }
    
    *lr_mult*,*decay_mult*？？？
    
- convolution_param{

    num_output: 96
    
     kernel_size: 11
     
     stride: 4
     
     weight_fliter{
     
     type: "gaussion"
     
     std: 0.01
     
     }
     
     num_output表示卷积数量为96，kernel_size表示卷积尺寸为11x11，stride表示卷积中心间距为4，*weight_fliter*？？？
     
### layer: relu1

- name: "relu1"

    type: "ReLU"
    
    该层为ReLU层，代号为relu1
    
- bottom: "conv1"

    top: "conv1"
    
    输入数据为"conv1"，输出数据仍为"conv1"
    
### layer: pool1

- name: "pool1"

    type: "Pooling"
    
    该层为Pooling层，代号为pool1
    
- bottom: "conv1"

    top: "pool1"
    
    输入数据为"conv1"，输出数据为"pool1"
    
- pooling_param{

    pool: MAX
    
    kernel_size: 3
    
    stride: 2
    
    }
    
   pool的形式为MAX，kernel_size表示尺寸为3x3，stride表示每个pool的间距为2      
   
### layer: norm1

- name: "norm1"

    type: "LRN"
    
    该层为Local Response Normalization层，代号为norm1
    
- bottom: "pool1"

    top: "norm1"
    
    输入数据为pool1，输出数据为norm1
    
- lrn_param{

    local_size: 5
    
    alpha: 0.0001
    
    beta: 0.75
    
    }
    
    *local_size*？？？alpha，beta表示公式参数的设置
    
### layer: fc6

- name: "fc6"

    type: "InnerProduct"
    
    该层为全连接层，代号为fc6
   
- bottom: "pool5"

    top: "fc6"
    
    数据输入为"pool5"，输入输出为"fc6"
    
- param{

    lr_mult: 1
    
    decay_mult: 1
    
    }
    
    *lr_mult*，*decay_mult*？？？
    
- inner_product_param{

    num_output: 4096
    
    }
    
    num_output表示全连接层的参数个数为4096
    
- weight_fliter{
    
    type: "gaussian"
    
    std: 0.005
    
    }
    
    *weight_fliter*？？？ type: "gaussian"表示类型为高斯，std表示标准差为0.005
    
- bias_filler{

    type: "constant"
    
    value: 1
    
    }
    
    *bias_filler*？？？
    
### layer: drop6

- name: "drop6"

    type: "Dropout"
    
    该层为Drop层，代号为drop6
    
- bottom: "fc6"

    top: "fc6"
    
    输入数据为fc6，输出数据仍为fc6
    
- dropout_param{

    dropout_ratio: 0.5
    
    }
    
    dropout_ratio表示以0.5的概率丢弃神经元
    
### layer: accuracy

- name: "accuracy"

    type: "Accuracy"
    
    该层为Accuracy层，代号为accuracy
    
- bottom: "fc8"
    
   bottom: "label"
   
   top: "accuracy"
   
   输入数据为fc8（即预测结果）与label，输出为accuracy
   
- include{

    phase: TEST
 
    }
    
    该部分模块用于测试网络中
    
### layer: loss

- name: "loss"

    type: "SoftmaxWithLoss"
    
    该层为softmaxWithLoss层，代号为loss
    
- bottom: "fc8"

    bottom: "label"
    
    top: "loss"
    
    输入输入数据为fc8（即预测结果）与label，输出为loss
