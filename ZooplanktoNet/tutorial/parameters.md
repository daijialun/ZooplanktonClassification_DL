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

caffenet具体结构可参考其![ ](http://s1.sinaimg.cn/large/004j58Jbzy6NvjDo5UI60&690)

### layer: data

- name: "data"  

    type:"Data"

    该层为数据层，代号为data
    
- top:"data"

    top:"label"
    
    该层有两个输出，分别为"data"与