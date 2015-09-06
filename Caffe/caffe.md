# Caffe解读

- blob中的信息是caffe用来存储、关联与操作的。对于整个框架来说，blob是标准排列和统一内存接口的。
    
    （基础的数据结构，保存学习到的参数以及网络传输过程中产生数据的类）

- layers是在blob后，作为模型和计算的基础

    （网络的基本单元，由此派生了各种层类。修改这部分的人主要是**特征表达**。layers{ type:Convolution, ReLU, pooling}等）
    
        layer {
        name: "conv2/3x3_reduce"
        type: "Convolution"
        bottom: "pool1/norm1"
        top: "conv2/3x3_reduce"
        param {
        lr_mult: 1.0
        decay_mult: 1.0
        }
        param {
        lr_mult: 2.0
        decay_mult: 0.0
        }
        convolution_param {
        num_output: 64
        kernel_size: 1
        weight_filler {
        type: "xavier"
        std: 0.1
        }
        bias_filler {
        type: "constant"
        value: 0.2
        }
        }
        }
- net是各层之间的聚合和链接

    （网络的搭建，**将Layer所派生出的层类layer合成网络**）
    
        **train_val.prototxt:**

        layer {
        name: "data"
        type: "Data"
        top: "data"
        top: "label"
        include {
        phase: TRAIN
        }
        transform_param {
        mirror: true
        crop_size: 227
        mean_value: 240.097579956
        }
        data_param {
        batch_size: 100
        backend: LMDB
        }
        }
        layer {
        name: "relu1"
        type: "ReLU"
        bottom: "conv1"
        top: "conv1"
        }
        layer {
        name: "norm1"
        type: "LRN"
        bottom: "conv1"
        top: "norm1"
        lrn_param {
        local_size: 5
        alpha: 0.0001
        beta: 0.75
        }
        }

    **deploy.prototxt:**
      
        input: "data"
        input_dim: 1
        input_dim: 1
        input_dim: 224
        input_dim: 224
        layer {
        name: "conv1/relu_7x7"
        type: "ReLU"
        bottom: "conv1/7x7_s2"
        top: "conv1/7x7_s2"
        }
        layer {
        name: "pool1/3x3_s2"
        type: "Pooling"
        bottom: "conv1/7x7_s2"
        top: "pool1/3x3_s2"
        pooling_param {
        pool: MAX
        kernel_size: 3
        stride: 2
        }
        }
   
- solving是降低建模和优化的配置

    （Net 的求解,修改这部分人主要是**研究 DL 求解方向的**
    
    **solver.prototxt:**
    
        test_iter: 24
        test_interval: 71
        base_lr: 0.01
        display: 8
        max_iter: 2130
        lr_policy: "step"
        gamma: 0.1
        momentum: 0.9
        weight_decay: 0.0005
        stepsize: 703
        snapshot: 71
        snapshot_prefix: "snapshot"
        solver_mode: GPU
        net: "train_val.prototxt"
        solver_type: SGD