# Caffe解读

## Caffe代码层次

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
        
## Layers(对象)具体分析 


### *Vision Layers*

#### Convolution

- Layer type: Convolution

- CPU implementation: convolution_layer.cpp

- CUDA GPU implementation: convolution_layer.cu
 
#### Pooling

- 池化

 - Layer type: Pooling

- CPU implementation: pooling_layer.cpp

- CUDA GPU implementation: pooling_layer.cu

#### Local Response Normalization ( LRN )
 
- 局部响应归一化

- Layer type: LRN

- CPU implementation: lrn_layer.cpp

- CUDA GPU implementation: lrn_layer.cu

- Parameters:

    - norm_region: 选择对相邻通道间（ACROSS_CHANNELS）归一化还是通道内空间区域（WITHIN_CHANNEL）归一化，默认为ACROSS_CHANNELS
    
     - local_size: 对ACROSS表示需要求和的通道间数量；对WITHIN表示需要求和的区间区域的边长

    - 局部响应归一化完成“临近抑制”操作，对局部输入区域进行归一化


### *Loss Layers*

#### Softmax

- Layer type: SoftmaxWithLoss

- 计算输入中，softmax的多项式逻辑损失，在数学上，提供了更稳定的梯度

#### Sum-of-Squares / Euclidean

- Layer type: EuclideanLoss

#### Hinge / Margin

- Layer type: HingeLoss

- CPU implementation: hinge_loss_layer.cpp

#### Sigmoid Cross-Entropy

#### Infogain

#### Accuracy and Top-k

Accuracy对输出所对应的目标的准确度，通过分数表达。Accuracy实际上不是Loss，没有反向步骤。

### *Activation / Neuron Layers*

通常，Activation / Neuron Layers是元素级操作，将底层blob生成同规模的顶层blob

#### ReLU / Rectified-Linear and Leaky-ReLU（常用）

- Layer type: ReLU

- CPU implementation: relu_layer.cpp

- CUDA GPU implementation: relu_layer.cu

#### Sigmoid

- Layer type: sigmoid

- CPU implementation: sigmoid_layer.cpp

- CUDA GPU implementation: sigmoid_layer.cu

#### TanH / Hyperbolic Tangent

- Layer type: TanH

- CPU implementation: tanh_layer.cpp

- CUDA GPU implementation: tanh_layer.cu

#### Absolute Value
#### Power
#### BNLL


### *Data Layers*

数据通过data layers进入Caffe，data layers位于网络底层。数据可从内存中直接获取有效的数据库格式（LevelDB或LMDB）。如果效率不是主要因素时，可从硬盘中读取HDF5格式或者普通图片格式。

通常的输入预处理（mean subtraction, scaling, random cropping , mirroring）可通过*TransformationParameters*来使用。

#### Database

- Layer type: Data

- Parameters:

    - backend: 选择LEVELDB或者LMDB
    
#### In-Memory
#### HDF5 Input
#### ImageData
#### Windows
#### Dummy


### *Common Layers*

#### Inner Product

- Layer type: TanH

- CPU implementation: inner_prouct_layer.cpp

- CUDA GPU implementation:  inner_prouct_layer.cu

#### Splitting
#### Flattening
#### Reshape
#### Concatenation
#### Slicing
#### Elementwise Operations
#### Argmax
#### Softmax
#### Mean-Variance Normalization

## 其他说明

### Caffe源码

[**Caffe源码阅读路线：**](http://blog.csdn.net/kkk584520/article/details/41681085)

1. 从src/caffe/proto/caffe.proto开始，了解各类数据结构，主要是内存对象和序列化磁盘文件的一一队对应关系，了解如何从磁盘Load一个对象到内存，以及如何将内存对象Save到磁盘，中间的过程实现都是由protobuf自动完成的

2. 看头文件，不急于看cpp文件，理解整个框架。Caffe中类数目众多，但是条理清晰。在Testing时，最外层的类是Caffe::Net，包含了多个Caffe::Layer对象，Layer对象派生出神经网络多种不同层的类(DataLayer, ConvolutionLayer, InnerProductionLayer, AccurancyLayer)，每层都有相应的输入输出（Blob对象）以及层的参数（Blob对象）；Blob中包括了SyncedMemory对象，统一了CPU和GPU存储器。自顶向下去看这些类，结合理论知识很容易掌握使用方法。

3. 针对性地取看cpp和cu文件。一般而言，Caffe框架不需要修改，只需要增加新的层实现即可。如果想实现自己的卷积层，从ConvolutionLayer派生一个新类MyConvolutionLayer，然后将几个虚函数改成自己的实现即可。这一阶段主要关注点在算法，而不是源码本身。

4. 可编写各类工具，集成到Caffe内部。在tools/中有很多实用工具，可以根据需要修改。例如，从训练好的模型中抽取参数进行可视化，可用Python结合matplot实现

5. 最后，想更深层次学习，自己重新编写Caffe，重新构建自己的框架

### 其他
- prototxt (protocol buffer definition file)
    
    protoculbuffe是google的一种数据交换格式，独立于语言和平台，为一种二进制文件，比使用xml进行数据交换快速，可用于网络传输、配置文件、数据存储等诸多领域
    
    prototxt主要是记录模型结构。另外，**caffe的layers和其参数主要定义在caffe.proto这protocol buffer definition中。** caffe.proto的路径在./src/caffe/proto
  
- 在卷积层，每一个filter对应输出层的一个feature map