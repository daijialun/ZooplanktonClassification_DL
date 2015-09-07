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

## 其他说明


- prototxt (protocol buffer definition file)
    
    protoculbuffe是google的一种数据交换格式，独立于语言和平台，为一种二进制文件，比使用xml进行数据交换快速，可用于网络传输、配置文件、数据存储等诸多领域
    
    prototxt主要是记录模型结构。另外，**caffe的layers和其参数主要定义在caffe.proto这protocol buffer definition中。** caffe.proto的路径在./src/caffe/proto
  
- 在卷积层，每一个filter对应输出层的一个feature map

- 通过用对*param*设置同样的名字，可共享其参数。		

		layer:{
		name:'innerproduct1'
		param:'shareweights'
		bottom:'data'
		top:'innerproduct1'}
		layer:{
		name:'innerproduct2'
		param:'shareweights'
		bottom:'data'
		top:'innerproduct2'}
	
### 主要流程

- 将数据转为为caffe-mat，有lmdb, leveldb, hdf5, mat, list of images

- 定义网络Net

- 配置解Solver

- 进行训练求解

		caffe train -solver solver.prototxt -gpu -0
		
- 参考例子

	- examples/mnist,cifar10,imagenet
	
	- examples/*.ipynb
	
	- model/*


### Pre-train和Fine-tune

通过将针对其他问题所训练好的模型，即pre-train模型；进行一点的修改，用来解决我们目前的问题，即fine-tune。这里主要参考了[Caffe主页上的文档](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.gc2fcdcce7_216_408)	

- 将pre-trained模型，通过fine-tunes实现到新的任务中，在模型定义中，修改部分内容

	- 将*data_param{source:"ilsvrc12_train_lmdb"}*改为*data_param{source:"style_train_lmdb"}*
	- 将*name:"fc8"*改为*name:"fc8-style"*；
	- 将*num_output:1000*改为*num_output:20*
	
- 输入模型与模型的解，进行调整


		> caffe train -solver models/finetune_flickr_style/solver.prototxt \
		-weight bvlc_reference_caffenet.caffemodel
		
	在pycaffe中的步骤为：
	
		pretrain_net=caffe.Net("net.prototxt","net.caffemodel")
		solver=caffe.SGDSolver("solver.prototxt")
		solver.net.copy_from(pretrained_net)
		solver.solve()
		
- fine-tuning是将特征转化为特别的可识别性质(style recognition)

	fine-tune适用于：
	
	- 更robust优化与好的初始化
		
	- 需要更少的数据
	
	- 更快的学习
	
- Fine-tuning的技巧(tricks)

	- 从最后一层开始
		- layers有基础的learning rate: param{lr_mult: 1}
		- 为了快速优化，只对最后一层进行操作；通过设置lr_mult=0来确定一个参数，避免出现early divergence
		- 如果效果足够好就停止调整，否则就继续
	
	- 降低学习率
	
		- 以10x或100x降低solver学习率
		- 保持pre-training的初始化，避免出现divergence
		
### 整体结构

caffe的核心代码都在src/caffe下，主要有以下部分：net, layers, blob, solver。参考[Deep learning in Practice](http://blog.csdn.net/abcjennifer/article/details/46424949)

- net.cpp
	
  net定义网络，整个网络中有很多layers，net.cpp负责**计算(computation)**网络在训练过程中的forwad,backward过程，即计算forward/backward过程中各层的参数
	
- layers.cpp

	在src/caffe/layers中的层，在protobuffer中调用时包含各属性（name type, layer-specific parameters） ,其中**.proto文件中定义message类型，.prototxt或binaryproto文件中定义message的值**。定义一个layer需要定义其setup, forward和backward过程
	
- blob.cpp

	net中的数据和求导结果通过4维的blob传递。一个layer有很多blobs
	
- solver.cpp

	结合loss，用gradient更新weights。主要函数：Init(), Solve(), ComputeUpdateValue(), Snapshot()等。三种solver：AdaGradSolver, SGDSolver和NesterovSolver可供选择
	
- Protocol buffer

	protocol buffer在.proto文件中定义message类型，.prototxt或.binaryproto文件中定义message的值
	
	- Caffe所有message定义在src/caffe/protp/caffe.proto中
	- Experimet中主要用protocol_buffer.solver和model，分别定义solver参数（学习率等）和model（网络结构）
	
	- 技巧：冻结一层不参与训练，设置blobs_lr=0；对于图像，读取数据尽量别用HDF5Layer
	
- 训练基本流程

	- 数据处理（转换格式）
	- 定义网络结构
	- 配置solver
	- 训练 
	
			> caffe train -solver solver.prototxt -gpu 0





	
	
	
	
	
	
	
	
	
	
	
	