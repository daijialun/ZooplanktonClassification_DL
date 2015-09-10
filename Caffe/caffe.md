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
        


## 其他说明

### Caffe源码

[**Caffe源码阅读路线：**](http://blog.csdn.net/kkk584520/article/details/41681085)

1. 从src/caffe/proto/caffe.proto开始，了解各类数据结构，主要是内存对象和序列化磁盘文件的一一队对应关系，了解如何从磁盘Load一个对象到内存，以及如何将内存对象Save到磁盘，中间的过程实现都是由protobuf自动完成的

2. 看头文件，不急于看cpp文件，理解整个框架。Caffe中类数目众多，但是条理清晰。在Testing时，最外层的类是Caffe::Net，包含了多个Caffe::Layer对象，Layer对象派生出神经网络多种不同层的类(DataLayer, ConvolutionLayer, InnerProductionLayer, AccurancyLayer)，每层都有相应的输入输出（Blob对象）以及层的参数（Blob对象）；Blob中包括了SyncedMemory对象，统一了CPU和GPU存储器。自顶向下去看这些类，结合理论知识很容易掌握使用方法。

3. 针对性地取看cpp和cu文件。一般而言，Caffe框架不需要修改，只需要增加新的层实现即可。如果想实现自己的卷积层，从ConvolutionLayer派生一个新类MyConvolutionLayer，然后将几个虚函数改成自己的实现即可。这一阶段主要关注点在算法，而不是源码本身。

4. 可编写各类工具，集成到Caffe内部。在tools/中有很多实用工具，可以根据需要修改。例如，从训练好的模型中抽取参数进行可视化，可用Python结合matplot实现

5. 最后，想更深层次学习，自己重新编写Caffe，重新构建自己的框架


	
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

### Caffe文件格式

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
		
- **model/bvlc_alexnet: **
    
    - bvlc_alexnet.caffemodel
    
        训练好的alexnet的caffe模型
      
    - solver.prototxt
    
        模型网络所求得解的值
        
    - train_val.prototxt
    
        网络结构以及各层详细配置
        
    - deploy.prototxt
    
        部署网络配置（与train_val.prototxt什么区别？？）
        
- **example/cifar10: **

    - cifar10_quick.prototxt
    
        网络结构以及各层详细配置
        
    - cifar10_quick_solver.prototxt
    
        网络模型所求得的解
        
    - cifar10_quick_solver_lr1.prototxt
    
        网络模型所求得的解（其基础learning rate不同，为0.0001）
        
    - get_cifar10.sh
    
        下载cifar10数据集
            
    - create_cifar10.sh
    
         创建cifar10，即转换cifar10为lmdb格式数据
         
    - train_quick.sh
    
            开始训练模型
            
    - cifar10_train_lmdb
    
            存放cifar10的lmdb数据
            
    - mean.binaryproto
    
        图像的均值信息
    
- **data/ilsvrc12: **

    - det_synset_words.txt
    
        定义序号对应的同义词
        
    - synsets.txt
    
          列出序号
              
    - synset_words.txt
    
        定义序号对应的同义词
        
    - imagenet_mean.binaryproto
    
        image_net图像的均值信息
        
    - train.txt
    
        训练图像对应的类别标号
        
    - test.txt
        
        测试图像对应的类别标号
        
    - val.txt

        验证图像对应的类别标号
        

	
### Caffe Model Zoo

- 模型可用来解决回归，大规模可视化分类，图像相似性等问题

- model zoo frameword：
	- 包含Caffe模型信息的标准格式
	- 从Github上上传/下载模型信息的工具，下载已训练好的二进制caffemodel文件
	- 分享模型信息的wiki页面
	
- 获得已训练的模型方式：

		> python script/download_model_binary.py <dirname>
		
	<dirname>具体如下：
	-model/bvlc_reference_caffenet
	AlexNet微小变动的版本
	
	- model/bvlc_alexnet
	- model/bvlc_reference_rcnn_ilsvrc13
	- model/bvlc_googlenet

- Caffe用户上传了许多community models在[wiki pages](https://github.com/BVLC/caffe/wiki/Model-Zoo)，可下载使用。

- **model可在github上的caffe/models/<model_name>页面中找到，并从readme.md提供的地址下载。**例如，[bvlc_alexnet.caffemodel](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)
	
	
	
	
	
	
	
	
	
	
	