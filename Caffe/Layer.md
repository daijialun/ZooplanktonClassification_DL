# Layers(对象)具体分析 

具体内容参照[Layer Catalogue](http://caffe.berkeleyvision.org/tutorial/layers.html)

## *Vision Layers*

### Convolution

- Layer type: Convolution

- CPU implementation: convolution_layer.cpp

- CUDA GPU implementation: convolution_layer.cu
 
### Pooling

- 池化

 - Layer type: Pooling

- CPU implementation: pooling_layer.cpp

- CUDA GPU implementation: pooling_layer.cu

### Local Response Normalization ( LRN )
 
- 局部响应归一化

- Layer type: LRN

- CPU implementation: lrn_layer.cpp

- CUDA GPU implementation: lrn_layer.cu

- Parameters:

    - norm_region: 选择对相邻通道间（ACROSS_CHANNELS）归一化还是通道内空间区域（WITHIN_CHANNEL）归一化，默认为ACROSS_CHANNELS
    
     - local_size: 对ACROSS表示需要求和的通道间数量；对WITHIN表示需要求和的区间区域的边长

    - 局部响应归一化完成“临近抑制”操作，对局部输入区域进行归一化


## *Loss Layers*

### Softmax

- Layer type: SoftmaxWithLoss

- 计算输入中，softmax的多项式逻辑损失，在数学上，提供了更稳定的梯度

### Sum-of-Squares / Euclidean

- Layer type: EuclideanLoss

### Hinge / Margin

- Layer type: HingeLoss

- CPU implementation: hinge_loss_layer.cpp

### Sigmoid Cross-Entropy

### Infogain

### Accuracy and Top-k

Accuracy对输出所对应的目标的准确度，通过分数表达。Accuracy实际上不是Loss，没有反向步骤。

## *Activation / Neuron Layers*

通常，Activation / Neuron Layers是元素级操作，将底层blob生成同规模的顶层blob

### ReLU / Rectified-Linear and Leaky-ReLU（常用）

- Layer type: ReLU

- CPU implementation: relu_layer.cpp

- CUDA GPU implementation: relu_layer.cu

### Sigmoid

- Layer type: sigmoid

- CPU implementation: sigmoid_layer.cpp

- CUDA GPU implementation: sigmoid_layer.cu

### TanH / Hyperbolic Tangent

- Layer type: TanH

- CPU implementation: tanh_layer.cpp

- CUDA GPU implementation: tanh_layer.cu

### Absolute Value
### Power
### BNLL


## *Data Layers*

数据通过data layers进入Caffe，data layers位于网络底层。数据可从内存中直接获取有效的数据库格式（LevelDB或LMDB）。如果效率不是主要因素时，可从硬盘中读取HDF5格式或者普通图片格式。

通常的输入预处理（mean subtraction, scaling, random cropping , mirroring）可通过*TransformationParameters*来使用。

### Database

- Layer type: Data

- Parameters:

    - backend: 选择LEVELDB或者LMDB
    
### In-Memory
### HDF5 Input
### ImageData
### Windows
### Dummy


## *Common Layers*

### Inner Product

- Layer type: TanH

- CPU implementation: inner_prouct_layer.cpp

- CUDA GPU implementation:  inner_prouct_layer.cu

### Splitting
### Flattening
### Reshape
### Concatenation
### Slicing
### Elementwise Operations
### Argmax
### Softmax
### Mean-Variance Normalization