# AlexNet

## 论文分析

### Abstract

- 在ILSVRC-2010竞赛上，该组训练了一个大型的、有深度的卷积神经网络，用来将1.2million的高分辨率图像分为1000个不同的种类。**在测试数据上，**top-1和top-5的错误率达到37.5%和17.0%。这个神经网络有60,000,000参数和650,000个神经元。

- 在ILSVRC 2012竞赛上，稍微调整这个模型，取得top-5错误率为15.3%。

- **这个神经网络由**

	- 5层卷积层

	- 3层全连接层
	
	- 1层1000-way的softmax层
	
### Introduction

- **论文贡献：**
	- 针对ILSVRC2010和ILSVRC2012竞赛，训练了一个大型的卷积神经网络，达到当时最好的效果
	
	- 在GPU上实现了高性能的2D卷积，以及网络中的其他操作
	
	- 包含一些提高性能和减少训练时间的方法，以及防止过拟合的技术
	
- 网络规模主要受限于

	- GPU存储容量
	
	- 训练时间
	
- 提升实验结果

	- 更快的GPU
	
	- 更大的数据集
	
### 数据集

- ILSVRC是ImageNet的一个子集，有1000个种类，每个种类中有1000张图像，大概有12,000,000张训练图像，50,000张验证图像和150,000张测试图像

- ILSVRC2010的测试集labels是可用的，所以主要是用这个进行实验

- ILSVRC2012也有实验，但是测试集labels不可用的

- 将数据集图片的尺寸规范为256x256

- 只有subtract the mean activity，没有其他的pre-process

### Architecture

**以下为网络架构的特点，按重要性排序**

- ReLU Nonlinearity

	- 用梯度下降法，饱和非线性比非饱和非线性的训练时间少
	
	- ReLU(Rectified Linear Units) 激活函数，AlexNet中使用了非线性ReLU
	
	- 在深度卷积网络中，用ReLU训练比*tanh*单元快好几倍
	
	- 用传统的饱和神经模型，将无法解决如此大规模的神经网络
	
- Training on Multiple GPUs

	- GPU容量会限制网络的最大尺寸，可能会出现用来训练网络的数据足够大，但是GPU却无法处理
	
	- 目前CPU都支持cross-GPU parallelization技术
	
	- trick：GPU只在特定的层通信，每个GPU各有一半的kernels
	
	- GPU使top-1和top-5的错误率减少了1.7%和1.2%
	
	- 两个GPU训练时间比一个GPU短

- Local Response Normalization

	- ReLU具有不需要防止饱和的input normalization的性质

	- Response normalization使top-1和top-5的错误率减少了1.4%和1.2%

- Overlapping Pooling

	- 这网络使用的pooling是重叠的
	
	- 使用overlap overfit可抑制过拟合
	
- Overall Architecture

	- 网络最大化多项式逻辑回归，即在预测分布的情况下，求最大化训练案例中正确label的对数概率

### Reducing Overfitting

- Data Augmentation

	- 用label-preserving transformations来扩大数据集是最简单和通用的方法来降低
	
	- 第一种方式：image translation和horizontal reflection
	
		- 从256x256图像中，提取224x224 patches

		- 通过这种方式，降低了过拟合		
	- 第二种方式：改变训练图像中RGB通道的值

		- 使用PCA方法
		
		- 从自然图像中提取了重要性质，物体的统一性不随光照的强度和颜色而改变
		
- Dropout

	- 对每个隐含层的输出用50%的概率置0。置0的神经元不参加前向和后向的传导。
	
	- 用dropout，尽管每次神经网络是不同框架，但是这些框架共享权值
	
	- 减少了神经元之间的依赖性
	
	- dropout加倍了收敛需要的迭代次数
	
### Details of learning

- 小数量的权值衰减对模型的学习是很重要的，即weight decay不仅仅是regularizer，还降低了模型的训练错误率。

- 在每一层初始化权值，用偏差为0.01的zero-mean Gaussian。用constant=1初始化2，4和5层卷积以及全连接层。通过对ReLU提供整输入，加速了learning。用constant=0初始化在剩下的层













