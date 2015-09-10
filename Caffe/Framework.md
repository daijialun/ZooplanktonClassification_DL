# Caffe结构分析

Caffe结构分析参考[Tour](http://caffe.berkeleyvision.org/tutorial/)

## Blobs, Layers和Nets

## Forward和Backward

- forward主要是根据输入，通过推断，计算输出。Caffe将各层的计算组合实现模型功能。forward pass是自底向上的

- backward主要是根据学习的loss，计算梯度。Caffe通过自动微分，将各层的梯度反向推导得出整个模型的梯度，即反向传播。backward pass是自顶向下的

- `Net::Forward()`和`Ner::Backward()`执行各自的pass；`Layer::Forward()`和`Layer::Backward()`计算每一步

- `Solver`优化模型：首先用forward产生输出和loss；再用backward生成模型梯度，将梯度合并到weight update中，来最小化loss

- Solver, Net与Layer的分开，使Caffe模块化


## Loss

- loss function通过匹配参数设定（例如，当前网络权值）实现学习目标

- 网络中的loss是由forward计算的，每层取输入（bottom）blobs，产生输出（top）blobs。部分层的输出可用在loss function上。

- 对于多选一的分类问题，典型的loss function为`SoftmaxWithLoss`，如下：

        layer{
            name:"loss"
            type:"SoftmaxWithLoss"
            bottom:"pred"
            bottom:"label"
            top:"loss"
            }

## Solver

- solver通过调整网络，前向的推断和后向的梯度，通过减小loss实现参数更新

- Solver负责监督优化和参数更新；Net负责生成loss与梯度

- solvers有SGD, ADAGRAD和NESTEROV

- The solver:

	- 创建网络来学习，测试网络来评价
	- 通过调用forward/backward来迭代优化，更新参数
	- 周期性评价该测试网络
	- 通过优化来快速展示model与slover state
	
- Where each iteration

	- forward计算输出与loss
	- backward计算梯度
	- 根据solver方法，将梯度合并到参数更新中
	- 根据learning, history和method，更新solver state
	
- 实际的weight更新是由slover实现的，再应用到net参数中

- solver中weight的快照以及其state，分别由`Solver::Snapshot()`与`Solver::qSnapshotSolverState()`实现。weight的快照允许训练从某一点继续，由`Solver::Restore()`实现

- weights保存没有拓展名，而solver state用solverstate拓展名保存。二者都有`_iter_N`后缀作为迭代次数快照

- snapshotting是在solver定义的prototxt中
            
## Interface

Caffe有通过三种接口进行使用：

- command line: **cmdcaffeb**

- python: **pycaffe**

- matlab: **matcaffe**

** Command Line: **

1. ** Training：** `train caffe`可从零开始训练模型，从保存的snapshots继续训练，以及fine-tune用于新数据与任务

    - 所有训练都需要solver配置通过`-solver solver.prototxt`参数
    - 继续训练需要`-snapshot model_iter_1000.solverstate`参数加载solver snapshot
    - fine-tune需要`-weights model.caffemodel`参数完成模型初始化
    
2. **Testing：** `caffe test`运行模型的测试模块，用分数输出网络结果。网络架构定义来输出准确率或loss。per-batch输出后，grand average最后输出

3. **Benchmarking（参照）：** `caffe time`通过时间和同步，作为层到层的模型执行参考。可用来检测系统星河和衡量模型时间

4. **Diagnostics（诊断）：** `caffe device_query`在多GPU机器上运行时，输出参考以及检查序号

### Data：Ins and Outs

- 数据通过Blobs进入caffe；Data Layers加载来自Blob的数据或者保存Blob数据为其他格式

- mean-subtraction和feature-scaling通过data layer配置完成

- 可通过加入新的数据层完成新的输入类型，Net的其余部分由layer目录的其他模块组成

- data layer定义：

        layer{
            name:"mnist"
            type:"Data"
            top:"data"
            top:"label"
            data_param{
            source:"examples/mnist/mnist_train_lmdb"
            backend:LMDB
            batch_size:64}
            transform_param{
            scale:0.0039}
            }
            
    - Tops和Bottoms
    
          data layer使top blobs成为输出数据；由于没有输入，则没有botoom blobs
          
    - Data和Label
            
        data layer至少有一个top叫data，一个次top叫label；二者都生成blobs，但是没有内在联系；（data，label）是为了分类模型的简便性
        
    - Transformations
    
        通过转换信息，将数据预处理参数化
        
    - Prefetching
    
         当Net计算当前batch时，data layer于后台操作，取下一个batch
         
     - Multiple inputs
     