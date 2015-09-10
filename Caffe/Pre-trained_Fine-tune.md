
# Pre-traied和Fine-tune实现

通过将针对其他问题所训练好的模型，即pre-train模型；进行一点的修改，用来解决我们目前的问题，即fine-tune。

## Caffe Document #1

以下内容主要参考了[DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.gc2fcdcce7_216_408)	
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

## Caffe Document #2

以下内容主要参考了 [Fine-tuning for style recognition](http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html)

- Fine-tuning:
    - 使用已学习的模型
    - 改进模型其框架
    - 继续训练已学习的模型权值
    
- 这个例子主要是将CaffeNet模型，通过fine-tune应用在另一个不同的数据集上，Flickr Style，实现预测图像风格的功能而不是物体分类

- Style数据集的Flickr-sourced图像，看起来与ImageNet数据集很相似，而`bylc_reference_caffenet`就是用ImageNet训练的。由于这个模型适用于物体分类，因此使用其架构作为style分类器

- 目前有80,000张图像进行训练，所以我们从已学习了1,000,000张图像的参数开始，需要时进行fine-tune

- 在命令行模式中，对`caffe train`提供`weights`参数，pretrained的权值就会加载进模型中，通过各层名字进行匹配

- 由于只需要预测20类而非1000类，只需要改变模型中最后一层。则在prototxt文件中，将最后一层名字由`fc8`改为`fc8_flickr`。由于在`bylc_reference_caffenet`中，没有与`fc8_flickr`匹配的层，则这层就会从随机权值开始训练

- 在solver prototxt中，降低整体learning rate，`base_lr`与提升新引入层的`blobs_lr`。即让新的层学习的很快，但是剩下的层学习得很慢

- 实现步骤：
    1. 通过script，可以下载数据小的子集，并分为训练与验证集
            
            > python example/finetune_flickr_style/assemble_data.py --workers=-1 --images=2000 --seed 831486
            
     script下载图像，将train/val文件写进data/flickr_style
         
     2.  得到ImageNet的mean文件
     
            > ./data/ilsvrc12/get_ilsvrc_aux.sh
            
    3. 下载ImageNet-trained模型
    
            > ./script/download_model_binary.py module/bvlc_reference_caffenet
            
            
         -gpu 0 表示使用CPU模式；-gpu all表示使用全部GPU
         
- `caffe train`可以从零开始训练模型，也可以从保存的snapshots继续训练

    - 所有训练都需要solver配置，加入-solver参数
    
            - solver solver.prototxt
            
    - 从断点继续训练需要solverstate文件，加入-snapshot参数，不需要-weights参数
    
            - snapshot model_iter_N.solverstate
            
   - 使用fine-tune需要caffemodel模型文件，加入-weights参数
   
            - weights model.caffemodel
            
### Zooplankton的fine-tune实现


