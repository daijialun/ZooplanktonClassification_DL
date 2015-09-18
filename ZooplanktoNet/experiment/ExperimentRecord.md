# ZooplanktoNet Experiment Record

ZooplanktoNet相关实验记录

## Train

在create_imagenet.sh，只有"Create train lmdb"与"Create val lmdb"两个选项，而ImageNet总有train, val与test三个数据集。

在**命令行**中：
在caffe的train_val.prototxt文件中，只有两个模块include{ phase:TRAIN }与include{ phase:TEST }。其中，在include{ phase:TRAIN }中，使用train_lmdb；而在include{ phase:TEST }中，使用val_lmdb。由此可知，正常情况下，是将train_lmdb作为训练，val_lmdb作为测试。

在**DIGITS**中：
创建数据集时，也有train, val与test三个数据选项。但是，DIGITS在训练过程中，accuracy与loss都是使用val进行判断，因此必须判断test数据选项在DIGITS训练过程中的作用。应该了解test数据的使用：(1) 作为val，即*train_9460 images, val_1300 images*与*train_9460 images, val_25%, 无test*； (2) 作为test，即*train_9460 images, test_1300 images*

**对于zooplankton**，只有train与test。而本质上，train与test都是相同类别的图像，可以理解为，从整体图像中分出train与test图像，二者性质应没有差别。经过实验证明，如果将是*Train_9460_Val_0_Test_25%*或*Train_9460_Val_0_Test_1300*的情况，没有显示accuracy与loss(val)。

所以，我认为**正常情况下，train为9460 images，val为1300 images，不需设置test**但是具体情况，还是通过实验来验证。

### LeNet（单通道）

LeNet是训练MNIST数据的网络模型，默认情况下，其图像size设置为28x28，通道为grayscale。所以，实验可分为2组：(1) *train_9460，val_1300*；(2)*train_9460，val_1300，test_1300*。通过以下实验，可以**确认LeNet网络性能**，以及**创建数据时，是否需要test**。


#### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）

- 1) 数据集为*Zooplankton_1Channel_Origin_Train_9460_Val_1300_28x28*，表示原始数据集为Zooplankton，通道数为1，即灰度图像，图像未处理(origin)，train输入为9460张，val为输入的1300张测试图像，图像size为28x28。所训练的模型为*LeNet_1Channel_Origin_Train_9460_Val_1300_28x28*。

    **dataset:**
 
            Image Size: 28x28
            Image Type: GRAYSCALE
            Create DB(train): 9400 images
            Create DB(val): 1300 images
            Encoding: png 
        
    **Result:**

            accuracy(val): 63.4615%
            loss(val): 1.49338
            loss(train): 0.05301
           
    **accuracy(val)，loss(val)与loss(train)曲线波动正常**
     
- 2) 数据集为*Zooplankton_1Channel_Origin_Train_9460_Val_1300_Test_1300_28x28*，表示原始数据集为Zooplankton，通道数为1，即为灰度图像，图像未处理(origin)，train输入为9460张，val输入为1300张测试图像，test输入为1300张测试图像。所训练的模型为*LeNet_1Channel_Origin_Train_9460_Val_1300_Test_1300_28x28*。

    **dataset:**
 
            Image Size: 28x28
            Image Type: GRAYSCALE
            Create DB(train): 9460 images
            Create DB(val): 1300 images
            Create DB(test): 1300 images
            Encoding: png 
        
    **Result:**

            accuracy(val): 62.6923%
            loss(val): 1.52084
            loss(train): 0.0769752

     **accuracy(val)，loss(val)与loss(train)曲线波动正常**

- 3) 由1）和2）可知，当test图像作为Val时，Test有无并不产生影响，其accuracy几乎相等。但是通过`./build/tools/caffe test`检测训练模型的准确率，其中：

    - 1）*LeNet_1Channel_Origin_Train_9460_Val_1300_28x28*的结果：
    
                accuracy=0.6362
                loss=1.48629
                
     -2）*LeNet_1Channel_Origin_Train_9460_Val_1300_Test_1300_28x28*的结果：
     
                accuracy=0.6272
                loss=1.51305

因此，由以上可知，**在创建数据集时，不使用test，对结果基本没影响。**所以，应该使用**Train_9460，Val_1300/25%，无Test**较为合适。

#### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

#### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）


### AlexNet（单通道）

AlexNet是训练ILSVRC2012竞赛的网络模型，默认情况下，图像设置为256x256，通道数为3，但是由于zooplankton是单通道图像，因此在这使用AlexNet，只使用grayscale通道。另外，实验可分为2组：（1）train_9460，val_1300；（2）train_9460，val_1300。通过以下实验，**确认AlexNet网络性能**，以及**创建数据时，Val设置为25%或1300 images**。

#### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）

- 1) 数据集为*Zooplankton_1Channel_Origin_Train_9460_Val_1300_256x256*，表示原始数据集为Zooplankton，通道数为1，图像未处理(origin)，train输入为9460张，val为test的1300张，image size为256x256。所训练的模型为*AlexNet_1Channel_Origin_Train_9460_Val_1300_256x256*。

    **dataset:**
 
            Image Size: 256x256
            Image Type: GRAYSCALE
            Create DB(train): 9460 images
            Create DB(val): 1300 images
            Encoding: png 
        
    **Result:**

            accuracy(val): 78.7692%
            loss(val): 0.659638
            loss(train): 0.371415
           
    **accuracy(val)，loss(val)与loss(train)曲线波动变化正常**        
        
- 2) 数据集为*Zooplankton_1Channel_Origin_Train_9460_Val_25_256x256*，表示原始数据集为Zooplankton，通道数为1，图像未处理(origin)，train输入为9460张，val为test的1300张，image size为256x256。所训练的模型为*AlexNet_1Channel_Origin_Train_9460_Val_25_256x256*。

    **dataset:**
 
            Image Size: 256x256
            Image Type: GRAYSCALE
            Create DB(train): 7098 images
            Create DB(val): 2362 images
            Encoding: png 
        
    **Result:**

	        accuracy(val): 78.625%
            loss(val): 0.611972
            loss(train): 0.47833
           
    **accuracy(val)，loss(val)与loss(train)曲线波动变化正常** 
    
 - 3) 由1）和2）可知，当test图像作为Val时，其accuracy几乎相等。但是通过`./build/tools/caffe test`来检测所训练模型的准确率，其中：
 
    - 1)的网络模型*AlexNet_1Channel_Origin_Train_9460_Val_1300_256x256*的结果：
    
                accuracy=0.787
                loss=0.662758
                
       - 2)的网络模型*AlexNet_1Channel_Origin_Train_9460_Val_25_256x256*的结果：
   
                accuracy=0.7744
                loss=0.765725
   
    通过以上实验，可以得出，在创建数据是，应该使用**Train_9460，Val_1300，无Test**。
           
#### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

#### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

### CaffeNet（单通道）

#### 训练图像9460张（训练集）+ 测试图像1300张（测试集）

#### 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

#### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

### GoogleNet（单通道）

#### 训练图像9460张（训练集）+ 测试图像1300张（测试集）

#### 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

#### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）



## Finetune

### 3通道pre-trained模型

#### CaffeNet

##### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）

##### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

#### GoogleNet

##### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）

##### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

### 单通道pre-trained模型


## Other Work

### 3通道LeNet实验，size 256x256 与 28x28

- 1) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_25_256x256*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为输入的25%，2362张，image size为256x256。

    **dataset:**
 
            Image Size: 256x256
            Image Type: COLOR
            Create DB(train): 7098 images
            Create DB(val): 2363 images
            Encoding: png 
        
    **Result:**

            accuracy(val): 58.9583%
            loss(val): 2.49416
            loss(train): 0.21369
           
    **accuracy(val)，loss(val)与loss(train)曲线波动正常**

- 2) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_25_Test_1300_256x256*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val占据25%，test输入为1300张，image size为256x256。

    **dataset:**
 
            Image Size: 256x256
            Image Type: COLOR
            Create DB(train): 7098 images
            Create DB(val): 2362 images
            Create DB(test): 1300 images
            Encoding: png 
        
    **Result:**

            accuracy(val): 56.625%
            loss(val): 2.57342
            loss(train): 0.0854

    **accuracy(val)，loss(val)与loss(train)曲线波动变化正常**
    
          
- 3) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300_256x256*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为test的1300张，image size为256x256。

    **dataset:**
 
            Image Size: 256x256
            Image Type: COLOR
            Create DB(train): 9460 images
            Create DB(val): 1300 images
            Encoding: png 
        
    **Result:**

            accuracy(val): 7.6923%
            loss(val): 2.91502
            loss(train): 2.24022
           
    **accuracy(val)与loss(val)变化基本呈直线，没有显著变化；loss(train)在范围内震荡变化**
           

- 4) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300_Test_1300_256x256*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为test的1300张，test输入为1300张，image size为256x256。

    **dataset:**
 
            Image Size: 256x256
            Image Type: COLOR
            Create DB(train): 9460 images
            Create DB(val): 1300 images
            Create DB(test): 1300 images
            Encoding: png 
        
    **Result:**

	          accuracy(val): 7.6923%
            loss(val): 2.91502
            loss(train): 2.24022
           
    **accuracy(val)与loss(val)变化基本呈直线，没有显著变化；loss(train)在范围内震荡变化**
 
   
- 5) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300_28x28*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为输入的25%，2362张，image size为28x28。

    **dataset:**
 
            Image Size: 28x28
            Image Type: COLOR
            Create DB(train): 9460 images
            Create DB(val): 1300 images
            Encoding: png 
        
    **Result:**

            accuracy(val): 61.6923%
            loss(val): 1.73593
            loss(train): 0.058678
           
    **accuracy(val)，loss(val)与loss(train)曲线波动正常**
    
- 6)  数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_25_28x28*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为输入的25%，2362张，image size为28x28。

     **dataset:**
 
            Image Size: 28x28
            Image Type: COLOR
            Create DB(train): 7098 images
            Create DB(val): 2362 images
            Encoding: png 
        
     **Result:**

            accuracy(val): 71.0417%
            loss(val): 1.10647
            loss(train): 0.1872
           
     **accuracy(val)，loss(val)与loss(train)曲线波动正常**    

- 7) 由于LeNet的Intended image size为28x28(gray)，所以如果在处理数据集的过程中，将图像size调整为256x256的大小，其结果不是很理想，正确率最高为60%左右；但是如果将图像size调整为28x28的大小，其3channel的结果，准确率最高为72%左右。

- 8) 根据以上各实验结果，可以观察到：1）与 2）的共同点为train都为7098images，val都为25%，image size都为256x256，而test不同，1）无test，而2）有test为1300 images，二者的accuracy与loss都几乎相同，accuracy大约在58%左右；3）与4）的共同点为train都为9460images，val为测试的1300images，3）无test，而4）有test为1300 images，二者的accuracy与loss都几乎相同，但accuracy只有8%左右。有上述两点可知，当val为25%时，其正确率高于，val为test的1300images，高出大约50%。
 
 	因此可以得出：在LeNet的3通道训练中，val应该选择25%。所以用DIGITS训练时，在**LeNet**的3通道网络下，使用**train9460+val25%+test1300+image size28x28**比较合适。
       
       
### 单通道LeNet实验，size 256x256 与 28x28

- 1) 数据集为*Zooplankton_1Channel_Origin_Train_9460_Val_25_256x256*，表示原始数据集为Zooplankton，通道数为1，即灰度图像，图像未处理(origin)，train输入为9460张，val为输入的25%，2362张。

    **dataset:**
 
            Image Size: 256x256
            Image Type: GRAYSCALE
            Create DB(train): 7098 images
            Create DB(val): 2363 images
            Encoding: png 
        
    **Result:**

           accuracy(val): 58.9583%
           loss(val): 2.80499
           loss(train): 0.1193
           
    **accuracy(val)，loss(val)与loss(train)曲线波动正常**
     
- 2) 数据集为*Zooplankton_1Channel_Origin_Train_9460_Val_25_Test_1300_256x256*，表示原始数据集为Zooplankton，通道数为1，即为灰度图像，图像未处理(origin)，train输入为9460张，val占据25%，test输入为1300张。

    **dataset:**
 
            Image Size: 256x256
            Image Type: GRAYSCALE
            Create DB(train): 7098 images
            Create DB(val): 2362 images
            Create DB(test): 1300 images
            Encoding: png 
        
    **Result:**

           accuracy(val): 61.125%
           loss(val): 2.67403
           loss(train): 0.0199318

     **accuracy(val)，loss(val)与loss(train)曲线波动正常**
    
- 2) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300_Test_1300_256x256*，表示原始数据集为Zooplankton，通道数为1，即灰度图像，图像未处理(origin)，train输入为9460张，val为test的1300张，test输入为1300张。

    **dataset:**
 
            Image Size: 256x256
            Image Type: GRAYSCAL
            Create DB(train): 9460 images
            Create DB(val): 1300 images
            Create DB(test): 1300 images
            Encoding: png 
        
    **Result:**

           accuracy(val): 48.4615%
           loss(val): 3.97581
           loss(train): 0.02518
      
    **accuracy(val)，loss(val)与loss(train)曲线波动正常** 
         
- 3) 数据集为*Zooplankton_1Channel_Origin_Train_9460_Val_1300_256x256*，表示原始数据集为Zooplankton，通道数为1，即为灰度图像，图像未处理(origin)，train输入为9460张，val为test的1300张。

    **dataset:**
 
            Image Size: 256x256
            Image Type: GRAYSCALE
            Create DB(train): 9460 images
            Create DB(val): 1300 images
            Encoding: png 
        
    **Result:**

           accuracy(val): 47.9231%
           loss(val): 4.56747
           loss(train): 0.02716
           
    **accuracy(val)，loss(val)与loss(train)曲线波动正常**
                      
### 3通道AlexNet实验

- 1) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_25_256x256*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val占据25%，image size为256x256。

    **dataset:**
 
            Image Size: 256x256
            Image Type: COLOR
            Create DB(train): 7098 images
            Create DB(val): 2362 images
            Encoding: png 
        
    **Result:**

            accuracy(val): 80.4583%
            loss(val): 0.58033
            loss(train): 0.18893(0.466331)

    **accuracy(val)，loss(val)与loss(train)曲线波动变化正常**

- 2) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_25_Test_1300_256x256*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val占据25%，test输入为1300张，image size为256x256。

    **dataset:**
 
            Image Size: 256x256
            Image Type: COLOR
            Create DB(train): 7098 images
            Create DB(val): 2362 images
            Create DB(test): 1300 images
            Encoding: png 
        
    **Result:**

            accuracy(val): 79.4583%
            loss(val): 0.57679
            loss(train): 0.326187

    **accuracy(val)，loss(val)与loss(train)曲线波动变化正常**
        
- 3) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300_256x256*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为test的1300张，image size为256x256。

    **dataset:**
 
            Image Size: 256x256
            Image Type: COLOR
            Create DB(train): 9460 images
            Create DB(val): 1300 images
            Encoding: png 
        
    **Result:**

            accuracy(val): 79.2308%
            loss(val): 0.661074
            loss(train): 0.244173
           
    **accuracy(val)，loss(val)与loss(train)曲线波动变化正常**        
        
- 4) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300_Test_1300_256x256*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为test的1300张，test输入为1300张，image size为256x256。

    **dataset:**
 
            Image Size: 256x256
            Image Type: COLOR
            Create DB(train): 9460 images
            Create DB(val): 1300 images
            Create DB(test): 1300 images
            Encoding: png 
        
    **Result:**

	          accuracy(val): 78.4615%
            loss(val): 0.696236
            loss(train): 0.286798
           
    **accuracy(val)，loss(val)与loss(train)曲线波动变化正常**
    
### 单通道AlexNet实验

- 1) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_25_256x256*，表示原始数据集为Zooplankton，通道数为1，图像未处理(origin)，train输入为9460张，val占据25%，image size为256x256。

    **dataset:**
 
            Image Size: 256x256
            Image Type: GRAYSCALE
            Create DB(train): 7098 images
            Create DB(val): 2362 images
            Encoding: png 
        
    **Result:**

            accuracy(val): 79.2917%
            loss(val): 0.610655
            loss(train): 0.498804

    **accuracy(val)，loss(val)与loss(train)曲线波动变化正常**

- 2) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_25_Test_1300_256x256*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val占据25%，test输入为1300张，image size为256x256。

    **dataset:**
 
            Image Size: 256x256
            Image Type: GRAYSCALE
            Create DB(train): 7098 images
            Create DB(val): 2362 images
            Create DB(test): 1300 images
            Encoding: png 
        
    **Result:**

            accuracy(val): 81.75%
            loss(val): 0.548539
            loss(train): 0.313309

    **accuracy(val)，loss(val)与loss(train)曲线波动变化正常**
        
- 3) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300_256x256*，表示原始数据集为Zooplankton，通道数为1，图像未处理(origin)，train输入为9460张，val为test的1300张，image size为256x256。

    **dataset:**
 
            Image Size: 256x256
            Image Type: GRAYSCALE
            Create DB(train): 9460 images
            Create DB(val): 1300 images
            Encoding: png 
        
    **Result:**

            accuracy(val): 77.4615%
            loss(val): 0.706065
            loss(train): 0.300
           
    **accuracy(val)，loss(val)与loss(train)曲线波动变化正常**        
        
- 4) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300_Test_1300_256x256*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为test的1300张，test输入为1300张，image size为256x256。

    **dataset:**
 
            Image Size: 256x256
            Image Type: GRAYSCALE
            Create DB(train): 9460 images
            Create DB(val): 1300 images
            Create DB(test): 1300 images
            Encoding: png 
        
    **Result:**

	        accuracy(val): 78.6154%
            loss(val): 0.669017
            loss(train): 0.45559
           
    **accuracy(val)，loss(val)与loss(train)曲线波动变化正常**
    