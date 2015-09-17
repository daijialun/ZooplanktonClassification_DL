# ZooplanktoNet Experiment Record

ZooplanktoNet相关实验记录

## Train

### 3通道图像

#### LeNet

##### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）

- 1) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_25_Test_1300_256x256*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val占据25%，test输入为1300张，image size为256x256。

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

    **accuracy(val)，loss(val)与loss(train)曲线波动变化基本正常**
    
- 2) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300_Test_1300_256x256*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为test的1300张，test输入为1300张，image size为256x256。

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
           
    **accuracy(val)与loss(val)基本呈直线，基本没有变化；loss(train)在范围内震荡变化**
           
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
           
    **accuracy(val)，loss(val)与loss(train)都图形呈直线状，基本不改变**
           
- 4) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_25_256x256*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为输入的25%，2362张，image size为256x256。

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
   
-5)  数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300_28x28*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为输入的25%，2362张，image size为28x28。

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
    
-6)  数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_25_28x28*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val为输入的25%，2362张，image size为28x28。

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

- 8) 根据以上各实验结果，可以观察到：1) 与 4) 的共同点为train都为7098images，val都为25%，image size都为256x256，而test不同，二者的accuracy与loss都几乎相同，accuracy大约在58%左右；2) 与 3)的共同点为train都为9460images，val为测试的1300images，二者的accuracy与loss都几乎相同，accuracy只有8%左右。有上述两点可知，当val为25%时，其正确率高于，val为test的1300images，高出大约50%。
 
 	因此可以得出：在LeNet训练中，val应该选择25%，有1300images的test，正确率会比无test的高大约3%。所以用DIGITS训练时，在**LeNet**网络下，使用**train9460+val25%+test1300+image size28x28**比较合适。
           
           
##### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）



#### AlexNet(CaffeNet)

##### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）+ 图像转换为3通道lmdb 

- - 1) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_25_256x256*，表示原始数据集为Zooplankton，通道数为3，图像未处理(origin)，train输入为9460张，val占据25%，image size为256x256。

    **dataset:**
 
            Image Size: 256x256
            Image Type: COLOR
            Create DB(train): 7098 images
            Create DB(val): 2362 images
            Encoding: png 
        
    **Result:**

            accuracy(val): 56.625%
            loss(val): 2.57342
            loss(train): 0.0854

    **accuracy(val)，loss(val)与loss(train)曲线波动变化基本正常**

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

    **accuracy(val)，loss(val)与loss(train)曲线波动变化基本正常**
        
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
           
    **accuracy(val)，loss(val)与loss(train)都图形呈直线状，基本不改变**        
        
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
           
    **accuracy(val)与loss(val)基本呈直线，基本没有变化；loss(train)在范围内震荡变化**
    
           
##### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

#### GoogleNet

##### 训练图像9460张（训练集）+ 测试图像1300张（测试集）

##### 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

### 单通道图像

#### LeNet

##### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）


- 1) 数据集为*Zooplankton_1Channel_Origin_Train_9460_Val_25_Test_1300*，表示原始数据集为Zooplankton，通道数为1，即为灰度图像，图像未处理(origin)，train输入为9460张，val占据25%，test输入为1300张。

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
    
- 2) 数据集为*Zooplankton_3Channel_Origin_Train_9460_Val_1300_Test_1300*，表示原始数据集为Zooplankton，通道数为1，即灰度图像，图像未处理(origin)，train输入为9460张，val为test的1300张，test输入为1300张。

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
         
- 3) 数据集为*Zooplankton_1Channel_Origin_Train_9460_Val_1300*，表示原始数据集为Zooplankton，通道数为1，即为灰度图像，图像未处理(origin)，train输入为9460张，val为test的1300张。

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
                      
- 4) 数据集为*Zooplankton_1Channel_Origin_Train_9460_Val_25*，表示原始数据集为Zooplankton，通道数为1，即灰度图像，图像未处理(origin)，train输入为9460张，val为输入的25%，2362张。

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

- 5) 根据以上各实验结果，可以观察到：1) 与 4) 的共同点为train都为7098images，val都为25%，而test不同，二者的accuracy与loss都几乎相同；2) 与 3)的共同点为train都为9460images，val为测试的1300images，二者的accuracy与loss都几乎相同；当val为25%时，其正确率高于，val为test的1300images，且高出很多。
 
 	因此可以得出：在LeNet训练中，val应该选择25%，有1300images的test，正确率会比无test的高大约3%。所以用DIGITS训练时，使用**train9460+val25%+test1300**比较合适。
 	
 	另外，单通道的图像输入训练网络，与3通道的图像输入训练网络，其正确率非常接近，在60%左右，因此在**通道数量**方面，是没有显著差别的。但是如果通道为3的情况下，网络的参数更多，而且训练时间更长，对于资源的浪费更明显。因此，在LeNet中，应选择**单通道，原始图像，train9460+val25%+test1300**的数据输入训练网络比较合适。
 	
##### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

#### AlexNet

##### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）+ 图像转换为3通道lmdb 

##### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

#### CaffeNet

##### 1. 训练图像9460张（训练集）+ 测试图像1300张（测试集）+ 图像转换为3通道lmdb 

##### 2. 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

#### GoogleNet

##### 训练图像9460张（训练集）+ 测试图像1300张（测试集）

##### 训练图像9460+1300张（训练集+测试集） + 测试图像1300张（测试集）

##### 3. 训练图像（取中心处理）9460张（训练集）+ 测试图像（取中心处理）1300张（测试集）

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